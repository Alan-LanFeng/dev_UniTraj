from typing import Any, List, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from unitraj.models.abstract_agent import AbstractAgent
from unitraj.models.transfuser.transfuser_config import TransfuserConfig
from unitraj.models.transfuser.transfuser_model import TransfuserModel
from unitraj.models.transfuser.transfuser_callback import TransfuserCallback
from unitraj.models.transfuser.transfuser_loss import transfuser_loss
from unitraj.models.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from unitraj.utils.dataclasses import SensorConfig
from unitraj.models.abstract_agent import AbstractFeatureBuilder, AbstractTargetBuilder
from unitraj.models.transfuser.transfuser_config import TransfuserConfig
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
import wandb

import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grl(x, lambda_=1.0):

    return GradientReversalFunction.apply(x, lambda_)
    
class DomainDiscriminator(nn.Module):
    def __init__(self,
                 in_channels=512,
                 hidden_dim=256,
                 feature_size=(8, 8),
                 num_groups=32,
                 num_layers=4):

        super(DomainDiscriminator, self).__init__()
        H, W = feature_size
        # build a stack of convolutional blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(num_groups, hidden_dim),
                nn.ReLU(inplace=True)
            ))
        # final classifier: flatten all features → 1 logit
        self.classifier = nn.Linear(hidden_dim * H * W, 1)
        # internal loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sr_feat: torch.Tensor, tg_feat: torch.Tensor,
                lambda_grl: float = 1.0):


        bs_src = sr_feat.size(0)
        bs_tgt = tg_feat.size(0)
        domain_labels = torch.cat([
            torch.zeros(bs_src, 1, device=sr_feat.device),
            torch.ones (bs_tgt, 1, device=sr_feat.device)
        ], dim=0)

        x = torch.cat([sr_feat, tg_feat], dim=0)

        x = grl(x, lambda_grl)
        for block in self.blocks:
            x = block(x)

        x = x.view(x.size(0), -1)
  
        logits = self.classifier(x)

        loss = self.criterion(logits, domain_labels)


        # logits_src = logits[:bs_src]
        # logits_tgt = logits[bs_src:]
        return loss


class PearsonCorrLoss4D(nn.Module):


    def __init__(self, eps: float = 1e-8, mode: str = 'global', reduction: str = 'mean'):

        super().__init__()
        assert mode in ('global', 'channel')
        assert reduction in ('mean', 'sum', 'none')
        self.eps = eps
        self.mode = mode
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        if self.mode == 'global':
            # Flatten to (B, C*H*W)
            xf = x.view(B, -1)
            yf = y.view(B, -1)

            # Subtract mean
            mx = xf.mean(dim=1, keepdim=True)
            my = yf.mean(dim=1, keepdim=True)
            xm = xf - mx
            ym = yf - my

            # Covariance and standard deviations
            cov = (xm * ym).sum(dim=1) / (C*H*W)
            sx = torch.sqrt((xm**2).sum(dim=1) + self.eps)
            sy = torch.sqrt((ym**2).sum(dim=1) + self.eps)

            # Pearson r and loss
            r = cov / (sx * sy + self.eps)    # shape: (B,)
            loss = 1 - r                      # shape: (B,)

        else:  # channel‐wise
            # Flatten spatial dims to (B, C, H*W)
            xf = x.view(B, C, -1)
            yf = y.view(B, C, -1)

            # Subtract per‐channel mean
            mx = xf.mean(dim=2, keepdim=True)  # (B, C, 1)
            my = yf.mean(dim=2, keepdim=True)
            xm = xf - mx
            ym = yf - my

            # Covariance and std per channel
            cov = (xm * ym).sum(dim=2) / (H * W)   # (B, C)
            sx = torch.sqrt((xm**2).sum(dim=2) + self.eps)  # (B, C)
            sy = torch.sqrt((ym**2).sum(dim=2) + self.eps)  # (B, C)

            r = cov / (sx * sy + self.eps)         # (B, C)
            loss = 1 - r                           # (B, C)

            # Average over channels → (B,)
            loss = loss.mean(dim=1)

        # Reduction over batch
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss  # shape: (B,)

class ProjHead(nn.Module):

    def __init__(self, in_channels=384, out_channels=512):
        super().__init__()
        # First conv: downsample 64→32, expand channels 384→512
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=2, padding=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=2, padding=1, bias=True
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: [B, 384, 64, 64]
        x = self.conv1(x)   # [B, 512, 32, 32]
        x = self.relu(x)

        x = self.conv2(x)   # [B, 512, 16, 16]
        x = self.relu(x)

        x = self.pool(x)    # [B, 512,  8,  8]
        return x

class TransfuserLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, config):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()

        self.agent = TransfuserAgent(config)

        self.use_dino_feat = config.get('use_dino_feat', False)
        self.distil_loss_weight = config.get('distill_loss_weight', 1.0)
        self.use_adv = config.get('use_adv', False)

        if self.use_dino_feat:
            self.proj_head = ProjHead(384, 512)
            self.fm_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        if self.use_adv: # Unfinished no gpu
            self.domain_discriminator = DomainDiscriminator(
                in_channels=512,
                hidden_dim=256, 
                feature_size=(8, 8), 
                num_groups=32, 
                num_layers=4)
            self.lambda_grl = config.get('lambda_grl', 0.1)
        
        self.distill_loss_type = config.get('distill_loss_type', 'mse')
        if self.distill_loss_type == 'mse':
            self.ditill_loss = nn.MSELoss()
        elif self.distill_loss_type == 'l1':
            self.ditill_loss = nn.L1Loss()
        elif self.distill_loss_type == 'pearson':
            self.ditill_loss = PearsonCorrLoss4D(mode='global', reduction='mean')
        else:
            raise ValueError(f"Unknown distillation loss: {self.distil_loss}")
        


    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        real_valid_mask = features['real_valid_mask']
        feature_render = features['camera_feature'][real_valid_mask].detach().clone()
        status_render = features['status_feature'][real_valid_mask].detach().clone()

        features['camera_feature'][real_valid_mask] = features['camera_feature_real'][real_valid_mask]
        prediction = self.agent.forward(features)
        loss = self.agent.compute_loss(features, targets, prediction)

        camera = features['camera_feature'][0].permute(1, 2, 0).cpu().numpy()
        ego_status = features['status_feature'][0].cpu().numpy()
        pred_traj = prediction['trajectory'][0].detach().cpu().numpy()[:, :2]
        gt_traj = targets['trajectory'][0].cpu().numpy()[:, :2]
        

        if real_valid_mask.any():
            with torch.no_grad():
                if self.use_dino_feat:
                    features['camera_feature'] = feature_render
                    features['status_feature'] = status_render
                    camera_img = features['camera_feature']
                    dino_feat = self.fm_model.get_intermediate_layers(F.interpolate(camera_img, size=(896, 896), mode="bilinear"), reshape=True)[0] # TODO check interplot; [41, 384, 64, 64]
                            
                else:
                    features['camera_feature'] = feature_render
                    features['status_feature'] = status_render
                    prediction_render = self.agent.forward(features)
                    render_bev_feature = prediction_render['bev_feature']


            if self.use_dino_feat:
                render_bev_feature = self.proj_head(dino_feat.detach()) # [41, 512, 8, 8] 
                real_bev_feature = prediction['bev_feature_raw'][real_valid_mask]
            else:
                real_bev_feature = prediction['bev_feature'][real_valid_mask]

            if self.distill_loss_type == 'pearson': # TODO fix bug
                render_bev_feature = prediction_render['bev_feature_raw'][real_valid_mask]
                real_bev_feature = prediction['bev_feature_raw'][real_valid_mask]

            
            loss_render = self.ditill_loss(render_bev_feature, real_bev_feature)


            ade_real = torch.mean(torch.norm(prediction['trajectory'][real_valid_mask][...,:2] - targets['trajectory'][real_valid_mask][...,:2], dim=-1))
            self.log(f"{logging_prefix}/ade_real", ade_real, batch_size=real_valid_mask.sum(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            loss+=loss_render
            self.log(f"{logging_prefix}/loss_render", loss_render, batch_size=real_valid_mask.sum(), on_step=False, on_epoch=True, prog_bar=True,
                     sync_dist=True)
        render_mask = ~real_valid_mask
        if render_mask.any():
            ade_render = torch.mean(torch.norm(prediction['trajectory'][render_mask][...,:2] - targets['trajectory'][render_mask][...,:2], dim=-1))
            self.log(f"{logging_prefix}/ade_render", ade_render, batch_size=render_mask.sum(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        ade = torch.mean(torch.norm(prediction['trajectory'][...,:2] - targets['trajectory'][...,:2], dim=-1))
        self.log(f"{logging_prefix}/ade", ade, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        # camera = features['camera_feature'][0].permute(1, 2, 0).cpu().numpy()
        # ego_status = features['status_feature'][0].cpu().numpy()
        # pred_traj = prediction['trajectory'][0].detach().cpu().numpy()[:, :2]
        # gt_traj = targets['trajectory'][0].cpu().numpy()[:, :2]

        if self.global_step % 1000 == 0 and False:
            # 创建图像
            # 创建两个并列子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # ==== 子图1：Camera 图像 + 状态 ====
            ax1.imshow(camera)
            ax1.set_title("Camera View")
            ax1.axis('off')

            # 在图像上显示 ego_status 数值
            status_text = "\n".join([f"{i}: {v:.2f}" for i, v in enumerate(ego_status)])
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax1.text(5, 20, status_text, fontsize=10, verticalalignment='top', bbox=props)

            # ==== 子图2：轨迹图 ====
            ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'ro-', label="Predicted Trajectory")
            ax2.plot(gt_traj[:, 0], gt_traj[:, 1], 'go-', label="Ground Truth Trajectory")

            # 可选：加编号标注
            for i in range(len(pred_traj)):
                ax2.annotate(str(i), (pred_traj[i, 0], pred_traj[i, 1]), color='red')
                ax2.annotate(str(i), (gt_traj[i, 0], gt_traj[i, 1]), color='green')

            ax2.set_title("Trajectory")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.legend()
            ax2.grid(True)
            ax2.axis('equal')  # 保持坐标轴比例一致
            # 保存为图像并上传到 wandb
            wandb.log({f"{logging_prefix}/trajectory_visualization": [wandb.Image(fig)]})

        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        self_config=TransfuserConfig,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling)

        self._config = self_config
        self._lr = config.learning_rate

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(self._trajectory_sampling, self_config)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # NOTE: Transfuser only uses current frame (with index 3 by default)
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=history_steps if not self._config.latent else False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]
