from typing import Any, List, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from unitraj.models.diffusiondrive.transfuser_config import TransfuserConfig

from unitraj.models.diffusiondrive.transfuser_model_v2 import V2TransfuserModel as TransfuserModel

from unitraj.models.diffusiondrive.transfuser_callback import TransfuserCallback
from unitraj.models.diffusiondrive.transfuser_loss import transfuser_loss
from unitraj.models.diffusiondrive.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from unitraj.models.diffusiondrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from torch import Tensor
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Dict, Tuple
import wandb

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, config):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.config = config
        self.agent = TransfuserAgent(config)

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        real_valid_mask = features['real_valid_mask']
        batch_size = real_valid_mask.shape[0]

        camera_real = features['camera_feature_real'][real_valid_mask]
        status_real = features['status_feature'][real_valid_mask]
        target_traj_real = targets['trajectory'][real_valid_mask]

        if self.config['real_only']:
            features = {}
            features['camera_feature'] = camera_real
            features['status_feature'] = status_real
            targets['trajectory'] = target_traj_real
            prediction = self.agent.forward(features,targets)
            loss_dict = self.agent.compute_loss(features, targets, prediction)
            loss = loss_dict['loss']
            ade_real = torch.mean(
                torch.norm(prediction['trajectory'][:, :, :2] - targets['trajectory'][:, :, :2],
                           dim=-1))
            self.log(f"{logging_prefix}/ade_real", ade_real, batch_size=real_valid_mask.sum(), on_step=False,
                     on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            all_feature = torch.cat([features['camera_feature'], camera_real], dim=0)
            all_status = torch.cat([features['status_feature'], status_real], dim=0)
            all_target_traj = torch.cat([targets['trajectory'], target_traj_real], dim=0)
            targets['trajectory'] = all_target_traj
            features = {}
            features['camera_feature'] = all_feature
            features['status_feature'] = all_status

            prediction = self.agent.forward(features,targets)

            loss_dict = self.agent.compute_loss(features, targets, prediction)
            loss = loss_dict['loss']
            if real_valid_mask.any():
                bev_feature = prediction['bev_feature']
                render_bev = bev_feature[:batch_size][real_valid_mask].detach()
                real_bev = bev_feature[batch_size:]
                loss_render = F.mse_loss(render_bev, real_bev)
                loss += 5 * loss_render
                self.log(f"{logging_prefix}/loss_render", loss_render, batch_size=real_valid_mask.sum(), on_step=False,
                         on_epoch=True, prog_bar=True,
                         sync_dist=True)
            ade_render = torch.mean(
                torch.norm(prediction['trajectory'][:batch_size, :, :2] - targets['trajectory'][:batch_size, :, :2],
                           dim=-1))
            self.log(f"{logging_prefix}/ade_render", ade_render, batch_size=batch_size, on_step=False,
                     on_epoch=True,
                     prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)




        camera = features['camera_feature'][0].permute(1, 2, 0).cpu().numpy()
        ego_status = features['status_feature'][0].cpu().numpy()
        pred_traj = prediction['trajectory'][0].detach().cpu().numpy()[:, :2]
        gt_traj = targets['trajectory'][0].cpu().numpy()[:, :2]
        # if self.global_step % 1000 == 0 and self.global_rank == 0:
        #     # 创建图像
        #     # 创建两个并列子图
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        #
        #     # ==== 子图1：Camera 图像 + 状态 ====
        #     ax1.imshow(camera)
        #     ax1.set_title("Camera View")
        #     ax1.axis('off')
        #
        #     # 在图像上显示 ego_status 数值
        #     status_text = "\n".join([f"{i}: {v:.2f}" for i, v in enumerate(ego_status)])
        #     props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        #     ax1.text(5, 20, status_text, fontsize=10, verticalalignment='top', bbox=props)
        #
        #     # ==== 子图2：轨迹图 ====
        #     ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'ro-', label="Predicted Trajectory")
        #     ax2.plot(gt_traj[:, 0], gt_traj[:, 1], 'go-', label="Ground Truth Trajectory")
        #
        #     # 可选：加编号标注
        #     for i in range(len(pred_traj)):
        #         ax2.annotate(str(i), (pred_traj[i, 0], pred_traj[i, 1]), color='red')
        #         ax2.annotate(str(i), (gt_traj[i, 0], gt_traj[i, 1]), color='green')
        #
        #     ax2.set_title("Trajectory")
        #     ax2.set_xlabel("X")
        #     ax2.set_ylabel("Y")
        #     ax2.legend()
        #     ax2.grid(True)
        #     ax2.axis('equal')  # 保持坐标轴比例一致
        #     # 保存为图像并上传到 wandb
        #     wandb.log({f"{logging_prefix}/trajectory_visualization": [wandb.Image(fig)]})


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
        self._lr = config.lr

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(self_config)
        self.init_from_pretrained()

    def init_from_pretrained(self):
        # import ipdb; ipdb.set_trace()
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
            
            # Load state dict and get info about missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")
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
            lidar_pc=False,
        )


    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features,targets=targets)
        
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._transfuser_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._transfuser_model.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [TransfuserCallback(self._config)]
