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



class TransfuserLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, config):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()

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
        if real_valid_mask.any():
            camera_real = features['camera_feature_real'][real_valid_mask]
            status_real = features['status_feature'][real_valid_mask]
            target_traj_real = targets['trajectory'][real_valid_mask]
            all_feature = torch.cat([features['camera_feature'], camera_real], dim=0)
            all_status = torch.cat([features['status_feature'], status_real], dim=0)
            all_target_traj = torch.cat([targets['trajectory'], target_traj_real], dim=0)
            targets['trajectory'] = all_target_traj
            features = {}
            features['camera_feature'] = all_feature
            features['status_feature'] = all_status

        prediction = self.agent.forward(features)
        loss = self.agent.compute_loss(features, targets, prediction)

        if real_valid_mask.any():
            bev_feature = prediction['bev_feature']
            render_bev = bev_feature[:batch_size][real_valid_mask].detach()
            real_bev = bev_feature[batch_size:]
            loss_render = F.mse_loss(render_bev, real_bev)
            loss+=loss_render
            ade_real = torch.mean(torch.norm(prediction['trajectory'][batch_size:,:,2] - targets['trajectory'][batch_size:,:,2], dim=-1))
            self.log(f"{logging_prefix}/ade_real", ade_real, batch_size=real_valid_mask.sum(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss_render", loss_render, batch_size=real_valid_mask.sum(), on_step=False, on_epoch=True, prog_bar=True,
                     sync_dist=True)

        ade_render = torch.mean(torch.norm(prediction['trajectory'][:batch_size,:,2] - targets['trajectory'][:batch_size,:,2], dim=-1))
        self.log(f"{logging_prefix}/ade_render", ade_render, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
