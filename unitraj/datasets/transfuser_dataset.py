from .base_dataset import BaseDataset
from unitraj.utils.dataclasses import get_agent_input
from unitraj.datasets.common_utils import perturb_positions,derive_heading_from_positions,blend_and_smooth_heading , recompute_velocity, normalize_angle
import numpy as np
from unitraj.models.transfuser.transfuser_features import TransfuserFeatureBuilder,TransfuserTargetBuilder
from unitraj.models.transfuser.transfuser_config import TransfuserConfig
import torch
from unitraj.utils.renderer import ScenarioRenderer, save_as_video

import os

class TransfuserDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        self._feature_builders = [TransfuserFeatureBuilder(TransfuserConfig)]
        self._target_builders = [TransfuserTargetBuilder(TransfuserConfig)]
        self.renderer = ScenarioRenderer()
        super().__init__(config, is_validation)



    def preprocess(self, scenario):
        return scenario

    def process(self, internal_format):
        sdc_id = internal_format['metadata']['sdc_id']
        tracks = internal_format['tracks']
        sdc_track = tracks[sdc_id]
        driving_command = internal_format['driving_command']

        camera_data_real = internal_format['real_camera']
        camera_data_render = internal_format['rendered_camera']
            
        data_len = internal_format['length']

        results = []
        sdc_pos = sdc_track['state']['position'][...,:2]
        sdc_heading = sdc_track['state']['heading'][...,np.newaxis]
        sdc_feature = np.concatenate([sdc_pos, sdc_heading], axis=-1)
        sdc_feature = [sdc_feature[i] for i in range(0, data_len, 5)]
        sdc_feature = np.stack(sdc_feature, axis=0)
        ego_dynamics = internal_format['ego_dynamics']
        dynamic_feature = [
            np.concatenate([ego_dynamics[i]['velocity'], ego_dynamics[i]['acceleration']], axis=-1) for i in
            range(len(ego_dynamics))]
        dynamic_feature = np.stack(dynamic_feature, axis=0)
        data_len = sdc_feature.shape[0]

        for current_index in range(3,data_len):
            max_future_index = current_index + 8
            if max_future_index >= data_len:
                break
            total_index = np.arange(current_index-3, max_future_index+1)
            past_index = total_index[:4]

            sdc_feature_raw = sdc_feature[total_index].copy()
            sdc_pos, sdc_heading = sdc_feature_raw[:, :2], sdc_feature_raw[:, 2:3]

            # Normalize position by translating to the origin (t=0)
            sdc_pos_norm = sdc_pos - sdc_pos[3]  # shape (T, 2)

            sdc_heading_norm = normalize_angle(sdc_heading - sdc_heading[3])  # shape (T,)

            # Get rotation matrix to align to heading at t=0
            theta0 = sdc_heading[3,0]
            cos_t, sin_t = np.cos(-theta0), np.sin(-theta0)
            R = np.array([[cos_t, -sin_t],
                          [sin_t, cos_t]])  # shape (2, 2)
            # Rotate position, velocity, acceleration into ego frame at t=0
            sdc_pos_norm = sdc_pos_norm @ R.T  # shape (T, 2)
            # Rebuild feature: pos (normalized), heading (normalized), vel (rotated), acc (rotated)
            sdc_feature_t = np.concatenate([sdc_pos_norm, sdc_heading_norm], axis=-1)
            command = [driving_command[i] for i in past_index]
            sdc_feature_t = np.concatenate([sdc_feature_t, dynamic_feature[total_index]], axis=-1)
            sdc_past_feature = sdc_feature_t[:4]
            sdc_future_feature = sdc_feature_t[4:]

            is_monotonic = self.is_monotonic_trajectory(sdc_feature_t, threshold=self.config['constant_velocity_threshold'])
            if is_monotonic:
                continue
            used_cameras = self.config['used_cameras']


            features_render = {}
            for builder in self._target_builders:
                features_render.update(builder.compute_targets(sdc_future_feature[:,:3]))
            camera = [camera_data_render[i] for i in past_index]
            agent_input = get_agent_input(sdc_past_feature, command, camera, used_cameras)

            for builder in self._feature_builders:
                features_render.update(builder.compute_features(agent_input))

            features = {}
            try:
                camera = [camera_data_real[i] for i in past_index]
                agent_input = get_agent_input(sdc_past_feature, command, camera, used_cameras)
                for builder in self._feature_builders:
                    features.update(builder.compute_features(agent_input))
                # img = augment_image(features['camera_feature'])
                # import matplotlib.pyplot as plt
                # plt.imshow(img.transpose(1, 2, 0))
                # plt.axis('off')
                # plt.show()
                features_render['camera_feature_real'] = features['camera_feature']
                features_render['real_valid_mask'] = True
            except:
                features_render['camera_feature_real'] = np.zeros_like(features_render['camera_feature'])
                features_render['real_valid_mask'] = False
            #features_render['camera_path'] = camera[3]['CAM_F0']
            results.append(features_render)

        return internal_format, results

    def is_monotonic_trajectory(self, sdc_feature, history_frames=4, future_frames=8, dt=0.5, threshold=0.5):
        """
        检查是否是单调轨迹（可用恒速模型外推且误差小）

        Args:
            sdc_feature: np.array of shape (12, 7) - 轨迹特征
            history_frames: int - 历史帧数
            future_frames: int - 未来帧数
            dt: float - 时间间隔（秒）
            threshold: float - 平均位置误差阈值（米）

        Returns:
            bool - 是否是单调轨迹（True 表示应被过滤）
        """
        ego_velocity_2d = sdc_feature[3, 3:5]  # [vx, vy]
        ego_speed = (ego_velocity_2d**2).sum(-1) ** 0.5

        num_poses, dt = (
            8,
            0.5,
        )
        poses = np.array(
            [[(time_idx + 1) * dt * ego_speed, 0.0] for time_idx in range(num_poses)],
            dtype=np.float32,
        )

        true_future_pos = sdc_feature[4:, :2]  # shape: (future_frames, 2)

        # 计算平均欧氏距离
        errors = np.linalg.norm(poses - true_future_pos, axis=1)
        mean_error = np.mean(errors)

        return mean_error < threshold

    def postprocess(self, data):
        internal_format, results = data
        return results
        # from tqdm import tqdm
        # img_list = []
        # for i in tqdm(range(0, 200, 5)):
        #     img_dict = self.renderer.observe(internal_format, i)
        #     img_list.append(img_dict)
        # save_as_video(img_list, f"output_before.mp4")

        sdc_id = internal_format['metadata']['sdc_id']
        tracks = internal_format['tracks']
        sdc_track = tracks[sdc_id]

        sdc_pos = sdc_track['state']['position'][..., :2]
        sdc_heading = sdc_track['state']['heading'][..., np.newaxis]
        sdc_feature = np.concatenate([sdc_pos, sdc_heading], axis=-1)
        sdc_feature = [sdc_feature[i] for i in range(0, internal_format['length'], 5)]
        sdc_feature = np.stack(sdc_feature, axis=0)
        ego_dynamics = internal_format['ego_dynamics']
        dynamic_feature = [
            np.concatenate([ego_dynamics[i]['velocity'], ego_dynamics[i]['acceleration']], axis=-1) for i in
            range(len(ego_dynamics))]

        dynamic_feature = np.stack(dynamic_feature, axis=0)
        data_len = sdc_feature.shape[0]
        perturb_prob = self.config['perturb_prob']
        perturb_scale = self.config['perturb_scale']

        for current_index in range(3, data_len):
            if np.random.rand() > perturb_prob:
                continue
            max_future_index = current_index + 8
            if max_future_index >= data_len:
                break

            total_index = np.arange(current_index-3, max_future_index+1)
            past_index = total_index[:4]

            internal_format['tracks'][sdc_id]['state']['position'] = sdc_pos
            internal_format['tracks'][sdc_id]['state']['heading'] = sdc_heading
            dx = np.random.normal(0.0, perturb_scale)  # 纵向
            dy = np.random.normal(0.0, perturb_scale)  # 横向
            perturbed_pos = perturb_positions(sdc_pos, current_index, dx, dy, std=10)
            heading_raw = derive_heading_from_positions(perturbed_pos)
            perturbed_heading = blend_and_smooth_heading(sdc_heading, heading_raw)
            perturbed_velocity = recompute_velocity(perturbed_pos)

            sdc_pos = perturbed_pos[:,:2]
            sdc_heading = perturbed_heading[:, np.newaxis]
            sdc_vel = perturbed_velocity
            sdc_acce = np.zeros_like(sdc_vel)
            sdc_acce[1:] = (sdc_vel[1:] - sdc_vel[:-1]) / 0.1
            sdc_acce[0] = sdc_acce[1]
            sdc_feature = np.concatenate([sdc_pos, sdc_heading, sdc_vel, sdc_acce], axis=-1)
            sdc_feature_raw = sdc_feature[total_index]

            sdc_pos, sdc_heading, sdc_vel, sdc_acce = sdc_feature_raw[:, :2], sdc_feature_raw[:, 2:3], sdc_feature_raw[:, 3:5], sdc_feature_raw[:, 5:]
            # Normalize position by translating to the origin (t=0)
            sdc_pos_norm = sdc_pos - sdc_pos[3]  # shape (T, 2)

            sdc_heading_norm = normalize_angle(sdc_heading - sdc_heading[3])  # shape (T,)

            # Get rotation matrix to align to heading at t=0
            theta0 = sdc_heading[3, 0]
            cos_t, sin_t = np.cos(-theta0), np.sin(-theta0)
            R = np.array([[cos_t, -sin_t],
                          [sin_t, cos_t]])  # shape (2, 2)

            # Rotate position, velocity, acceleration into ego frame at t=0
            sdc_pos_norm = sdc_pos_norm @ R.T  # shape (T, 2)
            sdc_vel_rot = sdc_vel @ R.T  # shape (T, 2)
            sdc_acce_rot = sdc_acce @ R.T  # shape (T, 2)

            # Rebuild feature: pos (normalized), heading (normalized), vel (rotated), acc (rotated)
            sdc_feature_t = np.concatenate([sdc_pos_norm, sdc_heading_norm, sdc_vel_rot, sdc_acce_rot], axis=-1)

            is_monotonic = self.is_monotonic_trajectory(sdc_feature_t,
                                                        threshold=self.config['constant_velocity_threshold'])
            if is_monotonic:
                continue

            # from tqdm import tqdm
            # img_list = []
            # for i in range(current_index-15, max_future_index+1, 5):
            #     img_dict = self.renderer.observe(internal_format, i)
            #     img_list.append(img_dict)
            # save_as_video(img_list, f"output_before.mp4")

            internal_format['tracks'][sdc_id]['state']['position'] = perturbed_pos
            internal_format['tracks'][sdc_id]['state']['heading'] = perturbed_heading
            internal_format['tracks'][sdc_id]['state']['velocity'] = perturbed_velocity
            img_dict = self.renderer.observe(internal_format, current_index)

            # from tqdm import tqdm
            # img_list = []
            # for i in range(current_index-15, max_future_index+1, 5):
            #     img_dict = self.renderer.observe(internal_format, i)
            #     img_list.append(img_dict)
            # save_as_video(img_list, f"output_after.mp4")

            used_cameras = self.config['used_cameras']
            camera_index = past_index//5

            command = [driving_command[i] for i in camera_index]
            sdc_past_feature = sdc_feature_t[:4]
            sdc_future_feature = sdc_feature_t[4:]
            features_render = {}

            for builder in self._target_builders:
                features_render.update(builder.compute_targets(sdc_future_feature[:,:3]))
            dummy_camera = {}
            for k,v in img_dict.items():
                dummy_camera[k] = None
            camera = [img_dict, img_dict, img_dict, img_dict]
            agent_input = get_agent_input(sdc_past_feature, command, camera, used_cameras)

            for builder in self._feature_builders:
                features_render.update(builder.compute_features(agent_input))

            features_render['camera_feature_real'] = np.zeros_like(features_render['camera_feature'])
            features_render['real_valid_mask'] = False
            results.append(features_render)

            #
            # import matplotlib.pyplot as plt
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            # camera = features_render['camera_feature'].transpose(1, 2, 0)
            # ego_status = features_render['status_feature']
            # input_traj = sdc_past_feature[:, :2]
            # gt_traj = features_render['trajectory'][:, :2]
            # ax1.imshow(camera)
            # ax1.set_title("Camera View")
            # ax1.axis('off')
            #
            # status_text = "\n".join([f"{i}: {v:.2f}" for i, v in enumerate(ego_status)])
            # props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            # ax1.text(5, 20, status_text, fontsize=10, verticalalignment='top', bbox=props)
            #
            # ax2.plot(input_traj[:, 0], input_traj[:, 1], 'ro-', label="Input Trajectory")
            # ax2.plot(gt_traj[:, 0], gt_traj[:, 1], 'go-', label="Ground Truth Trajectory")
            #
            # for i in range(len(gt_traj)):
            #     ax2.annotate(str(i), (gt_traj[i, 0], gt_traj[i, 1]), color='green')
            #
            # ax2.set_title("Trajectory")
            # ax2.set_xlabel("X")
            # ax2.set_ylabel("Y")
            # ax2.legend()
            # ax2.grid(True)
            # ax2.axis('equal')  # 保持坐标轴比例一致
            # plt.show()
            # print()
        return results

    def collate_fn(self, data_list):
        batch_list = []
        for batch in data_list:
            batch_list.append(batch)

        batch_size = len(batch_list)
        key_to_list = {}
        for key in batch_list[0].keys():
            key_to_list[key] = [batch_list[bs_idx][key] for bs_idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            # if val_list is str:
            try:
                input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
            except:
                input_dict[key] = val_list

        feature = {}
        target = {}
        feature['camera_feature'] = input_dict['camera_feature']
        feature['camera_feature_real'] = input_dict['camera_feature_real']
        feature['real_valid_mask'] = input_dict['real_valid_mask']
        feature['status_feature'] = input_dict['status_feature']
        target['trajectory'] = input_dict['trajectory']
        #feature['camera_path'] = input_dict['camera_path']
        #batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_size}
        return (feature, target)


def _gaussian_kernel1d(sigma, radius):
    """纯 NumPy 生成 1‑D 高斯核"""
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel

def _blur_numpy(img, sigma):
    """对 (C,H,W) 图像做 separable Gaussian blur（无 OpenCV 时使用）"""
    radius = int(3 * sigma)
    if radius == 0:
        return img
    k = _gaussian_kernel1d(sigma, radius)
    # separable：先 H 方向，再 W 方向
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, img)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 2, tmp)
    return blurred

def augment_image(
    img,
    noise_std_range=(0.0, 0.03),
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    gamma_range=(0.9, 1.1),
    flip_prob=0.5,
    dropout_prob=0.3,
    dropout_size=0.1,
    blur_prob=1,
    blur_sigma_range=(0.5, 2.0),
    seed=None,
):
    """
    Data augmentation with added Gaussian blur.

    Parameters
    ----------
    ...（前面的参数保持不变）...
    blur_prob : float
        Probability of applying Gaussian blur.
    blur_sigma_range : tuple
        Uniform range for σ when blurring.
    """

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    orig_dtype = img.dtype
    x = img.astype(np.float32)
    if orig_dtype == np.uint8:
        x /= 255.0

    # 1. Horizontal flip
    if np.random.rand() < flip_prob:
        x = x[:, :, ::-1]

    # 2. Brightness
    x *= np.random.uniform(*brightness_range)

    # 3. Contrast
    mean = x.mean(axis=(1, 2), keepdims=True)
    x = (x - mean) * np.random.uniform(*contrast_range) + mean

    # 4. Gamma tone‑mapping
    x = np.clip(x, 1e-6, 1.0) ** np.random.uniform(*gamma_range)

    # 5. **Gaussian blur**
    if np.random.rand() < blur_prob:
        sigma = np.random.uniform(*blur_sigma_range)
        try:
            import cv2  # 优先用 OpenCV
            # cv2 接收 H×W×C 且通道最后
            x = cv2.GaussianBlur(
                x.transpose(1, 2, 0), ksize=(0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT101
            ).transpose(2, 0, 1)
        except ImportError:
            x = _blur_numpy(x, sigma)

    # 6. Gaussian noise
    sigma_n = np.random.uniform(*noise_std_range)
    if sigma_n > 0:
        x += np.random.normal(0.0, sigma_n, size=x.shape).astype(np.float32)

    # 7. Coarse dropout (CutOut)
    if np.random.rand() < dropout_prob:
        h, w = x.shape[1:]
        sz = int(min(h, w) * dropout_size)
        if sz > 0:
            top = np.random.randint(0, h - sz + 1)
            left = np.random.randint(0, w - sz + 1)
            x[:, top : top + sz, left : left + sz] = 0.0

    # 8. Clip & cast back
    x = np.clip(x, 0.0, 1.0)
    if orig_dtype == np.uint8:
        x = (x * 255.0).round().astype(np.uint8)
    else:
        x = x.astype(orig_dtype)

    if seed is not None:
        np.random.set_state(rng_state)

    return x