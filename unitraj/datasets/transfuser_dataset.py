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
        sdc_vel = sdc_track['state']['velocity']
        # calculate the acceleration
        sdc_acce = np.zeros_like(sdc_vel)
        sdc_acce[1:] = (sdc_vel[1:] - sdc_vel[:-1]) / 0.1
        sdc_acce[0] = sdc_acce[1]
        sdc_feature = np.concatenate([sdc_pos, sdc_heading, sdc_vel, sdc_acce], axis=-1)


        for current_index in range(15, data_len,5):
            max_future_index = current_index + 40
            if max_future_index >= data_len:
                break
            total_index = np.arange(current_index-15, max_future_index+1, 5)
            past_index = total_index[:4]

            sdc_feature_raw = sdc_feature[total_index].copy()
            sdc_pos, sdc_heading, sdc_vel, sdc_acce = sdc_feature_raw[:, :2], sdc_feature_raw[:, 2:3], sdc_feature_raw[:, 3:5], sdc_feature_raw[:, 5:]

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
            sdc_vel_rot = sdc_vel @ R.T  # shape (T, 2)
            sdc_acce_rot = sdc_acce @ R.T  # shape (T, 2)

            # Rebuild feature: pos (normalized), heading (normalized), vel (rotated), acc (rotated)
            sdc_feature_t = np.concatenate([sdc_pos_norm, sdc_heading_norm, sdc_vel_rot, sdc_acce_rot], axis=-1)

            is_monotonic = self.is_monotonic_trajectory(sdc_feature_t, threshold=self.config['constant_velocity_threshold'])
            if is_monotonic:
                continue
            used_cameras = self.config['used_cameras']
            camera_index = past_index//5

            command = [driving_command[i] for i in camera_index]
            sdc_past_feature = sdc_feature_t[:4]
            sdc_future_feature = sdc_feature_t[4:]
            features_render = {}
            for builder in self._target_builders:
                features_render.update(builder.compute_targets(sdc_future_feature[:,:3]))
            camera = [camera_data_render[i] for i in camera_index]
            agent_input = get_agent_input(sdc_past_feature, command, camera, used_cameras)

            for builder in self._feature_builders:
                features_render.update(builder.compute_features(agent_input))

            features = {}
            try:
                camera = [camera_data_real[i] for i in camera_index]
                agent_input = get_agent_input(sdc_past_feature, command, camera, used_cameras)
                for builder in self._feature_builders:
                    features.update(builder.compute_features(agent_input))

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
        assert sdc_feature.shape == (history_frames + future_frames, 7)

        # 历史和未来轨迹
        history = sdc_feature[:history_frames]
        future = sdc_feature[history_frames:]

        # 获取最后一帧的历史位置和速度
        last_pos = history[-1, :2]  # [x, y]
        velocity = history[-1, 3:5]  # [vx, vy]

        # 使用 constant velocity model 外推未来位置
        predicted_future_pos = []
        for i in range(1, future_frames + 1):
            delta_t = i * dt
            pred_pos = last_pos + velocity * delta_t
            predicted_future_pos.append(pred_pos)
        predicted_future_pos = np.stack(predicted_future_pos, axis=0)  # shape: (future_frames, 2)

        # 真实未来位置
        true_future_pos = future[:, :2]  # shape: (future_frames, 2)

        # 计算平均欧氏距离
        errors = np.linalg.norm(predicted_future_pos - true_future_pos, axis=1)
        mean_error = np.mean(errors)

        return mean_error < threshold

    def postprocess(self, data):
        internal_format, results = data
        #return results
        # from tqdm import tqdm
        # img_list = []
        # for i in tqdm(range(0, 200, 5)):
        #     img_dict = self.renderer.observe(internal_format, i)
        #     img_list.append(img_dict)
        # save_as_video(img_list, f"output_before.mp4")

        sdc_id = internal_format['metadata']['sdc_id']
        tracks = internal_format['tracks']
        sdc_track = tracks[sdc_id]

        original_pos = sdc_track['state']['position'].copy() # (N,3)
        original_heading = sdc_track['state']['heading'].copy()  # (N,)
        original_velocity = sdc_track['state']['velocity'].copy()  # (N,3)
        data_len = original_pos.shape[0]
        driving_command = internal_format['driving_command']
        perturb_scale = self.config['perturb_scale']
        perturb_prob = self.config['perturb_prob']
        for current_index in range(15, data_len, 5):
            if np.random.rand() > perturb_prob:
                continue
            max_future_index = current_index + 40
            if max_future_index >= data_len:
                break

            total_index = np.arange(current_index-15, max_future_index+1, 5)
            past_index = total_index[:4]

            internal_format['tracks'][sdc_id]['state']['position'] = original_pos
            internal_format['tracks'][sdc_id]['state']['heading'] = original_heading
            internal_format['tracks'][sdc_id]['state']['velocity'] = original_velocity
            dx = np.random.normal(0.0, perturb_scale)  # 纵向
            dy = np.random.normal(0.0, perturb_scale)  # 横向
            perturbed_pos = perturb_positions(original_pos, current_index, dx, dy, std=10)
            heading_raw = derive_heading_from_positions(perturbed_pos)
            perturbed_heading = blend_and_smooth_heading(original_heading, heading_raw)
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

            internal_format['tracks'][sdc_id]['state']['position'] = perturbed_pos
            internal_format['tracks'][sdc_id]['state']['heading'] = perturbed_heading
            internal_format['tracks'][sdc_id]['state']['velocity'] = perturbed_velocity
            img_dict = self.renderer.observe(internal_format, current_index)

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

