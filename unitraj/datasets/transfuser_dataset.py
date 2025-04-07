from .base_dataset import BaseDataset
from unitraj.utils.dataclasses import get_agent_input
import numpy as np
from unitraj.models.transfuser.transfuser_features import TransfuserFeatureBuilder,TransfuserTargetBuilder
from unitraj.models.transfuser.transfuser_config import TransfuserConfig
import torch
class TransfuserDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        self._feature_builders = [TransfuserFeatureBuilder(TransfuserConfig)]
        self._target_builders = [TransfuserTargetBuilder(TransfuserConfig)]
        super().__init__(config, is_validation)



    def preprocess(self, scenario):
        return scenario

    def process(self, internal_format):
        sdc_id = internal_format['metadata']['sdc_id']
        tracks = internal_format['tracks']
        sdc_track = tracks[sdc_id]
        driving_command = internal_format['driving_command']
        if self.config['use_synthetic_sensors']:
            camera_data = internal_format['synthetic_camera']
        else:
            camera_data = internal_format['real_camera']
        data_len = len(driving_command)

        results = []
        sdc_pos = sdc_track['state']['position'][...,:2]
        sdc_heading = sdc_track['state']['heading'][...,np.newaxis]
        sdc_vel = sdc_track['state']['velocity']
        # calculate the acceleration
        sdc_acce = np.zeros_like(sdc_vel)
        sdc_acce[1:] = (sdc_vel[1:] - sdc_vel[:-1]) / 0.1
        sdc_acce[0] = sdc_acce[1]
        sdc_feature = np.concatenate([sdc_pos, sdc_heading, sdc_vel, sdc_acce], axis=-1)
        driving_command = internal_format['driving_command']


        for current_index in range(15, data_len,5):
            max_future_inex = current_index + 40
            if max_future_inex >= data_len:
                break
            total_index = np.arange(current_index-15, max_future_inex+1, 5)
            past_index = total_index[:4]
            future_index = total_index[4:]

            sdc_feature_raw = sdc_feature[total_index].copy()
            sdc_pos, sdc_heading, sdc_vel, sdc_acce = sdc_feature_raw[:, :2], sdc_feature_raw[:, 2:3], sdc_feature_raw[:, 3:5], sdc_feature_raw[:, 5:]

            # Normalize position by translating to the origin (t=0)
            sdc_pos_norm = sdc_pos - sdc_pos[3]  # shape (T, 2)

            def normalize_angle(angle):
                return (angle + np.pi) % (2 * np.pi) - np.pi

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
            camera = [camera_data[i] for i in camera_index]
            command = [driving_command[i] for i in past_index]
            sdc_past_feature = sdc_feature_t[:4]
            sdc_future_feature = sdc_feature_t[4:]

            agent_input = get_agent_input(sdc_past_feature,command, camera,used_cameras)
            features = {}
            for builder in self._feature_builders:
                features.update(builder.compute_features(agent_input))
            for builder in self._target_builders:
                features.update(builder.compute_targets(sdc_future_feature[:,:3]))
            features['kalman_difficulty'] = 0
            features['camera_path'] = camera[3]['CAM_F0']
            results.append(features)
        return results

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

        return data

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
        feature['status_feature'] = input_dict['status_feature']
        target['trajectory'] = input_dict['trajectory']
        feature['camera_path'] = input_dict['camera_path']
        #batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_size}
        return (feature, target)

    def get_agent_data(
            self, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
            obj_types
    ):

        num_center_objects = center_objects.shape[0]
        num_objects, num_timestamps, box_dim = obj_trajs_past.shape
        obj_trajs = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_past,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
        object_onehot_mask[:, obj_types == 1, :, 0] = 1
        object_onehot_mask[:, obj_types == 2, :, 1] = 1
        object_onehot_mask[:, obj_types == 3, :, 2] = 1
        object_onehot_mask[np.arange(num_center_objects), track_index_to_predict, :, 3] = 1
        object_onehot_mask[:, sdc_track_index, :, 4] = 1

        object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
        for i in range(num_timestamps):
            object_time_embedding[:, :, i, i] = 1
        object_time_embedding[:, :, :, -1] = timestamps

        object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
        object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
        object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

        vel = obj_trajs[:, :, :, 7:9]
        vel_pre = np.roll(vel, shift=1, axis=2)
        acce = (vel - vel_pre) / 0.1
        acce[:, :, 0, :] = acce[:, :, 1, :]

        obj_trajs_data = np.concatenate([
            obj_trajs[:, :, :, 0:6],
            object_onehot_mask,
            object_time_embedding,
            object_heading_embedding,
            obj_trajs[:, :, :, 7:9],
            acce,
        ], axis=-1)

        obj_trajs_mask = obj_trajs[:, :, :, -1]
        obj_trajs_data[obj_trajs_mask == 0] = 0

        obj_trajs_future = obj_trajs_future.astype(np.float32)
        obj_trajs_future = self.transform_trajs_to_center_coords(
            obj_trajs=obj_trajs_future,
            center_xyz=center_objects[:, 0:3],
            center_heading=center_objects[:, 6],
            heading_index=6, rot_vel_index=[7, 8]
        )

        obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 6]]
        #normalize the heading, last axis, to pi,-pi
        obj_trajs_future_state[:, :, :, 2] = np.arctan2(np.sin(obj_trajs_future_state[:, :, :, 2]),np.cos(obj_trajs_future_state[:, :, :, 2]))

        obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
        obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

        center_obj_idxs = np.arange(len(track_index_to_predict))
        center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
        center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
        center_gt_trajs[center_gt_trajs_mask == 0] = 0

        assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
        valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

        obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
        obj_trajs_data = obj_trajs_data[:, valid_past_mask]
        obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
        obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

        obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
        num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
        obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
        for k in range(num_timestamps):
            cur_valid_mask = obj_trajs_mask[:, :, k] > 0
            obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

        center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
        for k in range(center_gt_trajs_mask.shape[1]):
            cur_valid_mask = center_gt_trajs_mask[:, k] > 0
            center_gt_final_valid_idx[cur_valid_mask] = k

        max_num_agents = self.config['max_num_agents']
        object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

        object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
        topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]

        topk_idxs = np.expand_dims(topk_idxs, axis=-1)
        topk_idxs = np.expand_dims(topk_idxs, axis=-1)

        obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
        obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
        obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
        obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
        obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
        obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
        track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

        obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
        obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
        obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
        obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                    ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
        obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                        ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
        obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                       ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

        return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
                obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
                center_gt_final_valid_idx,
                track_index_to_predict_new)