from .base_dataset import BaseDataset

import numpy as np
from unitraj.models.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from unitraj.models.transfuser.transfuser_config import TransfuserConfig
import torch

import os
from typing import Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import math
import numpy as np
import cv2
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
import shutil
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
from tqdm import tqdm
import pickle
from torch.utils.data._utils.collate import default_collate

class WOMDDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        self._feature_builders = [TransfuserFeatureBuilder(TransfuserConfig)]
        self._target_builders = [TransfuserTargetBuilder(TransfuserConfig)]
       # self.renderer = ScenarioRenderer()
        super().__init__(config, is_validation)

    def load_data(self):
        
        data_path = self.data_path[0]
        if self.is_validation:
            print('Loading validation data...')
            phase = 'validation'
            FILES = os.path.join(data_path, 'validation*')
        else:
            print('Loading training data...')
            phase = 'training'
            FILES = os.path.join(data_path, 'training*')

        filenames = tf.io.matching_files(FILES)
        dataset = tf.data.TFRecordDataset(filenames, compression_type='')
        dataset_iter = dataset.as_numpy_iterator()

        self.cache_path = os.path.join(self.config['cache_path'], 'womd', phase)

        if os.path.exists(self.cache_path) and self.config.get('overwrite_cache', False) is False:
            print('Warning: cache path {} already exists, skip '.format(self.cache_path))
            with open(os.path.join(self.cache_path, 'file_list.pkl'), 'rb') as f:
                file_list = pickle.load(f)
        else:

            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)

            os.makedirs(self.cache_path, exist_ok=True)

            file_list = []
            for cnt, bytes_example in tqdm(enumerate(dataset_iter)):
                data = wod_e2ed_pb2.E2EDFrame()
                data.ParseFromString(bytes_example)

                data = self.process(data)

                with open(os.path.join(self.cache_path, str(cnt) + '.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                file_list.append(str(cnt) + '.pkl')

            with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                pickle.dump(file_list, f)

        print('Loaded {} samples from {}'.format(len(file_list), data_path))
        self.data_loaded = file_list


    def process(self, data):
        features = {}
        front3_camera_image_list, front3_camera_calibration_list = return_front3_cameras(data)

        l0 = front3_camera_image_list[0][28:-28, 416:-416]
        f0 = front3_camera_image_list[1][28:-28]
        r0 = front3_camera_image_list[2][28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        # # save image
        # image_path = os.path.join('test.png')
        # cv2.imwrite(image_path, resized_image[:, :, ::-1])
        resized_image = np.transpose(resized_image, (2, 0, 1)).astype(np.float32)/255.0
        features['camera_feature_real'] = resized_image
        features['real_valid_mask'] = True

        future_waypoints_matrix = np.stack([data.future_states.pos_x, data.future_states.pos_y], axis=1)
        future_index = np.arange(1, 1 + len(future_waypoints_matrix),2)
        future_waypoints_matrix = future_waypoints_matrix[future_index]


        features['trajectory'] = future_waypoints_matrix.astype(np.float32)

        history_vx = data.past_states.vel_x
        history_vy = data.past_states.vel_y
        history_ax = data.past_states.accel_x
        history_ay = data.past_states.accel_y
        history_dynamics = np.stack([history_vx, history_vy, history_ax, history_ay], axis=1)

        intent = data.intent
        if intent == 2:
            driving_command = np.array([1,0,0,0])
        elif intent == 3:
            driving_command = np.array([0,0,1,0])
        elif intent == 1:
            driving_command = np.array([0,1,0,0])
        else:
            driving_command = np.array([0,0,0,1])
        
        ego_status = np.concatenate([driving_command, history_dynamics[-1]]).astype(np.float32)
        features['status_feature'] = ego_status
        #features['camera_features'] = features['camera_features']
        return features
    
    def __getitem__(self, index):
        file_key = self.data_loaded[index]
        file_path = os.path.join(self.cache_path, file_key)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
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
        #feature['camera_feature'] = input_dict['camera_feature']
        feature['camera_feature_real'] = input_dict['camera_feature_real']
        feature['real_valid_mask'] = input_dict['real_valid_mask']
        feature['status_feature'] = input_dict['status_feature']
        target['trajectory'] = input_dict['trajectory']
        #feature['camera_path'] = input_dict['camera_path']
        #batch_dict = {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_size}
        return (feature, target)

    def __len__(self):
        return len(self.data_loaded)


def return_front3_cameras(data):
  """Return the front_left, front, and front_right cameras as a list of images"""
  image_list = []
  calibration_list = []
  # CameraName Enum reference:
  # https://github.com/waymo-research/waymo-open-dataset/blob/5f8a1cd42491210e7de629b6f8fc09b65e0cbe99/src/waymo_open_dataset/dataset.proto#L50
  order = [2, 1, 3]
  for camera_name in order:
    for index, image_content in enumerate(data.frame.images):
      if image_content.name == camera_name:
        # Decode the raw image string and convert to numpy type.
        calibration = data.frame.context.camera_calibrations[index]
        image = tf.io.decode_image(image_content.image).numpy()
        image_list.append(image)
        calibration_list.append(calibration)
        break

  return image_list, calibration_list