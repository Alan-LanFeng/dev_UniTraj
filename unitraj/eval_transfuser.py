import os
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
import hydra
from omegaconf import OmegaConf
from unitraj.utils.dataclasses import camera_params
import numpy as np
import cv2

@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)
    val_set = build_dataset(cfg, val=True)
    model.agent._checkpoint_path = '/Users/fenglan/Desktop/vita-group/code/UniTraj/unitraj/unitraj_ckpt/epoch=81-.2f=0.ckpt'
    model.agent.initialize()
    model.eval()
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)
    K = camera_params['CAM_F0']['intrinsics']
    sensor_to_car_rotation = camera_params['CAM_F0']['sensor2lidar_rotation']
    sensor_to_car_translation = camera_params['CAM_F0']['sensor2lidar_translation']

    os.makedirs('output', exist_ok=True)
    for idx, batch in enumerate(val_loader):
        features, targets = batch
        prediction = model.agent.forward(features)
        camera = features['camera_path'][0]
        predicted_trajs = prediction['trajectory'].detach().cpu().numpy()[0,:,:2]
        real_trajs = targets['trajectory'].detach().cpu().numpy()[0,:,:2]

        # === 推导 car_to_sensor 的变换 ===
        # 先取逆：R_car_to_sensor = R_sensor_to_car.T
        R_car_to_sensor = sensor_to_car_rotation.T
        t_car_to_sensor = -R_car_to_sensor @ sensor_to_car_translation.reshape(3, )
        # === 轨迹点从车体坐标 → 相机坐标 ===
        N = predicted_trajs.shape[0]
        xyz_car = np.hstack([predicted_trajs, np.zeros((N, 1))])  # 假设 z=0（地面）
        xyz_car_real = np.hstack([real_trajs, np.zeros((N, 1))])
        # 转换到相机坐标系
        xyz_cam = (R_car_to_sensor @ xyz_car.T).T + t_car_to_sensor  # (N, 3)
        xyz_cam_real = (R_car_to_sensor @ xyz_car_real.T).T + t_car_to_sensor
        # === 用相机内参投影到图像 ===
        uv_homo = (K @ xyz_cam.T).T  # (N, 3)
        uv = uv_homo[:, :2] / uv_homo[:, 2:]
        uv_homo_real = (K @ xyz_cam_real.T).T  # (N, 3)
        uv_real = uv_homo_real[:, :2] / uv_homo_real[:, 2:]
        # === 可视化到图像 ===
        image = cv2.imread(camera)
        for pt in uv:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # 仅画在图像范围内的点
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        for pt in uv_real:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # 仅画在图像范围内的点
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)


        # 展示并保存图像
        cv2.imwrite(f'output/{idx}.jpg', image)






if __name__ == '__main__':
    train()
