import pytorch_lightning as pl
import torch
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
import os
import math
import tensorflow as tf
torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
import hydra
from omegaconf import OmegaConf
import os
import numpy as np

import torch
from typing import List, Dict, Tuple

def nms_trajectories(
    model_outputs: List[Dict[str, torch.Tensor]],
    dist_threshold: float = 2.0,  # m，判定“太近”就抑制
    top_k: int = 1,               # 每个样本最多保留多少条轨迹
    metric: str = "endpoint",     # 'endpoint' 或 'average'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args
    ----
    model_outputs : list(dict)
        每个元素来自一个模型，形状要求  
          › 'trajectory': [bs, modes, 10, 3]  (x, y, yaw)  
          › 'score'     : [bs, modes]
    dist_threshold : float
        NMS 距离阈值 (m)。
    top_k : int
        最终保留的轨迹条数。
    metric : str
        计算两条轨迹距离的方式：
        - 'endpoint' 只看最后一个时间步的 (x, y) 欧氏距离；
        - 'average'  10 个时间步的平均 (x, y) 距离。

    Returns
    -------
    kept_trajs : torch.Tensor  [bs, top_k, 10, 3]
    kept_scores: torch.Tensor  [bs, top_k]
    """
    # ① 纵向拼接 → [bs, Σmodes, 10, 3] / [bs, Σmodes]
    trajs = torch.cat([m["trajectory"] for m in model_outputs], dim=1)  # 同步 bs 维
    scores = torch.cat([m["score"]      for m in model_outputs], dim=1)

    best   = scores.argmax(dim=1)                                # [bs]
    best_traj  = trajs[torch.arange(trajs.size(0)), best][...,:2]                   # [bs, 10, 3]
    return best_traj

@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluation(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)
    cfg['submission'] = True
    test = build_dataset(cfg)

    eval_batch_size = cfg.method['eval_batch_size']

    test_loader = DataLoader(
        test, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=test.collate_fn)



    ckpt_home = '/home/fenglan/dev_UniTraj/ckpts'
    ckpt_paths = os.listdir(ckpt_home)
    ckpt_paths = [os.path.join(ckpt_home, ckpt) for ckpt in ckpt_paths]

    all_model_preds = []  # list over models
    for idx, ckpt in enumerate(ckpt_paths, 1):
        print(f"[Ensemble] Running inference {idx}/{len(ckpt_paths)} → {ckpt}")
        model = build_model(cfg)
        trainer = pl.Trainer(
            inference_mode=True,
            logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name),
            devices=1,
            accelerator="cpu" if cfg.debug else "gpu",
            profiler="simple",
        )
        preds = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=ckpt)
        all_model_preds.append(preds[0])
        torch.cuda.empty_cache()
    


    merged_results = nms_trajectories(all_model_preds)

    predictions = []
    print('eval on ', merged_results.shape[0], 'trajectories')
    for i in range(merged_results.shape[0]):
        x = merged_results[i, :, 0].detach().cpu().numpy()
        y = merged_results[i, :, 1].detach().cpu().numpy()
        # 原始帧对应的时间点（偶数帧）
        even_frames = np.arange(2, 21, 2)  # [2, 4, ..., 20]

        # 添加第0帧为起点 (0, 0)
        full_x = [0.0]
        full_y = [0.0]
        full_frames = [0]

        # 插入偶数帧坐标
        full_x.extend(x.tolist())
        full_y.extend(y.tolist())
        full_frames.extend(even_frames.tolist())

        # 所有帧 0~20
        all_frames = np.arange(21)

        # 使用线性插值补齐所有帧
        interp_x = np.interp(all_frames, full_frames, full_x)[1:]
        interp_y = np.interp(all_frames, full_frames, full_y)[1:]

        predicted_trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(pos_x=interp_x,
                                                                    pos_y=interp_y)
        frame_name = all_model_preds[0]['frame_name'][i]
        frame_trajectory = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(frame_name=frame_name, trajectory=predicted_trajectory)
        predictions.append(frame_trajectory)
            
    num_submission_shards = 1  # Please modify accordingly.
    submission_file_base = './MySubmission'  # Please modify accordingly.
    if not os.path.exists(submission_file_base):
        os.makedirs(submission_file_base)
        
    sub_file_names = [
        os.path.join(submission_file_base, part)
        for part in [f'mysubmission.binproto-00000-of-00001']
    ]
    # As the submission file may be large, we shard them into different chunks.
    submissions = []
    num_predictions_per_shard =  math.ceil(len(predictions) / num_submission_shards)
    for i in range(num_submission_shards):
        start = i * num_predictions_per_shard
        end = (i + 1) * num_predictions_per_shard
        submissions.append(
        wod_e2ed_submission_pb2.E2EDChallengeSubmission(
            predictions=predictions[start:end]))
    for i, shard in enumerate(submissions):
        shard.submission_type  =  wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
        shard.authors[:] = ['Lan Feng']  # Please modify accordingly.
        shard.affiliation = 'EPFL'  # Please modify accordingly.
        shard.account_name = 'lf2681@gmail.com'  # Please modify accordingly.
        shard.unique_method_name = 'UniPlan'  # Please modify accordingly.
        shard.method_link = 'None'  # Please modify accordingly.
        shard.description = ''  # Please modify accordingly.
        shard.uses_public_model_pretraining = True # Please modify accordingly.
        shard.public_model_names.extend(['Resnet']) # Please modify accordingly.
        shard.num_model_parameters = "60M" # Please modify accordingly.
        with tf.io.gfile.GFile(sub_file_names[i], 'wb') as fp:
            fp.write(shard.SerializeToString())



if __name__ == '__main__':
    # seed everything
    pl.seed_everything(0)
    evaluation()
