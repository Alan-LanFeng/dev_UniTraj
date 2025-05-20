import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from datasets import build_dataset
from utils.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def cluster(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    train_set = build_dataset(cfg)

    train_loader = DataLoader(
        train_set, batch_size=1024, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    future_list = []
    
    for batch in train_loader:
        feature,target = batch
        future_list.append(target['trajectory'])
        

    future_array = np.concatenate(future_list).reshape(-1,20)
    K=20
    import matplotlib.pyplot as plt
    cluster = KMeans(n_clusters=K).fit(future_array).cluster_centers_
    cluster = cluster.reshape(-1, 10, 2)
    for j in range(K):
        plt.scatter(cluster[j, :, 0], cluster[j, :,1])
    plt.savefig(f'plan_{K}.png', bbox_inches='tight')
    plt.close()

    np.save(f'kmeans_womd_{K}.npy', cluster)



if __name__ == '__main__':
    cluster()
