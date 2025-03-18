from .MTR_dataset import MTRDataset
from .autobot_dataset import AutoBotDataset
from .wayformer_dataset import WayformerDataset
from .mlp_planner_dataset import MLPPlannerDataset
from .transfuser_dataset import TransfuserDataset

__all__ = {
    'autobot': AutoBotDataset,
    'wayformer': WayformerDataset,
    'MTR': MTRDataset,
    'MLPPlanner': MLPPlannerDataset,
    'transfuser': TransfuserDataset
}


def build_dataset(config, val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
