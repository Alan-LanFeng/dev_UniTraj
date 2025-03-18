from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.mlp_planner.MLPPlanner import MLPPlanner
from unitraj.models.transfuser.transfuser_agent import TransfuserLightningModule

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'MLPPlanner': MLPPlanner,
    'transfuser': TransfuserLightningModule
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
