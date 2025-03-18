from .base_dataset import BaseDataset


class MLPPlannerDataset(BaseDataset):

    def __init__(self, config=None, is_validation=False):
        super().__init__(config, is_validation)

    def preprocess(self, scenario):
        pass

