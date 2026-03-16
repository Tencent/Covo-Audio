import torch
import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(BaseModel, self).__init__()
        self.config = config
        self.integrity_test()

    def forward(self, feature):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self, feature):
        return self.forward(feature)

    def remove_weight_norm(self):
        pass

    def integrity_test(self):
        pass
