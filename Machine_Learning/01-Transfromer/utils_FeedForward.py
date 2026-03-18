import torch
import torch.nn as nn
from torch import Tensor

from torchtyping import TensorType, patch_typeguard # 用于tensor shape级类型提示
from typeguard import typechecked
patch_typeguard()


class FeedForward(nn.Module):
    def __init__(self, dim_model:int, dim_ff:int, dropout:float=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=dim_model, out_features=dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=dim_ff, out_features=dim_model) # 最后再转化回来
        )

    @typechecked
    def forward(self, x:TensorType["batch","seq","dim_model"]) -> TensorType["batch","seq","dim_model"]:
        return self.net(x)