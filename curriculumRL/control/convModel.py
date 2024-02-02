import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math

class ConvModel(nn.Module):
    def __init__(self, num_actions, num_inputs):
        super().__init__()
        self.layers = nn.Sequential(
            # Embedding num_inputs -> 64 of integers in map
            nn.Conv2d(num_inputs, 64, kernel_size=1, stride=1, padding=0), # num_inputs ,16,16 -> 64, 16, 16
            # Normal Conv
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64, 16, 16 -> 64, 16, 16
            nn.MaxPool2d(2,2), # 64, 16, 16 -> 64, 8, 8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 64, 8, 8 -> 128, 8, 8
            nn.MaxPool2d(2,2), # 128, 8, 8-> 128, 4, 4
            nn.Flatten(), # 128, 4, 4 -> 2048
            # Feed Forward
            nn.Linear(2048, 512), # 2048 -> 512
            nn.ReLU(),
            nn.Linear(512, 128), # 512 -> 128
            nn.ReLU(),
            nn.Linear(128, num_actions), # 128 -> out
        )
    def forward(self, x) -> Tensor:
        out = self.layers(x)
        return out