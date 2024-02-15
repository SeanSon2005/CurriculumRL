import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConvModel(nn.Module):
    def __init__(self, num_actions, num_inputs):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Embedding num_inputs -> 64 of integers in map
            nn.Conv2d(num_inputs, 64, kernel_size=1, stride=1, padding=0), # num_inputs ,17,17 -> 64, 17, 17
            # Normal Conv
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # 64, 17, 17 -> 64, 17, 17
            nn.MaxPool2d(2,2), # 64, 17, 17 -> 64, 8, 8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 64, 8, 8 -> 128, 8, 8
            nn.MaxPool2d(2,2), # 128, 8, 8-> 128, 4, 4
            nn.Flatten(), # 128, 4, 4 -> 2048
            # Feed Forward
            nn.Linear(2048, 1024), # 2048 -> 1024
            nn.Linear(1024, 512),  # 1024 -> 512
        )
        self.encode_descriptors = nn.Sequential(
            nn.Linear(5, 64),
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(512 + 64, 256),   # 512+64 -> 256
            nn.Linear(256, 64),        # 256 -> 64
            nn.Linear(64, num_actions), # 64 -> out
        )
        
    def forward(self, x: Tensor, descriptors : Tensor) -> Tensor:
        conv_out = self.conv_layers(x) # (bs, 512)
        disp_out = self.encode_descriptors(descriptors) # (bs, 16)
        features = torch.cat((conv_out,disp_out), dim=1) # (bs, 512+16)
        out = self.feed_forward(features) # (512+16) -> out
        return out