# given previous actions, and observation space, decide on optimal next action!
# understand previous actions : Transformer
# understand current observation space -> occupancy map: CNN, Linear Layers: numerical data
# decision -> DeepQ

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from collections import deque

class Action_Model(nn.Module):
    def __init__(self, **kwargs):
        dim = kwargs['dim']
        depth = kwargs['depth']
        heads = kwargs['heads']
        dropout = kwargs['dropout']
        super().__init__()
        self.emb = nn.Embedding(1,dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=4*dim,
                dropout=dropout, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
    
    def forward(self, obs, x):
        
        pass

class DeepQ_Control():
    def __init__(self, **kwargs):
        pass

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) 
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory) 
