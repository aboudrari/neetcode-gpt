import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        w = torch.randn(fan_out, fan_in) * std
        return w.round(decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2 / fan_in)
        w = torch.randn(fan_out, fan_in) * std
        return w.round(decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        Ws = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim
            if init_type == 'xavier':
                std = math.sqrt(2 / (fan_in + fan_out))
            elif init_type == 'kaiming':
                std = math.sqrt(2 / fan_in)
            else:
                std = 1
            Ws.append(torch.randn(fan_out, fan_in) * std)
        
        x = torch.randn(1, input_dim)
        stds = []
        for W in Ws:
            x = torch.relu(x @ W.T)
            stds.append(round(x.std().item(), 2))
        return stds

