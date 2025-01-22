import torch
import torch.nn as nn
from torch.nn import Parameter


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

@torch.jit.script
def snakebeta(x,alpha,beta):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)
    
class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super(SnakeBeta, self).__init__()

        self.alpha = Parameter(torch.ones(1, channels, 1))
        self.beta = Parameter(torch.ones(1, channels, 1))


    def forward(self, x):
        x = snakebeta(x, self.alpha, self.beta)
        return x