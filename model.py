import torch
import torch.nn as nn


# Tuple -> (out_channels, kernel_size, stride), List -> [B, num_repeats]
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S"
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bnorm_act= True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = not bnorm_act, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.use_bnorm_act = bnorm_act 
        
    def forward(self, x):        
        if self.use_bnorm_act:
            return self.leaky_relu(self.batch_norm(self.conv(x)))
        else:
            return self.conv(x)
            
class ResidualBlock(nn.Module):
    pass 

class ScalePrediction(nn.Module): 
    pass

class Yolov3():
    pass 