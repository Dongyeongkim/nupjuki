# Neural Network model based on PyTorch

import torch 
import torch.nn as nn



# Component of StateTranstion Model - visual encoding

class ResBlock(nn.Module):
    def __init__(self, channels_input, channels_first, channels_last, kernel_size):
        self.convblock1 = nn.Conv2d(in_channels=channels_input, out_channels=channels_first, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d()
        self.elu = nn.ELU()
        self.convblock2 = nn.Conv2d(in_channels=channels_first, out_channels=channels_last, kernel_size = kernel_size)
        self.bn2 = nn.BatchNorm2d()

    
    def forward(self, x):
        pass 

# Component of StateTransition Model - Actor

class Policy(nn.Module):
    def __init__(self, ):
        pass
    def forward(self, x):
        pass

class Value(nn.Module):
    def __init__(self, ):
        pass
    def 



class StateTransitionModel(nn.Module):
    def __init__(self, ):
        pass 
    
    def forward(self, x):
        pass
