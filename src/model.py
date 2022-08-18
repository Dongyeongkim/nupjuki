# Neural Network model based on PyTorch

import torch 
import torch.nn as nn



# Component of StateTranstion Model - visual encoding

class ResBlock(nn.Module):
    def __init__(self, channels_input, channels_first, channels_last, kernel_size):
        super(ResBlock, self).__init__()
        self.convblock1 = nn.Conv2d(in_channels=channels_input, out_channels=channels_first, kernel_size=kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=channels_first)
        self.inner_elu = nn.ELU()
        self.convblock2 = nn.Conv2d(in_channels=channels_first, out_channels=channels_last, kernel_size = kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=channels_last)
        self.outer_elu = nn.ELU()

    def forward(self, x):
        x = x + self.bn2(self.convblock2(self.inner_elu(self.bn1(self.convblock1(x)))))
        x = self.outer_elu(x)
        return x



# Component of StateTransition Model - PolicyNet

class Policy(nn.Module):
    def __init__(self, ):
        pass
    def forward(self, x):
        pass

# Component of StateTransition Model - ValueNet

class Value(nn.Module):
    def __init__(self, ):
        pass
    def 



class StateTransitionModel(nn.Module):
    def __init__(self):
        self.block1 = ResBlock(channels_first=1, channels_input=48, channels_last=192, kernel_size=5)
        self.block2 = ResBlock(channels_first=48, channels_input=192, channels_last=192, kernel_size=3)
        self.block3 = ResBlock(channels_first=192, channels_input=192, channels_last=192, kernel_size=3)
        self.block4 = ResBlock(channels_first=192, channels_input=192, channels_last=192, kernel_size=3)
        self.block5 = ResBlock(channels_first=192, channels_input=192, channels_last=192, kernel_size=3)
        self.policyhead = Policy()
        self.valuehead = Value()
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        policy = self.policyhead(x)
        value = self.valuehead(policy)
        
        return policy, value

        