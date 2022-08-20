# Neural Network model based on PyTorch

import torch 
import torch.nn as nn

# Input layer - using 1x1 conv to match the channel size

class ResInput(nn.Module):
    def __init__(self):
        super(ResInput, self).__init__()
        self.resinput = nn.Conv2d(in_channels=1, out_channels=48, kernel_size=1)
        self.actfunc = nn.ELU()
        self.firstconv = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5, padding=2)
    def forward(self, x):
        return self.firstconv(self.actfunc(self.resinput(x)))

# Component of StateTranstion Model - visual encoding

class ResBlock(nn.Module):
    def __init__(self, channels_input, channels_first, channels_last, kernel_size, padding):
        super(ResBlock, self).__init__()
        self.convblock1 = nn.Conv2d(in_channels=channels_input, out_channels=channels_first, kernel_size= kernel_size, padding = padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=channels_first)
        self.inner_elu = nn.ELU()
        self.convblock2 = nn.Conv2d(in_channels=channels_first, out_channels=channels_last, kernel_size = kernel_size, padding = padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=channels_last)
        self.outer_elu = nn.ELU()

    def forward(self, x):
        skipcon = x
        x = self.convblock1(x)
        x = self.bn1(x)
        x = self.convblock2(x)
        x = self.bn2(x)
        x = x + skipcon
        x = self.outer_elu(x)
        return x


# Component of StateTransition Model - PolicyNet (Position)

class Policy(nn.Module):
    def __init__(self, input_channels):
        super(Policy, self).__init__()
        self.policynet = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1)
    def forward(self, x):
        return self.policynet(x)

# Component of StateTransition Model - PolicyNet (Action type)

class PolicyType(nn.Module):
    def __init__(self, input_channels, feature_num):
        super(PolicyType, self).__init__()
        self.policyinput = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1)
        self.policyflatten = nn.Flatten()
        self.policyact = nn.Linear(in_features=feature_num,out_features=4)
        self.regularizer = nn.Softmax()
    
    def forward(self, x):
        x = self.policyinput(x)
        x = self.policyflatten(x)
        x = self.policyact(x)
        x = self.regularizer(x)
        return x


# Component of StateTransition Model - ValueNet

class Value(nn.Module):
    def __init__(self, feature_num):
        super(Value, self).__init__()
        self.lin = nn.Linear(in_features=feature_num, out_features=1)
        self.valuenet = nn.Tanh()
    def forward(self, x):
        return self.valuenet(self.lin(x))



class StateTransitionModel(nn.Module):
    def __init__(self):
        super(StateTransitionModel, self).__init__()
        self.resinput = ResInput().cuda()
        self.block1 = ResBlock(channels_first=192, channels_input=192, channels_last=192, kernel_size=3, padding=1)
        self.block2 = ResBlock(channels_first=192, channels_input=192, channels_last=192, kernel_size=3, padding=1)
        self.block3 = ResBlock(channels_first=192, channels_input=192, channels_last=192, kernel_size=3, padding=1)
        self.block4 = ResBlock(channels_first=192, channels_input=192, channels_last=192, kernel_size=3, padding=1)
        self.policyhead = Policy(input_channels=192)
        self.policyact = PolicyType(input_channels=192, feature_num=81)
        self.valuehead = Value(feature_num=85) # 9*9 + 4
        self.flatpol = nn.Flatten()
        
    
    def forward(self, x):
        x = self.resinput(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        policypoint = self.policyhead(x)
        policytype = self.policyact(x)
        flatpolicy = self.flatpol(policypoint)
        policy = torch.cat((flatpolicy, policytype), dim=1)

        value = self.valuehead(policy)
        
        return policypoint, policytype, value


if __name__ == "__main__":
    
    if torch.cuda.is_available():
        model = StateTransitionModel().cuda()
        input = torch.randn(1,1,9,9).cuda()
    else:
        model = StateTransitionModel()
        input = torch.randn(1,1,9,9)
    
    policypoint, policytype, value = model(input)
    print(policypoint, policytype, value)
    print(policypoint.size(), policytype.size(), value.size())