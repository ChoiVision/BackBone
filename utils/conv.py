import torch
import torch.nn as nn

class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, k, s, p= 'same', bias= False, **kwargs):
        super().__init__()
        
        self.conv_bn= nn.Sequential(
            nn.Conv2d(in_channels= in_c, out_channels= out_c, kernel_size= k, stride= s, padding= p, bias= bias),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.conv_bn(x)

class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k, s, act= None, p= 'same', bias= False, **kwargs):
        super().__init__()

        if act is None:
            act= nn.ReLU(inplace= True)

        self.conv_bn_act= nn.Sequential(
            nn.Conv2d(in_channels= in_c, out_channels= out_c, kernel_size= k, stride= s, padding= p, bias= bias),
            nn.BatchNorm2d(out_c),
            act
        )

    def forward(self, x):
        return self.conv_bn_act(x)