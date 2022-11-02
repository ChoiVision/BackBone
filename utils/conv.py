import torch
import torch.nn as nn

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, k, s, d= 1, p= None, bias= False, **kwargs):
        super().__init__()
        
        self.conv_bn= nn.Sequential(
            nn.Conv2d(in_channels= in_c, out_channels= out_c, kernel_size= k,
             stride= s, padding= autopad(k, s, d), bias= bias),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        return self.conv_bn(x)
    
class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k, s, d= 1, act= None, p= None, bias= False, **kwargs):
        super().__init__()

        if act is None:
            act= nn.ReLU(inplace= True)

        self.conv_bn_act= nn.Sequential(
            nn.Conv2d(in_channels= in_c, out_channels= out_c, kernel_size= k, stride= s, 
            padding= autopad(k, s, d), bias= bias),
            nn.BatchNorm2d(out_c),
            act
        )

    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        x= self.act(x)

        return x


class BnActConv(nn.Module):
    def __init__(self, in_c, out_c, k, s, d=1, p= None, b= False, **kwargs):
        super().__init__()
        self.block= nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace= True),
            nn.Conv2d(in_channels= in_c, out_channels=out_c, kernel_size= k, stride= s, 
            padding= autopad(k, s, d), bias= b, **kwargs)
        )

    def forward(self, x):
        return self.block(x)
