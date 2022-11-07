import torch
import torch.nn as nn
from utils.conv import ConvBn, ConvBnAct, autopad
from utils.block import SEBlock

class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, stride= 1):
        super().__init__()
        self.depthwise= ConvBnAct(in_c= in_c, out_c= in_c, k= 3, s= stride, 
                                 d= 1, act= None, p= autopad(3, 1, 1))
        
        self.pointwise= ConvBnAct(in_c= in_c, out_c= out_c, k= 1, s= 1, p= autopad(k= 2, p= 0, d= 1))

        self.seblock= SEBlock(out_c)

    def forward(self, x):
        x= self.depthwise(x)
        x= self.pointwise(x)
        x= self.seblock(x)

        return x



