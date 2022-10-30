import torch
import torch.nn as nn
from utils.conv import ConvBn, ConvBnAct, BnActConv
from utils.convert_onnx import export_onnx

class ResnetBlock(nn.Module):
    expansion= 1
    def __init__(self, in_c, out_c, s, k):
        self.block= nn.Sequential(
            ConvBnAct(in_c= in_c, out_c= out_c, k= k, s= s),
            ConvBn(in_c= out_c, out_c= out_c * ResnetBlock.expansion, k= k, s= s)
        )

        self.short_cut= nn.Sequential()
        self.act= nn.ReLU()

        if s != 1 or in_c != out_c * ResnetBlock.expansion:
            self.short_cut= ConvBn(in_c, out_c * ResnetBlock.expansion, k, s)

    def forward(self, x):
        x= self.block(x) + self.short_cut(x)
        x= self.act(x)

        return x

class ResnetBottleNeck(nn.Module):
    expansion= 4
    def __init__(self, in_c, out_c, s=1):
        self.block= nn.Sequential(
            ConvBnAct(in_c= in_c, out_c= out_c, k= 1, s= s),
            ConvBnAct(in_c= out_c, out_c= out_c, k= 3, s= s),
            ConvBn(in_c= out_c, out_c= out_c * ResnetBottleNeck.expansion)
        )

        self.short_cut= nn.Sequential()
        self.act= nn.ReLU()

        if s != 1 or in_c != out_c * ResnetBottleNeck.expansion:
            self.short_cut= nn.Sequential(
                ConvBn(in_c= in_c, out_c= out_c * ResnetBottleNeck.expansion, k=1, s= s)
            )

    def forward(self, x):
        x= self.block(x) + self.short_cut(x)
        x= self.act(x)

        return x

class DenseBlock(nn.Module):
    def __init__(self, in_c, growth_rate):
        super().__init__()
        inner_c= 4 * growth_rate

        self.block= nn.Sequential(
            BnActConv(in_c, inner_c, k=1, s=1, p= 0),
            BnActConv(inner_c, growth_rate, k=3, s=1, p= 1)
        )

        self.short_cut= nn.Sequential()

    def forward(self, x):
        x= torch.cat([self.short_cut(x), self.block(x)], 1)
        return x
 

class DenseTransition(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block= BnActConv(in_c, out_c, 1, 1, 0)
        self.pool= nn.AvgPool2d(2, 2)

    def forward(self, x):
        x= self.block(x)
        x= self.pool(x)

        return x


# class ResNextBlock(nn.Module):
#     def __init__(self, in_c, out_c, s):
#         super().__init__()
#         C= 32
#         DEPTH= 4
#         BASEWIDTH= 64

#         D= 