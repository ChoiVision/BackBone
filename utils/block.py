import torch
import torch.nn as nn
from utils.conv import ConvBn, ConvBnAct

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