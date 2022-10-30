import torch
import torch.nn as nn
from utils.conv import ConvBn, ConvBnAct, BnActConv
from utils.block import DenseBlock, DenseTransition
from utils.convert_onnx import export_onnx


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, num_class, growth_rate=12, reduction=0.5):
        super().__init__()
        self.growth_rate = growth_rate
        inner_c= 2 * growth_rate
        self.conv1= nn.Sequential(
            nn.Conv2d(3, inner_c, kernel_size=3, padding= 1, bias= False),
            nn.MaxPool2d(3, 2, 1))

        self.features= nn.Sequential()

        for idx in range(len(nblocks) - 1):
            self.features.add_module(f'dense_layer_{idx}', self._make_layers(block, inner_c, nblocks[idx]))
            inner_c += growth_rate * nblocks[idx]
            out_c= int(reduction * inner_c)
            self.features.add_module(f'transition_layer_{idx}', DenseTransition(inner_c, out_c))
            inner_c= out_c

        self.features.add_module(f'dense_block_{len(nblocks) -1}', self._make_layers(block, inner_c, nblocks[len(nblocks)- 1]))
        inner_c += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_c))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_c, num_class) 
    
    def _make_layers(self, block, in_c, nblocks):
        dense_block= nn.Sequential()
        for idx in range(nblocks):
            dense_block.add_module(f'bottle_neck_layer_{idx}', block(in_c, self.growth_rate))
            in_c += self.growth_rate
        
        return dense_block
        
    def forward(self, x):
        x= self.conv1(x)
        x= self.features(x)
        x= self.avgpool(x)
        x= x.view(x.size()[0], -1)
        x= self.linear(x)

        return x


