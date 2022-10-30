import sys
sys.path.append('')
import torch
import torch.nn as nn
from utils.conv import ConvBn, ConvBnAct
from utils.block import ResnetBlock, ResnetBottleNeck
from utils.convert_onnx import export_onnx

class Stem(nn.Module):
    def __init__(self, in_c, out_c, k, s):
        super().__init__()
        self.stem= ConvBnAct(in_c, out_c, k, s, p=None)
        self.pool= nn.MaxPool2d(kernel_size= 3, stride= 2)

    def forward(self, x):
        x= self.stem(x)
        x= self.pool(x)

        return x

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super().__init__()
        self.in_c= 64
        self.stem= Stem(in_c= 3, out_c= 64, k= 7, s= 2)

        self.block_1= self._create_layer(block, 64, num_block[0], 1)
        self.block_2= self._create_layer(block, 128, num_block[1], 2)
        self.block_3= self._create_layer(block, 256, num_block[2], 2)
        self.block_4= self._create_layer(block, 512, num_block[3], 2)

        self.pool= nn.AdaptiveAvgPool2d((1, 1))
        self.fc= nn.Linear(512 * block.expansion, num_classes)

    def _create_layer(self, block, out_c, num_blocks, s, k):
        strides= [s] + [1] * (num_blocks -1)
        layers= []
        for s in strides:
            layers.append(block(self.in_c, out_c, s, k))
            self.in_c= out_c * block.expansion

    def forward(self, x):
        x= self.stem(x)
        x= self.block_1(x)
        x= self.block_2(x)
        x= self.block_3(x)
        x= self.block_4(x)
        x= self.pool(x)
        x= x.view(x.size(0), -1)
        x= self.fc(x)

        return x

def resnet18():
    return ResNet(ResnetBlock, [2,2,2,2], 10)

def resnet34():
    return ResNet(ResnetBlock, [3, 4, 6, 3], 10)

def resnet50():
    return ResNet(ResnetBottleNeck, [3,4,6,3], 10)

def resnet101():
    return ResNet(ResnetBottleNeck, [3, 4, 23, 3], 10)

def resnet152():
    return ResNet(ResnetBottleNeck, [3, 8, 36, 3], 10)


model= resnet18()
export_onnx(model, torch.randn((1, 3, 224, 224)), 'resnet.onnx')