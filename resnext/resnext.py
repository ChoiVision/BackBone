import sys
sys.path.append('')
import torch
import torch.nn as nn
from utils.conv import ConvBn, ConvBnAct
from utils.block import ResNextBlock
from utils.convert_onnx import export_onnx

class ResNext(nn.Module):
    def __init__(self, block, num_blocks, num_classes, cardinality= 32, width= 4):
        super().__init__()
        self.in_c= 64
        self.group_conv_width= cardinality * width
        self.stem= nn.Sequential(
            ConvBn(3, self.in_c, k=7, s= 2, p= 3),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        )        

        self.layer1= self._layer(block, cardinality, num_blocks[0], 1)
        self.layer2= self._layer(block, cardinality, num_blocks[1], 1)
        self.layer3= self._layer(block, cardinality, num_blocks[2], 1)
        self.layer4= self._layer(block, cardinality, num_blocks[3], 1)

        self.pool= nn.AdaptiveAvgPool2d((1,1))

        self.linear= nn.Linear(self.group_conv_width, num_classes)
    
    def _layer(self, block, cardinality, num_blocks, stride):
        strides= [stride] + [1] * (num_blocks-1)
        layers= []

        for n in range(num_blocks):
            layers.append(block(self.in_c, self.group_conv_width, cardinality, strides[n]))
            self.in_c= block.expansion * self.group_conv_width
        self.group_conv_width *= block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x= self.stem(x)
        x= self.layer1(x)
        x= self.layer2(x)
        x= self.layer3(x)
        x= self.layer4(x)
        x= self.pool(x)
        x= torch.flatten(x, 1)
        x= self.linear(x)

def ResNeXt50():
    return ResNext(ResNextBlock, [3, 4, 6, 3], 10)
    
def ResNeXt101():
    return ResNext(ResNextBlock, [3, 4, 23, 3], 10)

def ResNeXt152():
    return ResNext(ResNextBlock, [3, 8, 36, 3], 10)


model= ResNeXt50()
rand= torch.randn((1,3,224,224))
export_onnx(model, rand, 'resnext50.onnx')