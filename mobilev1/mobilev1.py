import torch
import torch.nn as nn
from utils.conv import ConvBn, ConvBnAct
from utils.block import DepthSeperableBlock

class MobileV1(nn.Module):
    def __init__(self, width_multiplier, num_classes):
        super().__init__()
        alpha= width_multiplier
        self.stem= ConvBnAct(in_c= 3, out_c= int( alpha * 32), k= 3, s= 2, p='same')

        #conv1
        self.conv1= DepthSeperableBlock(in_c= int(alpha * 32), out_c= int(alpha * 64), k= 3, s= 1, p= 'same')

        #conv2
        self.conv2= DepthSeperableBlock(in_c= int(alpha * 64), out_c= int(alpha * 128), k= 3, s=2, p='same')

        #conv3
        self.conv3= DepthSeperableBlock(in_c= int(alpha * 128), out_c= int(alpha * 128), k=3, s=1, p= 'same')

        #conv4
        self.conv4= DepthSeperableBlock(in_c= int(alpha * 128), out_c= int(alpha * 256), k=3, s=2, p= 'same')

        #conv5
        self.conv5= DepthSeperableBlock(in_c= int(alpha * 256), out_c= int(alpha * 256), k=3, s=1, p='same')

        #conv6
        self.conv6= DepthSeperableBlock(in_c= int(alpha * 256), out_c= int(alpha * 512), k=3, s=2, p= 'same')

        #conv7~11
        self.conv7= DepthSeperableBlock(in_c= int(alpha * 512), out_c= int(alpha * 512), k=3, s=1, p= 'same')
        self.conv8= DepthSeperableBlock(in_c= int(alpha * 512), out_c= int(alpha * 512), k=3, s=1, p= 'same')
        self.conv9= DepthSeperableBlock(in_c= int(alpha * 512), out_c= int(alpha * 512), k=3, s=1, p= 'same')
        self.conv10= DepthSeperableBlock(in_c= int(alpha * 512), out_c= int(alpha * 512), k=3, s=1, p= 'same')
        self.conv11= DepthSeperableBlock(in_c= int(alpha * 512), out_c= int(alpha * 512), k=3, s=1, p= 'same')

        #conv12
        self.conv12= DepthSeperableBlock(in_c= int(alpha * 512), out_c= int(alpha * 1024), k=3, s=2, p= 'same')

        #conv13
        self.conv13= DepthSeperableBlock(in_c=int(alpha * 1024), out_c= int(alpha * 1024), k=3, s=2, p= 'same')

        self.fc= nn.Linear(int(1024 * alpha), num_classes)
        #avg
        self.pool= nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x= self.stem(x)
        x= self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x= self.conv4(x)
        x= self.conv5(x)
        x= self.conv6(x)
        x= self.conv7(x)
        x= self.conv8(x)
        x= self.conv9(x)
        x= self.conv10(x)
        x= self.conv11(x)
        x= self.conv12(x)
        x= self.conv13(x)

        x= self.pool(x)

        x= x.view(x.size(0), -1)
        x= self.fc(x)

        return x

def mobilenet(alpha=1, class_num=100):
    return MobileV1(alpha, class_num)

rand= torch.randn((1, 3, 224, 224))
model= mobilenet()
res= model(rand)

print(res)

