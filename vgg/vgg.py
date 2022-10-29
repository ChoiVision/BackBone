import sys
sys.path.append('')
import torch
import torch.nn as nn
from utils.conv import ConvBn, ConvBnAct
from utils.convert_onnx import export_onnx

class Vgg(nn.Module):
    def __init__(self, model, in_c, num_classes, init_weights=True):
        super().__init__()
        self.in_c= in_c
        self.conv= self.create_conv(VGG_types[model])
        self.fc= nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x= self.conv(x)
        x= x.view(-1, 512 * 7 * 7)
        x= self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def create_conv(self, architecture):
        layers= []
        in_c= self.in_c

        for x in architecture:
            if type(x)== int:
                out_c= x                
                layers+= [ConvBnAct(in_c, out_c, k= (3,3), s= (1,1))]
                in_c= x

            elif x == 'M':
                layers+= [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            
        return nn.Sequential(*layers)

VGG_types = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M'],
    'VGG13' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M'],
    'VGG16' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M'],
    'VGG19' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M']
}

def vgg11(num_classes):
    return Vgg('VGG11', 3, num_classes= num_classes, init_weights= True)  

def vgg13(num_classes):     
    return Vgg('VGG13', 3, num_classes= num_classes, init_weights= True)  

def vgg16(num_classes):     
    return Vgg('VGG16', 3, num_classes= num_classes, init_weights= True)  

def vgg19(num_classes):     
    return Vgg('VGG19', 3, num_classes= num_classes, init_weights= True)  

