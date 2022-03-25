import argparse
import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
from PIL import Image


# This is the bottleneck block without squueze and excitations. Will implement the squeeze and excitation here later
class Bottleneck_no_SE(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck_no_SE, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        #downsample if needed
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out+=identity
        
        return self.relu(out)
        

class ResNet(nn.Module):
    def __init__(self, block_type, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block_type, layer_list[0], planes=64)
        self.layer2 = self._make_layer(block_type, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(block_type, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(block_type, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block_type.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, block_type, blocks, planes, stride=1):
        downsample = None
        layers = []
        #print("Printing the self in_channels:",self.in_channels, planes*block_type.expansion)

        if stride != 1 or self.in_channels != planes*block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*block_type.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*block_type.expansion)
            )
            
        layers.append(block_type(self.in_channels, planes, downsample=downsample, stride=stride))
        self.in_channels = planes*block_type.expansion
        
        for num_layers in range(1,blocks):
            layers.append(block_type(self.in_channels, planes))
            
        return nn.Sequential(*layers)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck_no_SE, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck_no_SE, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck_no_SE, [3,8,36,3], num_classes, channels)



