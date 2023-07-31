import sys
import os
import glob
import random

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
import torch.nn.functional as f
from torch.utils.data.dataset import Dataset
from torch.nn import init
from cbam import CBAM3D

# sys.path.insert(0, os.getcwd())
# sys.path.insert(0, os.path.join(os.getcwd(), 'code'))

# from data_loader import TrainDataLoader
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def up_conv3d(in_channels, out_channels):
    # up sample only feature dimensions
    return nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1,2,2), stride=(1,2,2)),
                         nn.BatchNorm3d(out_channels),
                         nn.ReLU(inplace=False),
                         )


def time_conv3d(in_channels, out_channels):
    
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(4, 1, 1), stride=1, padding=0, bias=False),
                         nn.BatchNorm3d(out_channels),
                         nn.ReLU(inplace=False)
                         )

def double_conv3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=False),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=False)
    )




class seq_seg(nn.Module):
    def __init__(self, n_classes, seq_length):
        super(seq_seg, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = seq_length
        self.n_classes = n_classes

        encoder = models.vgg16(pretrained=True).features

        for param in encoder.parameters():
            param.requires_grad = False

        self.block1 = nn.Sequential(*encoder[:6])  # outshape 128, 256, 256
        self.block2 = nn.Sequential(*encoder[6:13])  # 256, 128, 128
        self.block3 = nn.Sequential(*encoder[13:20])  # 512, 64, 64
        self.block4 = nn.Sequential(*encoder[20:27])  # 512, 32, 32
        self.block5 = nn.Sequential(*encoder[27:34])  # 512, 16, 16


        self.upconv_block5 = up_conv3d(512, 256)  #
        self.block5_attn = CBAM3D(256)
        self.upconv_block4 = up_conv3d(256, 128)
        self.block4_attn = CBAM3D(128)
        self.upconv_block3 = up_conv3d(128, 64)
        self.block3_attn = CBAM3D(64)
        self.upconv_block2 = up_conv3d(64, 32)
        self.block2_attn = CBAM3D(32)
        self.upconv_block1 = up_conv3d(32, n_classes)  # not intutive

        self.double_conv3D1 = double_conv3d(256 + 512, 256)
        self.double_conv1_attn = CBAM3D(256)
        self.double_conv3D2 = double_conv3d(128 + 512, 128)
        self.double_conv2_attn = CBAM3D(128)
        self.double_conv3D3 = double_conv3d(64 + 256, 64)
        self.double_conv3_attn = CBAM3D(64)
        self.double_conv3D4 = double_conv3d(32 + 128, 32)
        self.double_conv4_attn = CBAM3D(32)

        self.convT_f_1 = time_conv3d(n_classes, n_classes)  #17
        self.convT_f_2 = time_conv3d(n_classes, n_classes)  #14
        self.convT_f_3 = time_conv3d(n_classes, n_classes)  #11
        self.convT_f_4 = time_conv3d(n_classes, n_classes)  #8
        self.convT_f_5 = time_conv3d(n_classes, n_classes)  #5
        self.convT_f_6 = time_conv3d(n_classes, n_classes)
        self.convT_f_7 = nn.Conv3d(n_classes, n_classes, kernel_size=(2, 1, 1), stride=1, padding=0, bias=True) #1

    def forward(self, x):
        
        bs, c, s, h, w = x.shape
        feat1 = torch.zeros((bs, 128, self.seq_length, 256, 256)).to(self.device)
        feat2 = torch.zeros((bs, 256, self.seq_length, 128, 128)).to(self.device)
        feat3 = torch.zeros((bs, 512, self.seq_length, 64, 64)).to(self.device)
        feat4 = torch.zeros((bs, 512, self.seq_length, 32, 32)).to(self.device)
        feat5 = torch.zeros((bs, 512, self.seq_length, 16, 16)).to(self.device)

        for i in range(self.seq_length):
            block1_out = self.block1(x[:, :, i, :, :])
            feat1[:, :, i, :, :] = block1_out #self.irrn1(block1_out)

            block2_out = self.block2(block1_out)
            feat2[:, :, i, :, :] = block2_out #self.irrn2(block2_out)

            block3_out = self.block3(block2_out)
            feat3[:, :, i, :, :] = block3_out #self.irrn3(block3_out)

            block4_out = self.block4(block3_out)
            feat4[:, :, i, :, :] = block4_out #self.irrn4(block4_out)

            block5_out = self.block5(block4_out)
            feat5[:, :, i, :, :] = block5_out #self.irrn5(block5_out)

        # decoder
        # upsample feat5
        x = self.upconv_block5(feat5)  # -> 256 x seq_len x 32 x 32
        x = self.block5_attn(x)
        
        x = torch.cat([x, feat4], dim=1)  # -> (256 + 512) x seq_len x 32 x 32
        x = self.double_conv3D1(x)  # -> 256 x 32 x 32
        x = self.double_conv1_attn(x)

        x = self.upconv_block4(x)  # -> 128 x 64 x 64
        x = self.block4_attn(x)
        x = torch.cat([x, feat3], dim=1)  # -> 128 + 512 x 64 x 64
        x = self.double_conv3D2(x)  # -> 128 x 64 x 64
        x = self.double_conv2_attn(x)

        x = self.upconv_block3(x)  # 64 x 128 x 128
        x = self.block3_attn(x)
        x = torch.cat([x, feat2], dim=1)  # 64 + 256 x 128 x 128
        x = self.double_conv3D3(x)  # 64 x 128 x 128
        x = self.double_conv3_attn(x)

        x = self.upconv_block2(x)  # 32 x 256 x 256
        x = self.block2_attn(x)
        x = torch.cat([x, feat1], dim=1)  # 32 + 128 x 256 x 256
        x = self.double_conv3D4(x)  # 32 x 256 x 256
        x = self.double_conv4_attn(x)

        x = self.upconv_block1(x)  # n_classes x 512 x 512 (b, 32, seq_length, 512, 512)
        x = self.convT_f_1(x)
        x = self.convT_f_2(x)
        x = self.convT_f_3(x)
        x = self.convT_f_4(x)
        x = self.convT_f_5(x)
        x = self.convT_f_6(x)
        x = self.convT_f_7(x)

        return x

if __name__ == '__main__':
    class DiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(DiceLoss, self).__init__()

        def forward(self, inputs, targets, smooth=1):
            
    #         print(inputs.shape, targets.shape)
            #comment out if your model contains a sigmoid or equivalent activation layer
            inputs = f.sigmoid(inputs)       
            
            #flatten label and prediction tensors
            inputs = inputs[:, 1, :, :]
            targets = targets[:, 1, :, :]
            #print(inputs.shape)
            #print(targets.shape)
            inputs = inputs.reshape(-1) #inputs.view(-1)
            targets = targets.reshape(-1) #targets.view(-1)
            
            intersection = (inputs * targets).sum()                            
            dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
            
            return 1 - dice

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    n_classes = 2
    seq_len = 20
    model = seq_seg(n_classes, seq_len)
    model.to(device)

    Input = torch.randint(0, 255, (batch_size, 3, seq_len, 512, 512)).type(torch.FloatTensor).to(device)
    out = model(Input)
    target = torch.ones((batch_size,n_classes, 1, 512, 512)).to(device)
    # criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001)

    print('Out shape - ', out.shape)
    print('Target shape - ', target.shape)
    loss = criterion(out, target)
    print(f'loss - {loss.item()}')
    
    with torch.autograd.set_detect_anomaly(True):
        loss.backward()
        optimizer.step()
