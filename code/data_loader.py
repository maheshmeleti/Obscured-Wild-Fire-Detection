import os
import random

import numpy as np
import json
from PIL import Image
import cv2

import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms


class SeqDataLoader(Dataset):
    def __init__(self, PNGPath, AnnPath, json_path, seq_length=10, n_classes=2, rgb_frames=False):
        self.PNGPath = PNGPath
        self.AnnPath = AnnPath
        self.json_path = json_path
        self.seq_length = seq_length
        self.folders = os.listdir(self.PNGPath)
        self.n_classes = n_classes
        self.rgb_frames_save = rgb_frames

        with open(self.json_path, 'r') as f:
            self.json_data = json.load(f)
        
    def __getitem__(self,index):
        selectFolder = self.folders[index]
        selectSegPixel = 1
        frames = self.json_data['videos'][selectFolder]['objects'][str(selectSegPixel)]['frames']

        #start = random.randint(1, len(frames) - self.seq_length)
        start = 0
        end = start + self.seq_length

        rgbFrames = np.zeros((3, self.seq_length, 512, 512))
        maskedSegFrame = np.zeros((1, self.seq_length , 512, 512))
        final_label = np.zeros((self.n_classes, 1 , 512, 512))

        if self.rgb_frames_save:
            orgFrames = np.zeros((self.seq_length, 512, 512, self.seq_length))
        
        for count, i in enumerate(range(start, end)):
            rgbpth = os.path.join(self.PNGPath, selectFolder + '/' + frames[i] + '.png')
            
            if self.rgb_frames_save:
                orgFrames[count, :, :, :] = (img_rgb)
            
            # print(rgbpth)
            img_rgb = cv2.imread(rgbpth)
            img_rgb = img_rgb[40:-20,340:-300,:] #[20:,340:-280,:]
            img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_CUBIC)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            
            img_rgb = img_rgb/255.0
            img_rgb = np.rollaxis(img_rgb, -1, 0)
            rgbFrames[:, count, :, :] = (img_rgb)

            segpth = os.path.join(self.AnnPath, selectFolder + '/' + frames[i] + '.png')
            mask = cv2.imread(segpth)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            maskedSegFrame[:,count,:,:] = mask/255.0
            merged_label = self.majority_label(maskedSegFrame)
            label = self.binarylabel(merged_label.squeeze(0), [0, 255])
            label = np.rollaxis(label, -1, 0)
            final_label[:,0,:,:] = label
            
        maskedSegFrame = torch.from_numpy(final_label)
        rgbFrames = torch.from_numpy(rgbFrames)
        
        if self.rgb_frames_save:
            return rgbFrames, maskedSegFrame, orgFrames

        return rgbFrames, maskedSegFrame
        
    def __len__(self):
        return len(self.folders)
    
    def binarylabel(self, im_label,classes):
        im_dims = im_label.shape

        lab=np.zeros([im_dims[0],im_dims[1],len(classes)],dtype="uint8")
        for index, class_index in enumerate(classes):

            lab[im_label==class_index, index] = 1

        return lab
    
    def majority_label(self, seq_mask):
        seq_length = seq_mask.shape[1]
        threshold = int(seq_length//2)
        
        summ = np.sum(seq_mask, axis=1)
        summ[summ >= threshold] = 255.0
        summ[summ < threshold] = 0
        
        return summ
