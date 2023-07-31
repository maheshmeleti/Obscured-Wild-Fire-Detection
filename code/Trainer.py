from torch.autograd import Variable
import time
import cv2
import os
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler

class Trainer():
    def __init__(self, train_loader, model, criterion, optimizer, n_epochs, seq_length, 
                 scheduler, check_points_path, val_loader = None, out_save_path=None, rgbsave_path=None, masksave_path=None, check_point_freq = 3):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.epoch_loss = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.check_points_path = check_points_path
        if out_save_path:
            self.outsave_path = out_save_path
        if rgbsave_path:
            self.rgbsave_path = rgbsave_path
        if masksave_path:
            self.masksave_path = masksave_path
        
        self.seq_length = seq_length
        self.scheduler = scheduler
        self.check_point_freq = check_point_freq
        self.scaler = GradScaler()
        
    def run_trainer(self):
        
        self.model.to(self.device)
        print('Training Started....')
        with open(os.path.join(self.outsave_path, "Train Logs.txt"), "a") as f:
                f.write('Training Started....\n')
                f.write('Length of train loader {}\n'.format(len(self.train_loader)))
                
        for epoch in range(self.n_epochs):
            
            start_time = time.time()
            epoch_loss = self._train()
            if self.val_loader:
                val_loss = self._val()
            else:
                val_loss = 0
            end_time = time.time()
            self.epoch_loss.append(epoch_loss)

            log_string = "Epoch - {} Train Loss - {} Val Loss - {} lr - {} time_taken - {} secs\n".format(epoch, epoch_loss, val_loss, self.optimizer.param_groups[0]["lr"], (end_time - start_time))
            with open(os.path.join(self.outsave_path, "Train Logs.txt"), "a") as f:
                f.write(log_string)
            print(log_string)
            if epoch%self.check_point_freq:
                torch.save(self.model.state_dict(),os.path.join(self.check_points_path, 'epoch-{} loss-{:.6f}.pth'.format(epoch, epoch_loss)))

            if epoch_loss < 0.09: # stopping criteria
                break
            
        return self.epoch_loss
    
    def _train(self):
        self.model.train()
        batch_loss = []
        for batch_idx, (rgbFrames, SegFrame) in enumerate(self.train_loader):
            rgb = Variable(rgbFrames).type(torch.FloatTensor).to(self.device)
            mask = Variable(SegFrame).type(torch.FloatTensor).to(self.device)
            # self.save_rgb(orgFrames)
            # self.save_mask(mask)
            # print(f'loss - {loss.item()}')

            with autocast():
                out = self.model(rgb)
                loss = self.criterion(out, mask)
            self.save_output(out)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            batch_loss.append(loss.item())
            self.scaler.update()

        self.scheduler.step()
        
        return np.mean(batch_loss)
    
    def _val(self):
        self.model.eval()
        batch_loss = []
        for batch_idx, (rgbFrames, SegFrame) in enumerate(self.val_loader):
            rgb = Variable(rgbFrames).type(torch.FloatTensor).to(self.device)
            mask = Variable(SegFrame).type(torch.FloatTensor).to(self.device)
            with autocast():
                out = self.model(rgb)
                loss = self.criterion(out, mask)
            batch_loss.append(loss.item())

        return np.mean(batch_loss)
    
    
    
    # def _train(self):
    #     batch_loss = []
    #     # for batch_idx, (rgbFrames, SegFrame, orgFrames) in enumerate(self.train_loader):
    #     for batch_idx, (rgbFrames, SegFrame) in enumerate(self.train_loader):
    #         rgb = Variable(rgbFrames).type(torch.FloatTensor).to(self.device)
    #         mask = Variable(SegFrame).type(torch.FloatTensor).to(self.device)
    #         out = self.model(rgb)
    #         self.save_output(out)
    #         # self.save_rgb(orgFrames)
    #         # self.save_mask(mask)

    #         loss = self.criterion(out, mask)
    #         # print(f'loss - {loss.item()}')
            
    #         with torch.autograd.set_detect_anomaly(True):
    #             loss.backward()
    #             self.optimizer.step()
    #             batch_loss.append(loss.item())

    #     self.scheduler.step()
        
    #     return np.mean(batch_loss)
    
    def save_rgb(self, rgb):
        b,s,h,w, c = rgb.shape
        for i in range(s):
            rgb_i = rgb[0,i,:,:,:]
            rgb_i = rgb_i.data.cpu().numpy()
#             print(rgb_i.shape)
            #rgb_i = np.moveaxis(rgb_i, 0, -1)
            #print(rgb_i.shape)
            rgb_i = rgb_i[:,:,::-1]
            cv2.imwrite(os.path.join(self.rgbsave_path, f'{i}.png'), rgb_i)
            
    def save_mask(self, mask): 
        mask = mask.argmax(1)
        b, s, h, w = mask.shape
        for i in range(s):
            mask_i = mask[0,i,:,:]
            mask_i = mask_i.data.cpu().numpy()
            mask_i = mask_i*255.0
            cv2.imwrite(os.path.join(self.masksave_path, f'{i}.png'), mask_i)
        
    
    def save_output(self, out):
        out = out.softmax(axis=1)
        out = out.argmax(axis=1)
#         print(out.shape)
        b,s,h,w = out.shape
        for i in range(s):
            out_i = out[0,i,:,:]
            out_i = out_i.data.cpu().numpy()
            out_i = out_i*255.0
#             print(np.unique(out_i))
            cv2.imwrite(os.path.join(self.outsave_path, f'{i}.png'), out_i)