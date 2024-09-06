import os
from torch.utils.data import DataLoader
from torch import nn
import torch
import torch.nn.functional as f
# from Model_Conv_Transformer import seq_seg
from Model_DS_Conv_Transformer import seq_seg
from data_loader import SeqDataLoader
from Trainer import Trainer

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


PNG_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_and_neg_data/splitted/train/PNG_Images/'
Annot_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_and_neg_data/splitted/train/Annotations/'
json_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_and_neg_data/splitted/train/meta.json'

val_PNG_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_and_neg_data/splitted/val/PNG_Images/'
val_Annot_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_and_neg_data/splitted/val/Annotations/'
val_json_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_and_neg_data/splitted/val/meta.json'

# PNG_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_data/splitted/train/PNG_Images/'
# Annot_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_data/splitted/train/Annotations/'
# json_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_data/splitted/train/meta.json'

# val_PNG_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_data/splitted/val/PNG_Images/'
# val_Annot_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_data/splitted/val/Annotations/'
# val_json_path = '../../../datasets/smoke_pattern/imp_vid_parts/Paper2_Data/pos_data/splitted/val/meta.json'

# experiment_name = 'pos_data_cvt'
experiment_name =  'dep_scl_pos_neg_data_cvt'

train_logs_path = experiment_name
check_points_path = experiment_name
out_save_path = os.path.join(experiment_name, 'output')

make_dir(check_points_path)
make_dir(out_save_path)

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

batch_size = 4
seq_length=20
n_classes = 2
data_loader = SeqDataLoader(PNG_path, Annot_path, json_path, seq_length=seq_length, n_classes=2)
val_data_loader = SeqDataLoader(val_PNG_path, val_Annot_path, val_json_path, seq_length=seq_length, n_classes=2)
train_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data_loader, batch_size=batch_size, shuffle=True)

model = seq_seg(n_classes, seq_length)


#criterion = nn.CrossEntropyLoss()
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=0.1)

n_epochs = 300
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200, 250], gamma=0.1) # 20 seq
trainer = Trainer(train_loader, model, criterion, optimizer, n_epochs, seq_length, scheduler, check_points_path, out_save_path=out_save_path,  val_loader = val_loader)
losses = trainer.run_trainer()
