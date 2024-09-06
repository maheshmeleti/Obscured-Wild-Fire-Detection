import torch
import torch.nn as nn
import torch.nn.functional as f

from decoder import Decoder

from conv_transformer_depth_scaling import Vid_Transformer
import torchvision.models as models

class seq_seg(nn.Module):
    def __init__(self, n_classes, seq_length):
        super(seq_seg, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = seq_length
        self.n_classes = n_classes
        
        self.encoder = nn.Sequential(*models.vgg16(pretrained=True).features[:27])
        
        self.seq_len = 20
        self.width = 32 
        self.height = 32

        in_channels = 512
        dim_head = 256 
        n_heads = 8 
        hidd_channles = 1024 
        out_channels = 512
        depth = 3

        self.feat_aggregator = Vid_Transformer(in_channels, dim_head, n_heads, hidd_channles, out_channels, self.width, self.height, self.seq_len, depth)

        self.decoder = Decoder(self.n_classes)

    
    def forward(self, x):
        
        batch_size, c, s, h, w = x.shape
        feat = torch.zeros((batch_size, 512, self.seq_len, self.width, self.height)).to(self.device)
        
        for i in range(self.seq_length):
            feat[:, :, i, :, :] = self.encoder(x[:,:,i,:,:])
        
        x = self.feat_aggregator(feat)
        x = self.decoder(x)

        return x
    
if __name__ == '__main__':
    from GPUtil import showUtilization as gpu_usage

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
    batch_size = 3
    n_classes = 2
    seq_len = 20
    model = seq_seg(n_classes, seq_len)
    model.to(device)

    print(gpu_usage())

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