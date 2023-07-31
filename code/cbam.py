import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F

class channel_Gate3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(channel_Gate3D, self).__init__()
        self.in_channels = in_channels
        self.pool_types = ['avg', 'max']
        self.avg_pool = nn.AdaptiveAvgPool3d((None,1,1))
        self.max_pool =  nn.AdaptiveMaxPool3d((None, 1,1))
        self.conv_block = nn.Sequential(
                 nn.Conv3d(in_channels, in_channels // reduction_ratio, 1),
                 nn.ReLU(inplace=True),
                 nn.Conv3d(in_channels // reduction_ratio, in_channels, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = self.avg_pool(x)
                channel_att_raw = self.conv_block(avg_pool)
            
            elif pool_type == 'max':
                max_pool = self.max_pool(x)
                channel_att_raw = self.conv_block(max_pool)
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
                
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        return x*self.sigmoid(channel_att_sum)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=(1, kernel_size,kernel_size), stride=(1, stride, stride), padding=(0,padding,padding), dilation=dilation, groups=groups, bias=bias)
        #self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_gate3D(nn.Module):
    def __init__(self):
        super(spatial_gate3D, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale


class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM3D, self).__init__()
        self.channel_gate = channel_Gate3D(in_channels, reduction_ratio=16, pool_types=['avg', 'max'])
        self.spatian_gate = spatial_gate3D()
    
    def forward(self, x):
        x_out = self.channel_gate(x)
        x_out = self.spatian_gate(x_out)
        
        return x_out
    

if __name__ == '__main__':
    batch_size = 10
    seq_len = 20
    feat = 32
    fake_input = torch.randint(0, 255, (batch_size, feat, seq_len, 32, 32)).type(torch.FloatTensor)

    print(fake_input.shape)
    cbam_attn = CBAM3D(32)
    cbam_attn_out = cbam_attn(fake_input)
    print(cbam_attn_out.shape)