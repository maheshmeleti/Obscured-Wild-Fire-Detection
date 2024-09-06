import torch
from torch.autograd import Variable
import torch.nn as nn
from einops import rearrange, repeat

class Attention(nn.Module):
    def __init__(self, dim_head, n_heads, in_channels, width, height):
        super().__init__()
        self.in_channels = in_channels
        self.attn_channels = dim_head*n_heads
        self.width = width
        self.height = height
        stride = 1
        
        self.key_conv = nn.Conv3d(self.in_channels, self.attn_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0,1,1))
        self.query_conv = nn.Conv3d(self.in_channels, self.attn_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0,1,1))
        self.value_conv = nn.Conv3d(self.in_channels, self.attn_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0,1,1))
        
        self.scale = (width*height)**(-0.5)
        self.sigmoid = nn.Sigmoid()
        
        self.out_conv = nn.Conv3d(self.attn_channels, self.in_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0,1,1))
        
    def forward(self, x):
        
        key = self.key_conv(x)
        query = self.query_conv(x)
        value = self.value_conv(x)
        
        key = rearrange(key, 'b c s h w -> b c s (h w)')
        query = rearrange(query, 'b c s h w -> b c s (h w)')
        value = rearrange(value, 'b c s h w -> b c s (h w)')
        
        attention_mat = self.sigmoid(torch.matmul(query, key.transpose(-1, -2))*self.scale)
        attention = torch.matmul(attention_mat, value)
        out = rearrange(attention, 'b c s (h w) -> b c s h w', h = self.height, w = self.width)
        
        return self.out_conv(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.BatchNorm3d(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)    
    
class Forward(nn.Module):
    def __init__(self, in_channels, hidd_channles):
        super().__init__()
        self.in_channels = in_channels
        self.hidd_channles = hidd_channles
        stride = 1

        self.net = nn.Sequential(nn.Conv3d(self.in_channels, self.hidd_channles, kernel_size=(1, 3, 3), stride=stride, padding=(0,1,1)),
                                 nn.ReLU(),
                                 nn.BatchNorm3d(self.hidd_channles),
                                 nn.Conv3d(self.hidd_channles,self.in_channels, kernel_size=(1, 3, 3), stride=stride, padding=(0,1,1)),
                                 nn.ReLU())
        
    def forward(self, x):
        
        x = self.net(x)
        
        return x

#dim_head, n_heads, in_channels, width, height

class Conv_Transfomer(nn.Module):
    def __init__(self, in_channels, dim_head, n_heads, hidd_channles, width, height, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([])
        

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(in_channels, Attention(dim_head, n_heads, in_channels, width, height)),
                PreNorm(in_channels, Forward(in_channels, hidd_channles))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        return x
    
# in_channels, dim_head, n_heads, hidd_channles, width, height, depth=2

class Vid_Transformer(nn.Module):
    def __init__(self, in_channels, dim_head, n_heads, hidd_channles, out_channels, width, height, seq_len, depth=1):
        super().__init__()
        
        self.in_channels =  in_channels
        self.dim_head = dim_head
        self.n_heads= n_heads
        self.hidd_channles = hidd_channles
        self.out_channels =  out_channels
        self.width = width
        self.height = height
        self.seq_len = seq_len
        self.depth = depth
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.seq_len, self.width, self.height))
        
        self.cvt = Conv_Transfomer(self.in_channels, self.dim_head, self.n_heads, self.hidd_channles, self.width, self.height, depth=2)
        
        self.final_conv = nn.Sequential(nn.BatchNorm3d(self.in_channels), 
                                       nn.Conv3d(self.in_channels, self.out_channels, kernel_size=(1, 3, 3), stride=1, padding=(0,1,1)))
        
    def forward(self, x):
        
        b, c, s, h, w = x.shape
        pos_embeddings = repeat(self.pos_embedding, '1 1 s w h -> b c s w h', b = b, c = c) 
        
        x += pos_embeddings
        
        x = self.cvt(x)
        x = self.final_conv(x)
        
        x = x.mean(dim=2)
        
        return x
    
#in_channels, dim_head, n_heads, hidd_channles, out_channels, width, height, seq_len, depth=1

if __name__ == '__main__':
    from GPUtil import showUtilization as gpu_usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1
    seq_len = 20
    width = 32 
    height = 32

    in_channels = 512
    dim_head = 256 
    n_heads = 8 
    hidd_channles = 1024 
    out_channels = 512
    depth = 3
    
    feat = torch.rand((batch_size, in_channels, seq_len, width, height)).to(device)

    vt = Vid_Transformer(in_channels, dim_head, n_heads, hidd_channles, out_channels, width, height, seq_len, depth).to(device)
    print(gpu_usage())

    out = vt(feat)

    print(out.shape)