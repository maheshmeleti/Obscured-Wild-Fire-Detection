from audioop import bias
from torch import nn
import torch

class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False))

    def forward(self, x):
        return self.net(x)

class Up_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=False),
                         )
        
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes

        self.up_conv1 = Up_Conv(512, 512) # 64 x 64
        self.double_conv1 = Double_Conv(512, 256)

        self.up_conv2 = Up_Conv(256, 256) # 128 x 128
        self.double_conv2 = Double_Conv(256, 128)

        self.up_conv3 = Up_Conv(128, 128) # 256 x 256
        self.double_conv3 = Double_Conv(128, 64)

        self.up_conv4 = Up_Conv(64, 64) # 512 x 512
        self.double_conv4 = Double_Conv(64, 32)

        self.double_conv5 = Double_Conv(32, 16)

        self.final_conv = nn.Conv2d(16, self.n_classes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):

        x = self.up_conv1(x)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = self.double_conv4(x)

        x = self.double_conv5(x)

        x = self.final_conv(x)

        x = x.unsqueeze(2)
        return x
    
if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 8
    n_classes = 2
    seq_len = 20
    feat = torch.randn((batch_size, 512, 16, 16)).to(device)

    decoder = Decoder(n_classes).to(device)

    out = decoder(feat)

    print(out.shape)


