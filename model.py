# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class SiameseUNet(nn.Module):
    def __init__(self, c0=32):
        super().__init__()
        self.enc1 = ConvBlock(3, c0)
        self.enc2 = ConvBlock(c0, c0*2)
        self.enc3 = ConvBlock(c0*2, c0*4)
        self.enc4 = ConvBlock(c0*4, c0*8)
        self.up3 = ConvBlock(c0*8 + c0*4, c0*4)
        self.up2 = ConvBlock(c0*4 + c0*2, c0*2)
        self.up1 = ConvBlock(c0*2 + c0, c0)
        self.head = nn.Conv2d(c0, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        d3 = self.up3(torch.cat([F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.up2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.up1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return self.head(d1)
