import torch
import torch.nn as nn

class EndpointFCN(nn.Module):
    def __init__(self):
        super(EndpointFCN, self).__init__()
        
        # Contracting Path (Encoder)
        self.enc1 = self._conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2, 2)  # 256 x 256
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)  # 128 x 128
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)  # 64 x 64
        
        # Bottleneck
        self.center = self._conv_block(256, 512)
        
        # Expansive Path (Decoder) - Using Deconv (ConvTranspose2d)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256) 
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output Layer: 1 Channel for Heatmap
        self.final = nn.Conv2d(64, 1, kernel_size=1) 
        self.sigmoid = nn.Sigmoid() 

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        c = self.center(p3)
        
        # Decoder with Skip Connections
        u3 = self.up3(c)
        cat3 = torch.cat([u3, e3], dim=1) 
        d3 = self.dec3(cat3)
        
        u2 = self.up2(d3)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        
        return self.sigmoid(self.final(d1))