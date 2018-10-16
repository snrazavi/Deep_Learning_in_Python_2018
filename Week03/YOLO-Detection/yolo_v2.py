import torch
import torch.nn as nn


class Conv2D(nn.Module):
    """ Conv2D + Batch Normalization + Leaky ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=False, activation=True):
        super(Conv2D, self).__init__()
        padding = (kernel_size - 1) // 2 if padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if activation else lambda x: x

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# reorg layer
class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        s = self.stride
        B, C, H, W = x.size()
        h, w = H // s, W // s
        x = x.view(B, C, h, s, w, s).transpose(3, 4).contiguous()
        x = x.view(B, C, h * w, s * s).transpose(2, 3).contiguous()
        x = x.view(B, C, s * s, h, w).transpose(1, 2).contiguous()
        return x.view(B, s * s * C, h, w)


# Darknet19
class Darknet(nn.Module):
    def __init__(self):
        super(Darknet, self).__init__()
        
        self.main1 = nn.Sequential(
            Conv2D(3, 32, 3, padding=True),
            nn.MaxPool2d(2, stride=2),
            
            Conv2D(32, 64, 3, padding=True),
            nn.MaxPool2d(2, stride=2),
            
            Conv2D(64, 128, 3, padding=True),
            Conv2D(128, 64, 1, padding=True),
            Conv2D(64, 128, 3, padding=True),
            nn.MaxPool2d(2, stride=2),
            
            Conv2D(128, 256, 3, padding=True),
            Conv2D(256, 128, 1, padding=True),
            Conv2D(128, 256, 3, padding=True),
            nn.MaxPool2d(2, stride=2),
            
            Conv2D(256, 512, 3, padding=True),
            Conv2D(512, 256, 1, padding=True),
            Conv2D(256, 512, 3, padding=True),
            Conv2D(512, 256, 1, padding=True),
            Conv2D(256, 512, 3, padding=True)   # 512 x 38 x 38
        )
        
        self.main2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            Conv2D(512, 1024, 3, padding=True),
            Conv2D(1024, 512, 1, padding=True),
            Conv2D(512, 1024, 3, padding=True),
            Conv2D(1024, 512, 1, padding=True),
            Conv2D(512, 1024, 3, padding=True),  # 1024 x 19 x 19   
        )
        
    def forward(self, x):
        x1 = self.main1(x)
        x2 = self.main2(x1)
        return x2, x1
    

class YoloNet(nn.Module):
    def __init__(self, num_classes, anchors):
        super(YoloNet, self).__init__()
        
        # Draknet
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors) // 2
        self.darknet = Darknet()    
        
        self.conv1 = nn.Sequential(
            Conv2D(1024, 1024, 3, padding=True),
            Conv2D(1024, 1024, 3, padding=True)  # 1024 x 19 x 19
        )
        
        
        self.conv2 = nn.Sequential(
            Conv2D(512, 64, 1, padding=True),  # 64     x 38 x 38
            Reorg(2)                           # (4x64) x 19 x 19
        )

        self.conv = nn.Sequential(
            Conv2D(1280, 1024, 3, padding=True), # (256 + 1024) x 19 x 19
            nn.Conv2d(1024, self.num_anchors * (self.num_classes + 5), 1) # 425 x 19 x 19
        )
            
    def forward(self, x):
        x1, x2 = self.darknet(x)    # (1024, 19, 19), (512, 38, 38) 
        x1 = self.conv1(x1)         # (1024, 19, 19)
        x2 = self.conv2(x2)         # (4x64, 19, 19)
        x = torch.cat([x2, x1], 1)  # (1280, 19, 19)
        x = self.conv(x)            # (425,  19, 19)
        return x