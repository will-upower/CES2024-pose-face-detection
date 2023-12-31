import math

import torch
import torch.nn as nn

from torch.nn import functional as F
#from torchvision.models import resnet

from .resnet import resnet34

class BottleConv2d(nn.Module):

    def __init__(self, num_channels, kernel_size=3, dilation=1, relu=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 4, 1),
            nn.BatchNorm2d(num_channels // 4),
            nn.ReLU(inplace = True),

            nn.Conv2d(num_channels // 4, num_channels // 4, kernel_size, padding=(kernel_size * dilation - 1)//2, dilation=dilation),
            nn.BatchNorm2d(num_channels // 4),
            nn.ReLU(inplace = True),

            nn.Conv2d(num_channels // 4, num_channels, 1),
            nn.BatchNorm2d(num_channels),
        )
        if relu:
            self.relu = nn.ReLU(inplace = True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.layer(x) + x
        if self.relu is not None:
            return self.relu(x)
        else:
            return x

class DecoderBlock(nn.Module):

    def __init__(self, out_c, relu=True):
        super().__init__()

        self.conv = nn.Sequential(
            BottleConv2d(out_c, 3),
            BottleConv2d(out_c, 3, dilation=2),
            BottleConv2d(out_c, 5, dilation=2),
        )

        self.bypass1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1),
            nn.BatchNorm2d(out_c)
        )

        self.bypass2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1),
            nn.BatchNorm2d(out_c)
        )

        self.tee  = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1),
            nn.BatchNorm2d(out_c)
        )

        if relu:
            self.relu = nn.ReLU(inplace = True)
        else:
            self.relu = None

    def forward(self, x, lat=None):
        # todo: try x+c
        if lat is not None:
            if x.numel() > lat.numel(): raise Exception(x.shape, lat.shape)
            _, _, h, w = lat.size()
            x = F.interpolate(x, size=(int(h), int(w)))
            x = self.tee(x)

            x = x + lat
            x = F.relu(x)

        x = self.bypass2(self.conv(x)) + self.bypass1(x)

        if self.relu is not None:
            return self.relu(x)
        else:
            return x

def add_feats(x, y):
    if (x.numel() > y.numel()): x, y = y, x
    _, _, h, w = y.size()
    x = F.interpolate(x, size=(int(h), int(w)))
    return x + y

class HRFPN34(nn.Module):

    def __init__(self, out_channels):
        super().__init__()

        # kaneeun modified True->False
        base                = resnet34(pretrained = False)
        base_channels       = [64, 128, 256, 512]
        num_core            = 128

        self.encoder        = nn.Sequential(
                                nn.Sequential(
                                    base.conv1,
                                    base.bn1,
                                    base.relu,
                                    base.maxpool,
                                    base.layer1
                                    ),
                                base.layer2,
                                base.layer3,
                                base.layer4,
                                )

        self.lateral_start  = nn.Sequential(
                                nn.Conv2d(base_channels[-1], num_core, 1),
                                nn.BatchNorm2d(num_core),
                                nn.ReLU(inplace=True)
                                )

        self.decoder_start  = DecoderBlock(num_core)

        self.decoder        = nn.ModuleList([])
        self.lateral        = nn.ModuleList([])

        for i, in_channels in enumerate(reversed(base_channels[:-1])):
            self.decoder.append(DecoderBlock(num_core))
            self.lateral.append(nn.Sequential(
                                    nn.Conv2d(in_channels, num_core, 1),
                                    nn.BatchNorm2d(num_core)
                                    )
                                )

        self.classifier     = nn.Sequential(
                                nn.Conv2d(num_core, num_core, 3, padding=1),
                                nn.BatchNorm2d(num_core),
                                nn.ReLU(inplace = True),
                                nn.Conv2d(num_core, out_channels, 1),
                                )

        needs_init          = list(self.lateral_start.modules())
        needs_init          += list(self.decoder_start.modules())
        needs_init          += list(self.decoder.modules())
        needs_init          += list(self.lateral.modules())
        needs_init          += list(self.classifier.modules())

        for m in needs_init:
            if isinstance(m, nn.Conv2d):
                n   = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        results = []

        residuals = []
        for e in self.encoder:
            x = e(x)
            residuals.append(x)

        laterals = []
        for i, l in enumerate(self.lateral):
            laterals.append(l(residuals[-(i+2)]))

        x = self.lateral_start(x)
        x = self.decoder_start(x)

        for i, d in enumerate(self.decoder):
            #x = add_feats(laterals[i], x)
            x = d(x, laterals[i])
            if self.training:
                c = self.classifier(x)
                results.append(c)

        if not self.training:
            c = self.classifier(x)
            results.append(c)

        return results
