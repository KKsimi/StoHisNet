from __future__ import print_function, division, absolute_import
from torch import nn
from Model.SwinTransformer_block import SwinTransformer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                               kernel_size=3, stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out
class BasicBlock_AFF(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock_AFF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

        inter_channels = int(out_channel)
        i_chan = out_channel
        self.local_att = nn.Sequential(
            nn.Conv2d(i_chan, inter_channels, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, i_chan, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(i_chan),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(i_chan, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, i_chan, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(i_chan),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #
        # out += identity
        # out = self.relu(out)

        t = out
        out += identity
        xl = self.local_att(out)
        xg = self.global_att(out)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * identity * wei + 2 * t * (1 - wei)

        # out = self.relu(out)

        return xo
class GFM(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(GFM, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7,stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock_AFF, 256, blocks_num[2], stride=2)
        self.layer4 = SwinTransformer(
            hidden_dim=96,
            layers=(4, 2, 6, 2),
            heads=(4, 6, 12, 24),
            channels=256,
            num_classes=num_classes,
            head_dim=24,
            window_size=7,
            downscaling_factors=(2, 2, 2, 2),
            relative_pos_embedding=True
        )


        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))   # output size=(1,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion,kernel_size=1,
                          stride=stride, bias=False)
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
