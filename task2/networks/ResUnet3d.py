import torch
import torch.nn as nn
from collections import OrderedDict


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, name, prob=0.2):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=ch_out)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),),
            (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=ch_out)),
            (name + "droupout2", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, name, prob=0.2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(OrderedDict([
            (name + "upsample", nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),),
            (name + "conv1", nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=ch_out)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, name, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=ch_out)),
            (name + "relu1", nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, name, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(OrderedDict([
            (name + "Recurrent_block1", Recurrent_block(ch_out, name='rec1', t=t)),
            (name + "Recurrent_block2", Recurrent_block(ch_out, name='rec2', t=t)),
        ]))
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class ResUNet3d(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=16, t=2):
        super(ResUNet3d, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = t

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=(2, 2, 2))

        self.RRCNN1 = RRCNN_block(ch_in=self.in_channels, ch_out=self.features, name='RRCNN1', t=self.t)
        self.RRCNN2 = RRCNN_block(ch_in=self.features, ch_out=self.features * 2, name='RRCNN2', t=self.t)
        self.RRCNN3 = RRCNN_block(ch_in=self.features * 2, ch_out=self.features * 4, name='RRCNN3', t=self.t)
        self.RRCNN4 = RRCNN_block(ch_in=self.features * 4, ch_out=self.features * 8, name='RRCNN4', t=self.t)
        self.RRCNN5 = RRCNN_block(ch_in=self.features * 8, ch_out=self.features * 16, name='RRCNN5', t=self.t)
        self.Up5 = up_conv(ch_in=self.features * 16, ch_out=self.features * 8, name='up5')
        self.Up_RRCNN5 = RRCNN_block(ch_in=self.features * 16, ch_out=self.features * 8, name='Up_RRCNN5', t=self.t)
        self.Up4 = up_conv(ch_in=self.features * 8, ch_out=self.features * 4, name='up4')
        self.Up_RRCNN4 = RRCNN_block(ch_in=self.features * 8, ch_out=self.features * 4, name='Up_RRCNN4', t=self.t)
        self.Up3 = up_conv(ch_in=self.features * 4, ch_out=self.features * 2, name='up3')
        self.Up_RRCNN3 = RRCNN_block(ch_in=self.features * 4, ch_out=self.features * 2, name='Up_RRCNN3', t=self.t)
        self.Up2 = up_conv(ch_in=self.features * 2, ch_out=self.features, name='up2')
        self.Up_RRCNN2 = RRCNN_block(ch_in=self.features * 2, ch_out=self.features, name='Up_RRCNN2', t=self.t)
        self.Conv_1x1 = nn.Conv3d(self.features, self.out_channels, kernel_size=1)

    def forward(self, x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)
        if self.out_channels == 1:
            d1_ouput = torch.sigmoid(d1)
        if self.out_channels > 1:
            d1_ouput = torch.softmax(d1, dim=1)
        return d1, d1_ouput


if __name__ == "__main__":
    net = ResUNet3d(1, 5, 16)
    in1 = torch.rand((1, 1, 64, 256, 256))
    out = net(in1)
    for i in range(len(out)):
        print(out[i].size())
