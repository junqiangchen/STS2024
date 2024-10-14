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


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, name):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(F_g, F_int, kernel_size=1),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=F_int)),
        ]))

        self.W_x = nn.Sequential(OrderedDict([
            (name + "conv2", nn.Conv3d(F_l, F_int, kernel_size=1),),
            (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=F_int)),
        ]))

        self.psi = nn.Sequential(OrderedDict([
            (name + "conv3", nn.Conv3d(F_int, 1, kernel_size=1),),
            (name + "sigmod", nn.Sigmoid()),
        ]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AGUNet3d(nn.Module):
    def __init__(self, img_ch, output_ch, init_feature=16):
        super(AGUNet3d, self).__init__()
        self.features = init_feature
        self.in_channels = img_ch
        self.out_channels = output_ch

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=self.features, name='enc1')
        self.Conv2 = conv_block(ch_in=self.features, ch_out=self.features * 2, name='enc2')
        self.Conv3 = conv_block(ch_in=self.features * 2, ch_out=self.features * 4, name='enc3')
        self.Conv4 = conv_block(ch_in=self.features * 4, ch_out=self.features * 8, name='enc4')
        self.Conv5 = conv_block(ch_in=self.features * 8, ch_out=self.features * 16, name='bottleneck')

        self.Up5 = up_conv(self.features * 16, ch_out=self.features * 8, name='up5')
        self.Att5 = Attention_block(F_g=self.features * 8, F_l=self.features * 8, F_int=self.features * 8,
                                    name='att5')
        self.Up_conv5 = conv_block(ch_in=self.features * 16, ch_out=self.features * 8, name='dec5')

        self.Up4 = up_conv(ch_in=self.features * 8, ch_out=self.features * 4, name='up4')
        self.Att4 = Attention_block(F_g=self.features * 4, F_l=self.features * 4, F_int=self.features * 2, name='att4')
        self.Up_conv4 = conv_block(ch_in=self.features * 8, ch_out=self.features * 4, name='dec4')

        self.Up3 = up_conv(ch_in=self.features * 4, ch_out=self.features * 2, name='up3')
        self.Att3 = Attention_block(F_g=self.features * 2, F_l=self.features * 2, F_int=self.features, name='att3')
        self.Up_conv3 = conv_block(ch_in=self.features * 4, ch_out=self.features * 2, name='dec3')

        self.Up2 = up_conv(ch_in=self.features * 2, ch_out=self.features, name='up2')
        self.Att2 = Attention_block(F_g=self.features, F_l=self.features, F_int=self.features // 2, name='att2')
        self.Up_conv2 = conv_block(ch_in=self.features * 2, ch_out=self.features, name='dec2')

        self.Conv_1x1 = nn.Conv3d(self.features, output_ch, kernel_size=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out_logit = self.Conv_1x1(d2)
        if self.out_channels == 1:
            output = torch.sigmoid(out_logit)
        if self.out_channels > 1:
            output = torch.softmax(out_logit, dim=1)
        return out_logit, output


if __name__ == "__main__":
    net = AGUNet3d(1, 4, 16)
    in1 = torch.rand((1, 1, 64, 256, 256))
    out = net(in1)
    for i in range(len(out)):
        print(out[i].size())
