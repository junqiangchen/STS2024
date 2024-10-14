import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.GroupNorm(8, out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = unetConv2(out_size * 2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class UNet3d3PlusDeepSup(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=16, is_deconv=True, is_batchnorm=True,
                 deep_supervision=False):
        super(UNet3d3PlusDeepSup, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.deep_supervision = deep_supervision

        self.filters = [self.features, self.features * 2, self.features * 4, self.features * 8, self.features * 16]

        # -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, self.filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = unetConv2(self.filters[0], self.filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = unetConv2(self.filters[1], self.filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = unetConv2(self.filters[2], self.filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = unetConv2(self.filters[3], self.filters[4], self.is_batchnorm)

        # -------------Decoder--------------
        self.CatChannels = self.filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(self.filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv3d(self.filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv3d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.GroupNorm(8, self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv3d(self.filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv3d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.GroupNorm(8, self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv3d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv3d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.GroupNorm(8, self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv3d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.GroupNorm(8, self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=(32, 32, 32), mode='trilinear')  ###
        self.upscore5 = nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear')
        self.upscore4 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')
        self.upscore3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')
        self.upscore2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv3 = nn.Conv3d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv4 = nn.Conv3d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv5 = nn.Conv3d(self.filters[4], self.out_channels, 3, padding=1)

    def forward(self, inputs):
        # -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        # -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        if self.deep_supervision:
            d5 = self.outconv5(hd5)
            d5 = self.upscore5(d5)  # 16->256
            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4)  # 32->256
            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3)  # 64->256
            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2)  # 128->256
            d1 = self.outconv1(hd1)  # 256
            if self.out_channels == 1:
                d1_ouput = torch.sigmoid(d1)
            if self.out_channels > 1:
                d1_ouput = torch.softmax(d1, dim=1)
            if self.out_channels == 1:
                d2_ouput = torch.sigmoid(d2)
            if self.out_channels > 1:
                d2_ouput = torch.softmax(d2, dim=1)
            if self.out_channels == 1:
                d3_ouput = torch.sigmoid(d3)
            if self.out_channels > 1:
                d3_ouput = torch.softmax(d3, dim=1)
            if self.out_channels == 1:
                d4_ouput = torch.sigmoid(d4)
            if self.out_channels > 1:
                d4_ouput = torch.softmax(d4, dim=1)
            if self.out_channels == 1:
                d5_ouput = torch.sigmoid(d5)
            if self.out_channels > 1:
                d5_ouput = torch.softmax(d5, dim=1)
            return d1, d2, d3, d4, d5, d1_ouput, d2_ouput, d3_ouput, d4_ouput, d5_ouput
        else:
            out_logit = self.outconv1(hd1)
            if self.out_channels == 1:
                output = torch.sigmoid(out_logit)
            if self.out_channels > 1:
                output = torch.softmax(out_logit, dim=1)
            return out_logit, output


'''
    UNet 3+ with deep supervision and class-guided module
'''


class UNet3d3PlusDeepSupCGM(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=16, is_deconv=True, is_batchnorm=True,
                 deep_supervision=False):
        super(UNet3d3PlusDeepSupCGM, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.deep_supervision = deep_supervision

        self.filters = [self.features, self.features * 2, self.features * 4, self.features * 8, self.features * 16]

        # -------------Encoder--------------
        self.conv1 = unetConv2(self.in_channels, self.filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = unetConv2(self.filters[0], self.filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = unetConv2(self.filters[1], self.filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = unetConv2(self.filters[2], self.filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = unetConv2(self.filters[3], self.filters[4], self.is_batchnorm)

        # -------------Decoder--------------
        self.CatChannels = self.filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(self.filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv3d(self.filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv3d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.GroupNorm(8, self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv3d(self.filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv3d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.GroupNorm(8, self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv3d(self.filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv3d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.GroupNorm(8, self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(self.filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv3d(self.filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.GroupNorm(8, self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.GroupNorm(8, self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=(32, 32, 32), mode='trilinear')
        self.upscore5 = nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear')
        self.upscore4 = nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear')
        self.upscore3 = nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear')
        self.upscore2 = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # DeepSup
        self.outconv1 = nn.Conv3d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv2 = nn.Conv3d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv3 = nn.Conv3d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv4 = nn.Conv3d(self.UpChannels, self.out_channels, 3, padding=1)
        self.outconv5 = nn.Conv3d(self.filters[4], self.out_channels, 3, padding=1)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv3d(self.filters[4], 2, 1),
            nn.AdaptiveMaxPool3d(1),
            nn.Softmax(dim=1))

    def dotProduct(self, seg, cls):
        B, N, D, H, W = seg.size()
        seg = seg.view(B, N, D * H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, D, H, W)
        return final

    def forward(self, inputs):
        # -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        # -------------Classification-------------
        cls_branch = self.cls(hd5).squeeze(4).squeeze(3).squeeze(2)  # (B,N,1,1,1)->(B,N)
        cls_branch_max = cls_branch.argmax(dim=1)
        cls_branch_max = cls_branch_max[:, np.newaxis].float()

        # -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels
        if self.deep_supervision:
            d5 = self.outconv5(hd5)
            d5 = self.upscore5(d5)  # 16->256
            d4 = self.outconv4(hd4)
            d4 = self.upscore4(d4)  # 32->256
            d3 = self.outconv3(hd3)
            d3 = self.upscore3(d3)  # 64->256
            d2 = self.outconv2(hd2)
            d2 = self.upscore2(d2)  # 128->256
            d1 = self.outconv1(hd1)  # 256

            d1 = self.dotProduct(d1, cls_branch_max)
            d2 = self.dotProduct(d2, cls_branch_max)
            d3 = self.dotProduct(d3, cls_branch_max)
            d4 = self.dotProduct(d4, cls_branch_max)
            d5 = self.dotProduct(d5, cls_branch_max)

            if self.out_channels == 1:
                d1_ouput = torch.sigmoid(d1)
            if self.out_channels > 1:
                d1_ouput = torch.softmax(d1, dim=1)
            if self.out_channels == 1:
                d2_ouput = torch.sigmoid(d2)
            if self.out_channels > 1:
                d2_ouput = torch.softmax(d2, dim=1)
            if self.out_channels == 1:
                d3_ouput = torch.sigmoid(d3)
            if self.out_channels > 1:
                d3_ouput = torch.softmax(d3, dim=1)
            if self.out_channels == 1:
                d4_ouput = torch.sigmoid(d4)
            if self.out_channels > 1:
                d4_ouput = torch.softmax(d4, dim=1)
            if self.out_channels == 1:
                d5_ouput = torch.sigmoid(d5)
            if self.out_channels > 1:
                d5_ouput = torch.softmax(d5, dim=1)
            return d1, d2, d3, d4, d5, d1_ouput, d2_ouput, d3_ouput, d4_ouput, d5_ouput
        else:
            out_logit = self.outconv1(hd1)  # 256
            out_logit = self.dotProduct(out_logit, cls_branch_max)
            if self.out_channels == 1:
                output = torch.sigmoid(out_logit)
            if self.out_channels > 1:
                output = torch.softmax(out_logit, dim=1)
            return out_logit, output


if __name__ == "__main__":
    net = UNet3d3PlusDeepSup(1, 3, 16, deep_supervision=True)
    in1 = torch.rand((1, 1, 64, 256, 256))
    out = net(in1)
    for i in range(len(out)):
        print(out[i].size())
