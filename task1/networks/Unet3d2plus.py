import torch
from torch import nn


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class NestedUNet3d(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=16, deep_supervision=False):
        super(NestedUNet3d, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_filter = [self.features, self.features * 2, self.features * 4, self.features * 8, self.features * 16]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        self.conv0_0 = VGGBlock(self.in_channels, self.nb_filter[0], self.nb_filter[0])
        self.conv1_0 = VGGBlock(self.nb_filter[0], self.nb_filter[1], self.nb_filter[1])
        self.conv2_0 = VGGBlock(self.nb_filter[1], self.nb_filter[2], self.nb_filter[2])
        self.conv3_0 = VGGBlock(self.nb_filter[2], self.nb_filter[3], self.nb_filter[3])
        self.conv4_0 = VGGBlock(self.nb_filter[3], self.nb_filter[4], self.nb_filter[4])

        self.conv0_1 = VGGBlock(self.nb_filter[0] + self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        self.conv1_1 = VGGBlock(self.nb_filter[1] + self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])
        self.conv2_1 = VGGBlock(self.nb_filter[2] + self.nb_filter[3], self.nb_filter[2], self.nb_filter[2])
        self.conv3_1 = VGGBlock(self.nb_filter[3] + self.nb_filter[4], self.nb_filter[3], self.nb_filter[3])

        self.conv0_2 = VGGBlock(self.nb_filter[0] * 2 + self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        self.conv1_2 = VGGBlock(self.nb_filter[1] * 2 + self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])
        self.conv2_2 = VGGBlock(self.nb_filter[2] * 2 + self.nb_filter[3], self.nb_filter[2], self.nb_filter[2])

        self.conv0_3 = VGGBlock(self.nb_filter[0] * 3 + self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])
        self.conv1_3 = VGGBlock(self.nb_filter[1] * 3 + self.nb_filter[2], self.nb_filter[1], self.nb_filter[1])

        self.conv0_4 = VGGBlock(self.nb_filter[0] * 4 + self.nb_filter[1], self.nb_filter[0], self.nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv3d(self.nb_filter[0], self.out_channels, kernel_size=1)
            self.final2 = nn.Conv3d(self.nb_filter[0], self.out_channels, kernel_size=1)
            self.final3 = nn.Conv3d(self.nb_filter[0], self.out_channels, kernel_size=1)
            self.final4 = nn.Conv3d(self.nb_filter[0], self.out_channels, kernel_size=1)
        else:
            self.final = nn.Conv3d(self.nb_filter[0], self.out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            out_logit1 = self.final1(x0_1)
            out_logit2 = self.final2(x0_2)
            out_logit3 = self.final3(x0_3)
            out_logit4 = self.final4(x0_4)
            if self.out_channels == 1:
                out1 = torch.sigmoid(out_logit1)
            if self.out_channels > 1:
                out1 = torch.softmax(out_logit1, dim=1)
            if self.out_channels == 1:
                out2 = torch.sigmoid(out_logit2)
            if self.out_channels > 1:
                out2 = torch.softmax(out_logit2, dim=1)
            if self.out_channels == 1:
                out3 = torch.sigmoid(out_logit3)
            if self.out_channels > 1:
                out3 = torch.softmax(out_logit3, dim=1)
            if self.out_channels == 1:
                out4 = torch.sigmoid(out_logit4)
            if self.out_channels > 1:
                out4 = torch.softmax(out_logit4, dim=1)
            return [out1, out2, out3, out4], [out_logit1, out_logit2, out_logit3, out_logit4]
        else:
            out_logit = self.final(x0_4)
            if self.out_channels == 1:
                output = torch.sigmoid(out_logit)
            if self.out_channels > 1:
                output = torch.softmax(out_logit, dim=1)
            return out_logit, output


if __name__ == "__main__":
    net = NestedUNet3d(1, 1, 16)
    in1 = torch.rand((1, 1, 64, 256, 256))
    out = net(in1)
    for i in range(len(out)):
        print(out[i].size())
