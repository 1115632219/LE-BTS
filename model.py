import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch.nn.functional as F
import pywt
from nets.Inverted_Resblock import StandardInvertedResidual3D
from nets.LightTransformer import LightTransformerBlock

class DSC3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, norm=True, activation=True):
        super(DSC3D, self).__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            bias=bias
        )
        self.norm = nn.InstanceNorm3d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BOBDFFM(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(BOBDFFM, self).__init__()

        mip = inp // reduction
        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.in1 = nn.InstanceNorm3d(mip)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.in2 = nn.InstanceNorm3d(mip)
        self.relu2 = nn.ReLU()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 将输入通道数减半，并通过 1x1x1 3D 卷积调整通道数
        self.fc1 = nn.Conv3d(inp, inp // 2, kernel_size=1, bias=False)

        # 生成边界权重图的 3D 可学习卷积层，输入通道修改为 input_dim // 2
        self.boundary_weight_conv = nn.Conv3d(inp // 2, 1, kernel_size=3, padding=1, bias=False)

        # Sigmoid 激活函数
        self.Sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        b, c, d, h, w = x.size()
        L_feature = self.fc1(x)

        # 生成可学习的边界权重图
        boundary_weight = self.boundary_weight_conv(L_feature)  # Shape: (B, 1, D, H, W)
        boundary_weight = self.Sigmoid(boundary_weight)  # 生成边界权重的 sigmoid 输出
        # print(boundary_weight.shape)


        g_d = F.adaptive_avg_pool3d(g, (d, 1, 1))
        g_h = F.adaptive_avg_pool3d(g, (1, h, 1)).permute(0, 1, 3, 2, 4)
        g_w = F.adaptive_avg_pool3d(g, (1, 1, w)).permute(0, 1, 4, 2, 3)

        x_d = F.adaptive_avg_pool3d(x, (d, 1, 1))
        x_h = F.adaptive_avg_pool3d(x, (1, h, 1)).permute(0, 1, 3, 2, 4)
        x_w = F.adaptive_avg_pool3d(x, (1, 1, w)).permute(0, 1, 4, 2, 3)

        g_y = torch.cat([g_d, g_h, g_w], dim=2)
        # print(g_y.shape)
        g_y = self.conv1(g_y)
        g_y = self.in1(g_y)
        g_y = self.relu1(g_y)
        # print(g_y.shape)

        x_y = torch.cat([x_d, x_h, x_w], dim=2)
        x_y = self.conv2(x_y)
        x_y = self.in2(x_y)
        x_y = self.relu2(x_y)

        g_d, g_h, g_w = torch.split(g_y, [d, h, w], dim=2)
        g_h = g_h.permute(0, 1, 3, 2, 4)
        g_w = g_w.permute(0, 1, 3, 4, 2)

        x_d, x_h, x_w = torch.split(x_y, [d, h, w], dim=2)
        x_h = x_h.permute(0, 1, 3, 2, 4)
        x_w = x_w.permute(0, 1, 3, 4, 2)

        a_d = (x_d + g_d) / 2
        a_h = (x_h + g_h) / 2
        a_w = (x_w + g_w) / 2

        a_d, a_h, a_w = torch.sigmoid(self.conv_d(a_d)), torch.sigmoid(self.conv_h(a_h)), torch.sigmoid(
            self.conv_w(a_w))

        x = x * (a_d * a_h * a_w +boundary_weight)
        # print(x.shape)
        return x




class Conv_1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x1x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0), bias=False)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_1x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False)
        # self.norm = nn.BatchNorm3d(out_dim)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x


class Conv_3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(Conv_3x3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        return x

class MSSC(nn.Module):
    def __init__(self, in_dim, out_dim, activation,proj_size=64):
        super(MSSC, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inter_dim = in_dim // 4
        self.out_inter_dim = out_dim // 4
        self.proj_size = proj_size
        self.conv_3x3x1_1 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_2 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_3 = Conv_3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x3x3_1 = Conv_1x3x3(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x3x3_2 = Conv_1x3x3(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x1x1_1 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv_1x1x1(out_dim, out_dim, activation)
        self.bobdffm = BOBDFFM(out_dim,out_dim,reduction=4)
        # self.epa = None
        if self.in_dim > self.out_dim:
            self.conv_1x1x1_3 = Conv_1x1x1(in_dim, out_dim, activation)
        self.conv_1x3x3 = Conv_1x3x3(out_dim, out_dim, activation)

    def forward(self, x):
        b, c, d, h, w = x.size()
        x_1 = self.conv_1x1x1_1(x)
        x1 = x_1[:, 0:self.out_inter_dim, ...]
        x2 = x_1[:, self.out_inter_dim:self.out_inter_dim * 2, ...]
        x3 = x_1[:, self.out_inter_dim * 2:self.out_inter_dim * 3, ...]
        x4 = x_1[:, self.out_inter_dim * 3:self.out_inter_dim * 4, ...]
        x1 = self.conv_3x3x1_1(x1)
        x2 = self.conv_3x3x1_2(x2+x1)
        x3 = self.conv_1x3x3_1(x3)
        x4 = self.conv_1x3x3_2(x4+x3)
        x_1 = torch.cat((x1, x2, x3, x4), dim=1)
        x_1 = self.conv_1x1x1_2(x_1)
        if self.in_dim > self.out_dim:
            x = self.conv_1x1x1_3(x)
        x_1 = self.bobdffm(x_1,x)
        return x_1

device1 = torch.device("cuda")


class AMDA(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(AMDA, self).__init__()
        self.sp = Conv_3x3x3(2, 1, activation)
        self.g = nn.AdaptiveAvgPool3d(1)
        self.m = nn.AdaptiveMaxPool3d(1)
        self.aH = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.aW = nn.AdaptiveAvgPool3d((1, None, 1))
        self.aD = nn.AdaptiveAvgPool3d((1, 1, None))
        self.mH = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.mW = nn.AdaptiveMaxPool3d((1, None, 1))
        self.mD = nn.AdaptiveMaxPool3d((1, 1, None))
        self.conv1 = Conv_1x1x1(in_dim, in_dim // 8, activation)
        self.conv2 = nn.Conv3d(in_dim // 8, in_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.convout = nn.Conv3d(in_dim, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.InstanceNorm3d(in_dim)
        self.act = activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xah = self.norm(self.conv2(self.conv1(self.aH(x))))
        xaw = self.norm(self.conv2(self.conv1(self.aW(x))))
        xad = self.norm(self.conv2(self.conv1(self.aD(x))))
        xahw = xah * xaw
        qk_avg = self.softmax(xahw)
        qkv_avg = qk_avg * xad
        xmh = self.norm(self.conv2(self.conv1(self.mH(x))))
        xmw = self.norm(self.conv2(self.conv1(self.mW(x))))
        xmd = self.norm(self.conv2(self.conv1(self.mD(x))))
        xmhw = xmh * xmw
        qk_max = self.softmax(xmhw)
        qkv_max = qk_max * xmd
        qkv = qkv_max+qkv_avg
        xc = self.convout(x * qkv)
        avg_out = torch.mean(xc, dim=1, keepdim=True)
        max_out, _ = torch.max(xc, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.sp(out))
        xo = out * xc
        return xo

def hdc(image, num=2):
    subs = [image[:, :, k::num, i::num, j::num]
            for k in range(num)
            for i in range(num)
            for j in range(num)]
    return torch.cat(subs, dim=1)

class LE_BTS(nn.Module):
    def __init__(self, in_dim=4, out_dim=3, num_filters=32):
        super(LE_BTS, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_f = num_filters
        self.activation = nn.ReLU(inplace=False)
        self.InvertRes1 = StandardInvertedResidual3D(self.n_f, self.n_f, expansion_ratio=4)
        self.InvertRes2 = StandardInvertedResidual3D(self.n_f, self.n_f, expansion_ratio=4)
        self.InvertRes3 = StandardInvertedResidual3D(self.n_f, self.n_f, expansion_ratio=4)
        self.InvertRes4 = StandardInvertedResidual3D(self.n_f, self.n_f, expansion_ratio=4)
        self.InvertRes5 = StandardInvertedResidual3D(self.n_f, self.n_f, expansion_ratio=4)
        self.InvertRes6 = StandardInvertedResidual3D(self.n_f, self.n_f, expansion_ratio=4)
        self.InvertRes7 = StandardInvertedResidual3D(self.n_f, self.n_f, expansion_ratio=4)

        self.dwt1 = DyHWT(self.n_f, self.n_f, use_attention=True) #采用了动态选取，赋权重关系
        self.dwt2 = DyHWT(self.n_f, self.n_f, use_attention=True)  # 采用了动态选取，赋权重关系
        self.dwt3 = DyHWT(self.n_f, self.n_f, use_attention=True)  # 采用了动态选取，赋权重关系
        self.conv_3x3x3 = Conv_3x3x3(self.n_f, self.n_f, self.activation)

        self.dw = DSC3D(self.n_f,self.n_f)
        self.conv_1 = MSSC(self.n_f, self.n_f, self.activation,proj_size=64)
        self.conv_2 = MSSC(self.n_f, self.n_f, self.activation,proj_size=64)
        self.conv_3 = MSSC(self.n_f, self.n_f, self.activation, proj_size=64)
        self.conv_33 = MSSC(self.n_f, self.n_f, self.activation, proj_size=64)
        # bridge
        self.csaformer = CSAFormer(32)
        # define fusion module
        self.fusion_dim = [32, 32, 32, 32]
        self.fusion_layers = nn.ModuleList()
        for i in range(4):
            self.fusion_layers.append(
                nn.Conv3d(self.fusion_dim[i] * 2, self.fusion_dim[i], 1, 1)
            )
        # up
        self.up_1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv_4 = MSSC(2 * self.n_f, self.n_f, self.activation,proj_size=64)
        self.up_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv_5 = MSSC(2 * self.n_f, self.n_f, self.activation,proj_size=64)
        self.up_3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv_6 = MSSC(2 * self.n_f, self.n_f, self.activation,proj_size=64)
        self.up_4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.amda = AMDA(self.n_f, out_dim, self.activation)
        self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = []  # 用于保存各层特征 [x1, x2, x3, x4]
        x = hdc(x)
        x = self.dw(x)
        x1 = self.conv_1(x)
        x1 = self.InvertRes1(x1)
        xs.append(x1)  # 添加x1
        x = self.dwt1(x1)
        x2 = self.conv_2(x)
        x2 = self.InvertRes2(x2)
        xs.append(x2)
        x = self.dwt2(x2)
        x3 = self.conv_3(x)
        x3 = self.InvertRes3(x3)
        xs.append(x3)
        x = self.dwt3(x3)
        x4 = self.conv_33(x)
        x4 = self.InvertRes4(x4)
        xs.append(x4)
        feat = self.csaformer([x1,x2,x3,x4])
        # fusion module
        fusions = []
        for i in range(4):
            skip = self.fusion_layers[i](torch.cat((feat[i], xs[i]), dim=1))
            fusions.append(skip)

        x = self.up_1(fusions[3])
        x = torch.cat((x, fusions[2]), dim=1)
        x = self.conv_4(x)
        x = self.InvertRes5(x)
        x = self.up_2(x)
        x = torch.cat((x, fusions[1]), dim=1)
        x = self.conv_5(x)
        x = self.InvertRes6(x)
        x = self.up_3(x)
        x = torch.cat((x, fusions[0]), dim=1)
        x = self.conv_6(x)
        x = self.InvertRes7(x)
        x = self.up_4(x)
        x = self.amda(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda')
    image_size = 128
    x = torch.rand((1, 4, 128, 128, 128), device=device)
    model = LE_BTS(in_dim=4, out_dim=3, num_filters=32).to(device)
    y = model(x)
    # print(model)
    print(y.shape)


