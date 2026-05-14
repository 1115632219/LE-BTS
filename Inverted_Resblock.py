import torch
import torch.nn as nn


class StandardInvertedResidual3D(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=4, stride=1):
        super(StandardInvertedResidual3D, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        # Expansion phase (升维4倍)
        expanded_channels = int(in_channels * expansion_ratio)

        # 1. 扩展层：1x1x1卷积升维
        self.conv_expand = nn.Sequential(
            nn.Conv3d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(expanded_channels, affine=True),  # 医学图像适配
            nn.GELU()  # 更平滑的激活函数  
        )

        # 2. 标准3x3x3深度可分离卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(
                expanded_channels, expanded_channels,
                kernel_size=3,
                stride=stride,
                padding=1,  # 保持空间尺寸的padding
                groups=expanded_channels,  # 深度可分离卷积
                bias=False
            ),
            nn.InstanceNorm3d(expanded_channels, affine=True),
            nn.GELU()
        )

        # 3. 压缩层：1x1x1卷积降维
        self.conv_compress = nn.Sequential(
            nn.Conv3d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True)
            # 注意：压缩层后无激活函数
        )

    def forward(self, x):
        residual = x

        # 扩展阶段
        x = self.conv_expand(x)

        # 深度卷积
        x = self.depthwise_conv(x)

        # 压缩阶段
        x = self.conv_compress(x)

        # 条件残差连接
        if self.use_res_connect:
            x = x + residual    

        return x

if __name__ == '__main__':
    x = torch.randn(1, 32, 64, 64, 64)
    res = StandardInvertedResidual3D(32, 32, expansion_ratio=4)
    x = res(x)
    print(x.shape)
