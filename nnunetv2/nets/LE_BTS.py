import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardInvertedResidual3D(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=4, stride=1):
        super().__init__()
        expanded_channels = int(in_channels * expansion_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.conv_expand = nn.Sequential(
            nn.Conv3d(in_channels, expanded_channels, 1, bias=False),
            nn.InstanceNorm3d(expanded_channels, affine=True),
            nn.GELU()
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(
                expanded_channels,
                expanded_channels,
                3,
                stride=stride,
                padding=1,
                groups=expanded_channels,
                bias=False
            ),
            nn.InstanceNorm3d(expanded_channels, affine=True),
            nn.GELU()
        )
        self.conv_compress = nn.Sequential(
            nn.Conv3d(expanded_channels, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True)
        )

    def forward(self, x):
        residual = x
        x = self.conv_expand(x)
        x = self.depthwise_conv(x)
        x = self.conv_compress(x)
        if self.use_res_connect:
            x = x + residual
        return x


class DSC3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            3,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class PoolingAttention(nn.Module):
    def forward(self, x):
        pooled = [
            torch.mean(x, dim=i, keepdim=True).expand_as(x)
            for i in range(2, x.ndim)
        ]
        attention = torch.stack(pooled, dim=-1).mean(dim=-1)
        return torch.sigmoid(attention) * x


class ChannelStatistics(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.token_mixer = PoolingAttention()
        self.mlp = Mlp(dim, dim * 4, 0.1)

    def forward(self, x):
        B, num_d, num_h, num_w, num_patches, C = x.shape
        x = x.reshape(B * num_d * num_h * num_w, num_patches, C)
        x = x + self.token_mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x.reshape(B, num_d, num_h, num_w, num_patches, C)


class MSCG(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        depth = 4
        self.fc_module = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.fc_rever_module = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
        self.CS = ChannelStatistics(dim)
        self.split_list = [512, 64, 8, 1]
        self.dw_block = nn.ModuleList([DSC3D(dim, dim) for _ in range(depth)])
        self.fusion_layers = nn.ModuleList([
            nn.Conv3d(dim * 2, dim, 1) for _ in range(depth)
        ])

    def forward(self, x_list):
        if len(x_list) != 4:
            raise ValueError(f"MSCG expects 4 feature maps, got {len(x_list)}")

        x_proj = [
            fc(x.permute(0, 2, 3, 4, 1))
            for fc, x in zip(self.fc_module, x_list)
        ]

        for j in range(4):
            B, D, H, W, C = x_proj[j].shape
            win_size = 8 // (2 ** j)
            if D % win_size or H % win_size or W % win_size:
                raise ValueError(
                    f"Feature map at scale {j} with shape {(D, H, W)} "
                    f"cannot be divided by window size {win_size}"
                )
            x_proj[j] = x_proj[j].reshape(
                B,
                D // win_size,
                win_size,
                H // win_size,
                win_size,
                W // win_size,
                win_size,
                C
            ).permute(
                0, 1, 3, 5, 2, 4, 6, 7
            ).reshape(
                B,
                D // win_size,
                H // win_size,
                W // win_size,
                win_size ** 3,
                C
            )

        x_cat = torch.cat(x_proj, dim=-2)
        x_attn = self.CS(x_cat) + x_cat
        x_split = torch.split(x_attn, self.split_list, dim=-2)

        outputs = []
        for j in range(4):
            B, bD, bH, bW, _, C = x_split[j].shape
            win_size = 8 // (2 ** j)
            x_restore = x_split[j].reshape(
                B, bD, bH, bW, win_size, win_size, win_size, C
            ).permute(
                0, 1, 4, 2, 5, 3, 6, 7
            ).reshape(
                B, bD * win_size, bH * win_size, bW * win_size, C
            )
            x_restore = self.fc_rever_module[j](x_restore).permute(0, 4, 1, 2, 3)
            outputs.append(self.dw_block[j](x_restore))

        return [
            fusion(torch.cat((output, skip), dim=1))
            for fusion, output, skip in zip(self.fusion_layers, outputs, x_list)
        ]


class BOBDFFM(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        mip = inp // 4
        self.conv1 = nn.Conv3d(inp, mip, 1)
        self.in1 = nn.InstanceNorm3d(mip)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(inp, mip, 1)
        self.in2 = nn.InstanceNorm3d(mip)
        self.relu2 = nn.ReLU()
        self.conv_d = nn.Conv3d(mip, oup, 1)
        self.conv_h = nn.Conv3d(mip, oup, 1)
        self.conv_w = nn.Conv3d(mip, oup, 1)
        self.fc1 = nn.Conv3d(inp, inp // 2, 1, bias=False)
        self.boundary_weight_conv = nn.Conv3d(inp // 2, 1, 3, padding=1, bias=False)

    def forward(self, g, x):
        _, _, d, h, w = x.shape
        boundary_weight = torch.sigmoid(self.boundary_weight_conv(self.fc1(x)))

        g_d = F.adaptive_avg_pool3d(g, (d, 1, 1))
        g_h = F.adaptive_avg_pool3d(g, (1, h, 1)).permute(0, 1, 3, 2, 4)
        g_w = F.adaptive_avg_pool3d(g, (1, 1, w)).permute(0, 1, 4, 2, 3)

        x_d = F.adaptive_avg_pool3d(x, (d, 1, 1))
        x_h = F.adaptive_avg_pool3d(x, (1, h, 1)).permute(0, 1, 3, 2, 4)
        x_w = F.adaptive_avg_pool3d(x, (1, 1, w)).permute(0, 1, 4, 2, 3)

        g_y = torch.cat([g_d, g_h, g_w], dim=2)
        g_y = self.relu1(self.in1(self.conv1(g_y)))
        x_y = torch.cat([x_d, x_h, x_w], dim=2)
        x_y = self.relu2(self.in2(self.conv2(x_y)))

        g_d, g_h, g_w = torch.split(g_y, [d, h, w], dim=2)
        g_h = g_h.permute(0, 1, 3, 2, 4)
        g_w = g_w.permute(0, 1, 3, 4, 2)

        x_d, x_h, x_w = torch.split(x_y, [d, h, w], dim=2)
        x_h = x_h.permute(0, 1, 3, 2, 4)
        x_w = x_w.permute(0, 1, 3, 4, 2)

        a_d = torch.sigmoid(self.conv_d((x_d + g_d) / 2))
        a_h = torch.sigmoid(self.conv_h((x_h + g_h) / 2))
        a_w = torch.sigmoid(self.conv_w((x_w + g_w) / 2))

        return x * (a_d * a_h * a_w + boundary_weight)


class DynamicFrequencySelection(nn.Module):
    def __init__(self, num_subbands, in_channels):
        super().__init__()
        hidden = max(1, num_subbands // 4)
        self.num_subbands = num_subbands
        self.attention = nn.Sequential(
            nn.Linear(num_subbands * in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_subbands),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, channels, D, H, W = x.shape
        C = channels // self.num_subbands
        x = x.reshape(B, C, self.num_subbands, D, H, W)
        weights = self.attention(x.mean(dim=(3, 4, 5)).reshape(B, -1))
        x = x * weights.reshape(B, 1, self.num_subbands, 1, 1, 1)
        return x.reshape(B, -1, D, H, W)


class DyHWT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        num_subbands = 8
        inv_sqrt2 = 1.0 / (2.0 ** 0.5)
        low = torch.tensor([1.0, 1.0]) * inv_sqrt2
        high = torch.tensor([-1.0, 1.0]) * inv_sqrt2
        filters = {"a": low, "d": high}
        order = ["aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd"]

        kernels = [
            torch.einsum(
                "i,j,k->ijk",
                filters[band[0]],
                filters[band[1]],
                filters[band[2]]
            )
            for band in order
        ]
        kernels = torch.stack(kernels).unsqueeze(1).repeat(in_channels, 1, 1, 1, 1)
        self.register_buffer("haar_kernels", kernels)

        self.freq_selector = DynamicFrequencySelection(num_subbands, in_channels)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels * num_subbands, out_channels, 1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, _, _, _ = x.shape
        x = F.conv3d(
            x,
            self.haar_kernels,
            stride=2,
            padding=0,
            groups=C
        )
        x = self.freq_selector(x)
        return self.conv(x)


class Conv1x1x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, 1, bias=False)
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv3x3x1(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.conv = nn.Conv3d(
            in_dim,
            out_dim,
            (3, 3, 1),
            padding=(1, 1, 0),
            bias=False
        )
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv1x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.conv = nn.Conv3d(
            in_dim,
            out_dim,
            (1, 3, 3),
            padding=(0, 1, 1),
            bias=False
        )
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv3x3x3(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.conv = nn.Conv3d(
            in_dim,
            out_dim,
            3,
            padding=1,
            bias=False
        )
        self.norm = nn.InstanceNorm3d(out_dim)
        self.act = activation

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MSSC(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.out_inter_dim = out_dim // 4
        self.conv_3x3x1_1 = Conv3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_3x3x1_2 = Conv3x3x1(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x3x3_1 = Conv1x3x3(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x3x3_2 = Conv1x3x3(self.out_inter_dim, self.out_inter_dim, activation)
        self.conv_1x1x1_1 = Conv1x1x1(in_dim, out_dim, activation)
        self.conv_1x1x1_2 = Conv1x1x1(out_dim, out_dim, activation)
        self.bobdffm = BOBDFFM(out_dim, out_dim)
        self.conv_1x1x1_3 = (
            Conv1x1x1(in_dim, out_dim, activation)
            if in_dim > out_dim
            else None
        )

    def forward(self, x):
        x_1 = self.conv_1x1x1_1(x)
        x1, x2, x3, x4 = torch.chunk(x_1, 4, dim=1)
        x1 = self.conv_3x3x1_1(x1)
        x2 = self.conv_3x3x1_2(x2 + x1)
        x3 = self.conv_1x3x3_1(x3)
        x4 = self.conv_1x3x3_2(x4 + x3)
        x_1 = self.conv_1x1x1_2(torch.cat((x1, x2, x3, x4), dim=1))

        if self.conv_1x1x1_3 is not None:
            x = self.conv_1x1x1_3(x)

        return self.bobdffm(x_1, x)


class AMDA(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.sp = Conv3x3x3(2, 1, activation)
        self.aH = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.aW = nn.AdaptiveAvgPool3d((1, None, 1))
        self.aD = nn.AdaptiveAvgPool3d((1, 1, None))
        self.mH = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.mW = nn.AdaptiveMaxPool3d((1, None, 1))
        self.mD = nn.AdaptiveMaxPool3d((1, 1, None))
        self.conv1 = Conv1x1x1(in_dim, in_dim // 8, activation)
        self.conv2 = nn.Conv3d(in_dim // 8, in_dim, 1, bias=False)
        self.convout = nn.Conv3d(in_dim, out_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.InstanceNorm3d(in_dim)

    def forward(self, x):
        xah = self.norm(self.conv2(self.conv1(self.aH(x))))
        xaw = self.norm(self.conv2(self.conv1(self.aW(x))))
        xad = self.norm(self.conv2(self.conv1(self.aD(x))))
        qkv_avg = self.softmax(xah * xaw) * xad

        xmh = self.norm(self.conv2(self.conv1(self.mH(x))))
        xmw = self.norm(self.conv2(self.conv1(self.mW(x))))
        xmd = self.norm(self.conv2(self.conv1(self.mD(x))))
        qkv_max = self.softmax(xmh * xmw) * xmd

        xc = self.convout(x * (qkv_avg + qkv_max))
        spatial = torch.cat([
            torch.mean(xc, dim=1, keepdim=True),
            torch.max(xc, dim=1, keepdim=True).values
        ], dim=1)
        return torch.sigmoid(self.sp(spatial)) * xc


def hdc(image):
    return torch.cat([
        image[:, :, k::2, i::2, j::2]
        for k in range(2)
        for i in range(2)
        for j in range(2)
    ], dim=1)


class LE_BTS(nn.Module):
    def __init__(self, out_dim=3, num_filters=32):
        super().__init__()
        n = num_filters
        activation = nn.ReLU(inplace=False)

        self.invert_res = nn.ModuleList([
            StandardInvertedResidual3D(n, n, expansion_ratio=4)
            for _ in range(7)
        ])

        self.dwt = nn.ModuleList([
            DyHWT(n, n) for _ in range(3)
        ])

        self.dw = DSC3D(n, n)

        self.encoder_blocks = nn.ModuleList([
            MSSC(n, n, activation),
            MSSC(n, n, activation),
            MSSC(n, n, activation),
            MSSC(n, n, activation)
        ])

        self.mscg = MSCG(dim=n)

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        self.decoder_blocks = nn.ModuleList([
            MSSC(n * 2, n, activation),
            MSSC(n * 2, n, activation),
            MSSC(n * 2, n, activation)
        ])

        self.amda = AMDA(n, out_dim, activation)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.dw(hdc(x))

        features = []
        for i in range(4):
            x = self.encoder_blocks[i](x)
            x = self.invert_res[i](x)
            features.append(x)
            if i < 3:
                x = self.dwt[i](x)

        fusions = self.mscg(features)

        x = self.up(fusions[3])
        x = self.decoder_blocks[0](torch.cat((x, fusions[2]), dim=1))
        x = self.invert_res[4](x)

        x = self.up(x)
        x = self.decoder_blocks[1](torch.cat((x, fusions[1]), dim=1))
        x = self.invert_res[5](x)

        x = self.up(x)
        x = self.decoder_blocks[2](torch.cat((x, fusions[0]), dim=1))
        x = self.invert_res[6](x)

        x = self.up(x)
        return self.amda(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 4, 128, 128, 128, device=device)
    model = LE_BTS(out_dim=3, num_filters=32).to(device)
    y = model(x)
    print(y.shape)
