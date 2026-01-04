
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CSPHet']

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

# ---------- BN + IN ----------
class BN_IN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.inorm = nn.InstanceNorm2d(channels, affine=True)
    def forward(self, x):
        return self.inorm(self.bn(x))

# ---------- DyReLU ----------
class DyReLU(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels * 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        pooled = self.pool(x)
        x_fc = self.relu(self.fc1(pooled))
        params = self.sigmoid(self.fc2(x_fc))
        a, b = params.chunk(2, dim=1)
        return torch.max(a * x + b, torch.zeros_like(x))

# ---------- CBAM Attention ----------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.channel_att(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * sa

# ---------- Depthwise Separable Conv (DWConv) ----------
def DWConv(in_channels, out_channels, k=3, s=1, p=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, k, s, p, groups=in_channels, dilation=d, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.InstanceNorm2d(in_channels, affine=True),
        DyReLU(in_channels),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.InstanceNorm2d(out_channels, affine=True),
        DyReLU(out_channels),
    )

# ---------- HetConv ----------
class HetConv(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, p=4):
        super(HetConv, self).__init__()
        self.p = p
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = nn.ModuleList()
        self.convolution_1x1_index = []
        for i in range(self.p):
            self.convolution_1x1_index.append(self.compute_convolution_1x1_index(i))
        for i in range(self.p):
            self.filters.append(self.build_HetConv_filters(stride, p))

    def compute_convolution_1x1_index(self, i):
        index = [j for j in range(0, self.input_channels)]
        while i < self.input_channels:
            index.remove(i)
            i += self.p
        return index

    def build_HetConv_filters(self, stride, p):
        temp_filters = nn.ModuleList()
        # 用DWConv替代普通Conv，提升轻量和激活表现
        temp_filters.append(DWConv(self.input_channels // p, self.output_channels // p, 3, stride, 1))
        temp_filters.append(DWConv(self.input_channels - self.input_channels // p, self.output_channels // p, 1, stride, 0))
        return temp_filters

    def forward(self, input_data):
        output_feature_maps = []
        for i in range(self.p):
            output_feature_3x3 = self.filters[i][0](input_data[:, i::self.p, :, :])
            output_feature_1x1 = self.filters[i][1](input_data[:, self.convolution_1x1_index[i], :, :])
            output_feature_map = output_feature_1x1 + output_feature_3x3
            output_feature_maps.append(output_feature_map)

        N, C, H, W = output_feature_maps[0].size()
        C = self.p * C
        return torch.cat(output_feature_maps, 1).view(N, self.p, C // self.p, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

# ---------- CSPHet Bottleneck ----------
class CSPHet_Bottleneck(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            HetConv(dim, dim),
            CBAM(dim)
        )
        self.act = DyReLU(dim)

    def forward(self, x):
        out = self.conv(x)
        return self.act(out + x)

# ---------- CSPHet Module ----------
class CSPHet(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=DyReLU(2*self.c))
        self.cv2 = Conv((2 + n) * self.c, c2, 1, act=DyReLU(c2))
        self.blocks = nn.ModuleList([CSPHet_Bottleneck(self.c) for _ in range(n)])

    def forward(self, x):
        y1, y2 = self.cv1(x).chunk(2, dim=1)
        out = [y1, y2]
        for m in self.blocks:
            y2 = m(y2)
            out.append(y2)
        return self.cv2(torch.cat(out, dim=1))

# ---------- Test ----------
if __name__ == "__main__":
    model = CSPHet(64, 128, n=2)
    x = torch.randn(1, 64, 224, 224)
    y = model(x)
    print(y.shape)