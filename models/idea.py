import torch
import torch.nn as nn

class DSConv(nn.Module):  # Depthwise Separable Convolution
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super(DSConv, self).__init__()
        # Depth convolution (Channel level)
        self.depth_conv = nn.Conv2d(c1, c1, kernel_size=k, stride=1, padding=k//2, groups=c1, bias=False)
        self.depth_bn = nn.BatchNorm2d(c1)

        # Point convolution (Pixel level)
        self.point_conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.point_bn = nn.BatchNorm2d(c2)

        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        depth_conv = self.act(self.depth_bn(self.depth_conv(x)))
        point_conv = self.act(self.point_bn(self.point_conv(depth_conv)))
        return point_conv


class SliceSamp(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SliceSamp, self).__init__()
        self.conv = DSConv(c1 * 4, c2, k, s)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class SliceUpsamp(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SliceUpsamp, self).__init__()
        self.conv = DSConv(c1 // 4, c2, 3, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,c/4,w*2,h*2)
        b, c, h, w = x.shape
        c = c//4
        z = torch.zeros(b, c, h*2, w*2, device=x.device, dtype=x.dtype)
        z[..., ::2, ::2] = x[:, :c, :, :]
        z[..., 1::2, ::2] = x[:, c:2*c, :, :]
        z[..., ::2, 1::2] = x[:, 2*c:3*c, :, :]
        z[..., 1::2, 1::2] = x[:, 3*c:, :, :]
        return self.conv(z)
