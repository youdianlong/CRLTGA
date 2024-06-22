import torch
from torch import nn
from torch.nn.init import orthogonal_
from torch.nn.utils import spectral_norm

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        orthogonal_(m.weight)
        m.bias.data.fill_(0.)
def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

def snconvtrans2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


class DiscOptBlock(nn.Module):
    # Compared with block, optimized_block always downsamples the spatial resolution of the input vector by a factor of 4.
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.downsample = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.downsample(x)

        x0 = self.downsample(x0)
        x0 = self.conv_0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()
        self.downsample = nn.AvgPool2d(2)
        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

    def forward(self, x, downsample=True):
        x0 = x

        x = self.relu(x)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        if downsample:
            x = self.downsample(x)

        if downsample or self.ch_mismatch:
            x0 = self.conv_0(x0)
            if downsample:
                x0 = self.downsample(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, conv_dim, image_size=128, in_channels=3, out_channels=1, out_feature=False):
        super().__init__()
        self.conv_dim = conv_dim
        self.image_size = image_size
        self.out_feature = out_feature

        self.fromRGB = snconv2d(in_channels, conv_dim, 1, bias=True)

        self.block1 = DiscBlock(conv_dim, conv_dim * 2)
        self.self_attn = Self_Attn(conv_dim*2)
        self.block2 = DiscBlock(conv_dim * 2, conv_dim * 4)
        self.block3 = DiscBlock(conv_dim * 4, conv_dim * 8)
        if image_size == 64:
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 16)
            self.block5 = DiscBlock(conv_dim * 16, conv_dim * 16)
        elif image_size == 128:
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 16)
            self.block5 = DiscBlock(conv_dim * 16, conv_dim * 16)
            self.block6 = DiscBlock(conv_dim * 16, conv_dim * 16)
        else:
            self.block4 = DiscBlock(conv_dim * 8, conv_dim * 8)
            self.block5 = DiscBlock(conv_dim * 8, conv_dim * 16)
            self.block6 = DiscBlock(conv_dim * 16, conv_dim * 16)
        self.relu = nn.ReLU(inplace=True)
        self.snlinear1 = snlinear(in_features=conv_dim*16, out_features=out_channels)

        # Weight init
        self.apply(init_weights)

    def forward(self, x):
        h0 = self.fromRGB(x)
        h1 = self.block1(h0)
        h1 = self.self_attn(h1)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        if self.image_size == 64:
            h5 = self.block5(h4, downsample=False)
            h6 = h5
        elif self.image_size == 128:
            h5 = self.block5(h4)
            h6 = self.block6(h5, downsample=False)
        else:
            h5 = self.block5(h4)
            h6 = self.block6(h5)
            h6 = self.block7(h6, downsample=False)
        h6 = self.relu(h6)

        # Global sum pooling
        h7 = torch.sum(h6, dim=[2,3])
        out = torch.squeeze(self.snlinear1(h7))

        if self.out_feature:
            return out, h7
        else:
            return out


class DisFinalBlock(nn.Module):
    """ Final block for the Discriminator """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_1 = snconv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_2 = snconv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4)
        self.conv_3 = snconv2d(in_channels=in_channels, out_channels=1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.conv_1(x))
        y = self.relu(self.conv_2(y))
        y = self.conv_3(y)

        return y