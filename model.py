import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blk = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.blk(x)


class Generator(nn.Module):
    """ResNet-based CycleGAN generator (encoder -> residual blocks -> decoder)."""

    def __init__(self, in_ch=3, out_ch=3, nf=64, n_res=6):
        super().__init__()
        layers = []

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, nf, 7),
                   nn.InstanceNorm2d(nf), nn.ReLU(inplace=True)]

        for i in range(2):
            m = 2 ** i
            layers += [nn.Conv2d(nf * m, nf * m * 2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(nf * m * 2), nn.ReLU(inplace=True)]

        for _ in range(n_res):
            layers.append(ResBlock(nf * 4))

        for i in range(2):
            m = 2 ** (2 - i)
            layers += [nn.ConvTranspose2d(nf * m, nf * m // 2, 3, stride=2,
                                          padding=1, output_padding=1),
                       nn.InstanceNorm2d(nf * m // 2), nn.ReLU(inplace=True)]

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(nf, out_ch, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """PatchGAN discriminator for CycleGAN."""

    def __init__(self, in_ch=3, nf=64):
        super().__init__()

        def blk(ic, oc, stride=2, norm=True):
            layers = [nn.Conv2d(ic, oc, 4, stride=stride, padding=1,
                                padding_mode="reflect")]
            if norm:
                layers.append(nn.InstanceNorm2d(oc))
            return layers + [nn.LeakyReLU(0.2, inplace=True)]

        self.net = nn.Sequential(
            *blk(in_ch, nf, norm=False),
            *blk(nf, nf * 2),
            *blk(nf * 2, nf * 4),
            *blk(nf * 4, nf * 8, stride=1),
            nn.Conv2d(nf * 8, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.net(x)
