from torch import nn

from models.layers import VGGConvBlock

# TorchVision's implementation:


class VGG16(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.img_size == 224
        init_dim = 64
        max_dim = 512
        layers = [
            VGGConvBlock(2, args.img_dim, init_dim),  # 64x112x112
            VGGConvBlock(2, init_dim, init_dim * 2),  # 128x56x56
            VGGConvBlock(3, init_dim * 2, init_dim * 4),  # 256x28x28
            VGGConvBlock(3, init_dim * 4, max_dim),  # 512x14x14
            VGGConvBlock(3, max_dim, max_dim),  # 512x7x7
        ]
        self.conv = nn.Sequential(*layers)
        dim_conv = 512 * 7 * 7
        dim_fc = 4096
        layers = [
            nn.Linear(dim_conv, dim_fc),
            nn.Linear(dim_fc, dim_fc),
            nn.Linear(dim_fc, args.num_classes)
        ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv(x)
        y = self.fc(h.view(x.shape[0], -1))
        return y
