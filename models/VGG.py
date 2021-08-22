from torch import nn


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
            nn.ReLU(),
            nn.Linear(dim_fc, dim_fc),
            nn.ReLU(),
            nn.Linear(dim_fc, args.num_classes)
        ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv(x)
        y = self.fc(h.view(x.shape[0], -1))
        return y


class VGGConvBlock(nn.Module):
    def __init__(self, num_conv, in_dim, out_dim, kernel_size=3, stride=1):
        super().__init__()
        layers = []
        for i in range(num_conv):
            layers.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding=1),
                nn.ReLU()
            ])
            in_dim = out_dim
        layers.append(nn.MaxPool2d(2, 2))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
