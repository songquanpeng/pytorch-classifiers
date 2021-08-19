from torch import nn


class AlexNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.img_size == 227
        layers = [
            nn.Conv2d(args.img_dim, 96, 11, 4),  # 96x55x55
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # 96x27x27
            nn.Conv2d(96, 256, 5, 1, 2),  # 256x27x27
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # 256x13x13
            nn.Conv2d(256, 384, 3, 1, 1),  # 384x13x13
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),  # 384x13x13
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),  # 256x13x13
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # 256x6x6
        ]
        self.conv = nn.Sequential(*layers)
        layers = [
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, args.num_classes)
        ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv(x)
        y = self.fc(h.view(x.shape[0], -1))
        return y
