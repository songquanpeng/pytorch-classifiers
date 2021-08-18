from torch import nn


class LeNet5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.img_size == 32
        layers = [  # 1x32x32
            nn.Conv2d(args.img_dim, 6, 5, 1),  # 6x28x28
            nn.Tanh(),
            nn.MaxPool2d(2, 2),  # 6x14x14
            nn.Conv2d(6, 16, 5, 1),  # 16x10x10
            nn.Tanh(),
            nn.MaxPool2d(2, 2),  # 16x5x5
        ]
        self.conv = nn.Sequential(*layers)

        layers = [
            nn.Linear(16 * 5 * 5, 84),
            nn.Tanh(),
            nn.Linear(84, args.num_classes),
            # nn.Softmax()  # output logits (-inf, inf)
        ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv(x)
        y = self.fc(h.view(x.shape[0], -1))
        return y
