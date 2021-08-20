import torch.nn as nn


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
