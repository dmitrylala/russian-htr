from torch import nn


class CRNN(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        conv_layers = []

        def convRelu(i, batchNormalization=False):
            nIn = 1 if i == 0 else nm[i - 1]
            nOut = nm[i]
            conv_layers.append(nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

            if batchNormalization:
                conv_layers.append(nn.BatchNorm2d(nOut))

            conv_layers.append(nn.ReLU(inplace=True))

        convRelu(0)
        conv_layers.append(nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        conv_layers.append(nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        conv_layers.append(nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        conv_layers.append(nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.conv_layers = nn.Sequential(*conv_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.squeeze(2).permute(2, 0, 1)  # [w, b, c]
        return self.linear(x)
