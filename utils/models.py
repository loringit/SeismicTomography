import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AccelerationPredictor2(nn.Module):
    def __init__(self):
        super(AccelerationPredictor2, self).__init__()

        self.non_linearity = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(50, 15), dilation=(2, 1), stride=(2, 1)),
            self.non_linearity,
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(15, 5), dilation=(4, 2), stride=(2, 1)),
            self.non_linearity,
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(15, 5), dilation=(4, 2), stride=(2, 1)),
            self.non_linearity
        )

        self.transposed_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(5, 15), dilation=(2, 4)),
            self.non_linearity,
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=(5, 15), dilation=(2, 4)),
            self.non_linearity,
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(20, 30), dilation=(2, 1), stride=(1, 2))
        )

        self.linear = nn.Linear(592, 500)

    def forward(self, x):
        out = self.conv(x.unsqueeze(1))
        out = self.non_linearity(self.transposed_conv(out))
        out = self.non_linearity(self.linear(out.squeeze(1)))
        return out


class AccelerationPredictor(nn.Module):
    def __init__(self):
        super(AccelerationPredictor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(2000, 1500, 10),
            nn.ReLU(),
            nn.Conv1d(1500, 1000, 10),
            nn.ReLU(),
            nn.Conv1d(1000, 500, 10),
            nn.ReLU(),
            nn.Conv1d(500, 250, 10),
            nn.BatchNorm1d(250),
            nn.ReLU(),
        )

        self.linear = nn.Linear(164, 500)

    def forward(self, x):
        y = self.conv(x)
        y = self.linear(y)
        return y


if __name__ == '__main__':
    print('Model AccelerationPredictor2: {}'.format(count_parameters(AccelerationPredictor2())))
    print('Model AccelerationPredictor: {}'.format(count_parameters(AccelerationPredictor())))