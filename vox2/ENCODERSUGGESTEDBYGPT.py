import torch
import torch.nn as nn
import torch.nn.functional as F


class SincNet(nn.Module):
    def __init__(self):
        super(SincNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=80, kernel_size=251, stride=1),
            nn.BatchNorm1d(80),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=80, out_channels=60, kernel_size=5, stride=1),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=60, out_channels=60, kernel_size=5, stride=1),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=60 * 46, out_features=2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
        )

    def sinc_conv(self, x, K, n_filters, stride=1, padding=0, dilation=1):
        # Define a custom sinc-based convolutional function
        n_samples = x.shape[2]
        time_steps = torch.linspace(start=-1., end=1., steps=n_samples)
        time_steps = time_steps.to(x.device)
        time_steps = time_steps.unsqueeze(0).unsqueeze(0).repeat(n_filters, 1, 1)
        sinc_filter = torch.sin(math.pi * (time_steps - (K - 1) / 2) / K) / (math.pi * (time_steps - (K - 1) / 2) / K)
        sinc_filter = torch.where(torch.abs(time_steps) < 1e-9, torch.ones_like(time_steps), sinc_filter)
        sinc_filter = sinc_filter.unsqueeze(1).repeat(1, x.shape[1], 1)
        sinc_conv = F.conv1d(x, weight=sinc_filter, stride=stride, padding=padding, dilation=dilation)
        return sinc_conv

    def forward(self, x):
        x = self.sinc_conv(x, K=251, n_filters=80, stride=1)
        x = self.cnn(x)
        x = x.view(x.shape[0], -1)
        x
