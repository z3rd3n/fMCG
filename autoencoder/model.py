import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(8)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(4)
        self.conv4 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(2)
        self.conv5 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, samples) for Conv1d
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        return x.permute(0, 2, 1)  # Change shape back to (batch, samples, channels)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv0 = nn.ConvTranspose1d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose1d(in_channels=2, out_channels=4, kernel_size=5, padding=2)
        self.deconv2 = nn.ConvTranspose1d(in_channels=4, out_channels=8, kernel_size=7, padding=3)
        self.deconv3 = nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=9, padding=4)
        self.deconv4 = nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=11, padding=5)

    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, samples) for Conv1d
        
        x = F.relu(self.deconv0(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))

        return x.permute(0, 2, 1)  # Change shape back to (batch, samples, channels)


