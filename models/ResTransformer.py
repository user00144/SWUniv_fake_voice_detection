import torch
from torch import nn

class ResNetTransformerModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNetTransformerModel, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 16x16 -> 8x8
        )
        self.transformer_input_dim = 256 * 8  # 256 채널과 8 (h) 곱
        self.linear = nn.Linear(self.transformer_input_dim, 256)  # 2048 -> 256
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, lengths):
        x = x.permute(0, 1, 3, 2)  # (batch, channels, width, height)
        x = self.resnet(x)
        print(x.shape)
        x = x.permute(0, 2, 1, 3)  # (batch, h, 256, w)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, h, 256 * w)
        x = self.linear(x)  # (batch, h, 256)
        x = x.permute(1, 0, 2)  # (h, batch, 256)
        x = self.transformer(x)  # (seq_len, batch, embed_dim)
        x = x.mean(dim=0)  # Global average pooling (batch, 256)
        x = self.fc(x)  # (batch, num_classes)
        x = torch.sigmoid(x)
        return x












