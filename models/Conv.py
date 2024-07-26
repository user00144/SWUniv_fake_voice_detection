import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class AudioClassificationModel(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained("HyperMoon/wav2vec2-base-finetuned-deepfake-0919")
        #[batch, 302, 512]

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block([1,32], first=True)),
            nn.Sequential(Residual_block([32,32])),
            nn.Sequential(Residual_block([32,64])),
            nn.Sequential(Residual_block([64,64])),
            nn.Sequential(Residual_block([64,64])))
        # Define layers
        self.pool = nn.AdaptiveAvgPool3d(output_size = [64,1,1])
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        # Apply layer
        
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.pool(x)
        x = self.flat(x) # Flatten the tensor for the fully connected layer
        x = F.relu(x)
        x = torch.sigmoid(self.fc1(x))
        return x

class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        #print('out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        #print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        #print('Residual_out :', out.shape)
        return out


class ResNetTransformerModel(nn.Module):
    def __init__(self, input_channels, num_classes, CONFIG):
        super(ResNetTransformerModel, self).__init__()
        #self.backbone = Wav2Vec2Model.from_pretrained(CONFIG.PRE_MODEL)
        #self.backbone.eval()
        self.resnet = nn.Sequential(
            nn.Sequential(Residual_block([1,32], first=True)),
            nn.Sequential(Residual_block([32,32])),
            nn.Sequential(Residual_block([32,64])),
            nn.Sequential(Residual_block([64,64]))
            )
        
        # nn.Sequential(
        #     nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
        #     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)   # 16x16 -> 8x8
        # )
        self.transformer_input_dim = 192  # 256 채널과 8 (h) 곱
        self.linear = nn.Linear(self.transformer_input_dim, 256)  # 2048 -> 256
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):      
        x = x.unsqueeze(1)
        x = self.resnet(x)  # (batch, 256, h, w)
        x = x.permute(0, 2, 1, 3)  # (batch, h, 256, w)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch, h, 256 * w)
        x = self.linear(x)  # (batch, h, 256)
        x = x.permute(1, 0, 2)  # (h, batch, 256)
        x = self.transformer(x)  # (seq_len, batch, embed_dim)
        x = x.mean(dim=0)  # Global average pooling (batch, 256)
        x = self.fc(x)  # (batch, num_classes)
        x = torch.sigmoid(x)
        return x
