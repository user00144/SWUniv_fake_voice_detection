from transformers import ViTModel
import torch
import torch.nn as nn

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
        
class ViTmodel(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.encoder = ViTModel.from_pretrained("MattyB95/VIT-VoxCelebSpoof-Mel_Spectrogram-Synthetic-Voice-Detection")
        self.resnet = nn.Sequential(
            nn.Sequential(Residual_block([1,32], first=True)),
            nn.Sequential(Residual_block([32,32])),
            nn.Sequential(Residual_block([32,64])))
        self.transformer_input_dim = 1792 
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.transformer_input_dim, 512)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=6)
        self.fc = nn.Linear(512, 2)
    def forward(self, x) :
        with torch.no_grad() :
            x = self.encoder(x)
            x = x.last_hidden_state
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.linear(x)  # (batch, h, 256)
        x = x.permute(1, 0, 2)  # (h, batch, 256)
        x = self.transformer(x)  # (seq_len, batch, embed_dim)
        x = x.mean(dim=0)  # Global average pooling (batch, 256)
        x = self.fc(x)  # (batch, num_classes)
        x = torch.sigmoid(x)
        return x


#torch.Size([96, 197, 768])

class ViTmodel_fcn(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.encoder = ViTModel.from_pretrained("MattyB95/VIT-VoxCelebSpoof-Mel_Spectrogram-Synthetic-Voice-Detection")
        self.flat = nn.Flatten()
        self.fcn = nn.Sequential(nn.Dropout(0.3), nn.ReLU(), nn.Linear(151296, 2))
    def forward(self, x) :
        with torch.no_grad() :
            x = self.encoder(x)
            x = x.last_hidden_state

        x = self.flat(x)
        x = self.fcn(x)
        x = torch.sigmoid(x)
        return x