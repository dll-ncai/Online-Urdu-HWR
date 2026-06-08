import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """CNN feature extractor."""
    def __init__(self):
        super(CNNEncoder, self).__init__()            # 512 * 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(2, 2)  # 256 * 32
            # nn.MaxPool2d((2, 4), (2, 4)),   # 128 * 8
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2)    # 128 * 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((1, 2), (1, 2)),   # 128 * 8
            nn.Dropout2d(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d((1, 2), (1, 2)),    # 128 * 4
            nn.Dropout2d(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 4),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

    def forward(self,
                src: torch.Tensor,
                ):

        src = self.conv1(src)
        # print("CNN Encoder Conv1 output shape:", src.shape)                                 # (*, 16, 32, 256)
        src = self.conv2(src)
        # print("CNN Encoder Conv2 output shape:", src.shape)                                 # (*, 32, 16, 128)
        src = self.conv3(src)
        # print("CNN Encoder Conv3 output shape:", src.shape)                                 # (*, 64, 8, 128)
        src = self.conv4(src)
        # print("CNN Encoder Conv4 output shape:", src.shape)                                 # (*, 128, 4, 128)
        src = self.conv5(src)
        # print("CNN Encoder Conv5 output shape:", src.shape)                                 # (*, 256, 1, 125)
        src = src.squeeze(-1)
        # print("CNN Encoder after squeeze shape:", src.shape)                                 # (*, 256, 125)
        src = src.permute((0, 2, 1)).contiguous()        # (*, 125, 256)
        # print("CNN Encoder after permute shape:", src.shape)                                 # (*, 125, 256)

        return src
