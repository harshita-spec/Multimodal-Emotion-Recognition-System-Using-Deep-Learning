import torch
import torch.nn as nn

class AudioEmotionModel(nn.Module):
    """
    2D CNN architecture for Mel-Spectrogram audio emotion recognition.
    """
    def __init__(self, num_classes: int = 7, embedding_dim: int = 128):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # (32, 64, 64)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # (64, 32, 32)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # (128, 16, 16)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # (256, 4, 4)
        )
        
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        embed = self.embedding_head(out)
        return embed

    def forward(self, x: torch.Tensor):
        embed = self.extract_features(x)
        logits = self.classifier(embed)
        return logits, embed
