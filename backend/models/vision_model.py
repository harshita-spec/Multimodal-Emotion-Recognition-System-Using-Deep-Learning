import torch
import torch.nn as nn
import torchvision.models as models

EMOTION_CLASSES = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"]

class VisionEmotionModel(nn.Module):
    def __init__(self, num_classes: int = 7, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        # Use ResNet50 transfer learning backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity() # Remove default ImageNet FC layer
        
        # Feature Projection Head (ResNet features -> 128-dim visual embedding)
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        # Emotion Classifier Head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 128-dimensional visual embedding vector.
        """
        res_feat = self.resnet(x)
        embed = self.embedding_head(res_feat)
        return embed

    def forward(self, x: torch.Tensor):
        embed = self.extract_features(x)
        logits = self.classifier(embed)
        return logits, embed
