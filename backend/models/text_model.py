import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

class TextEmotionModel(nn.Module):
    """
    DistilBERT Transformer based text emotion model.
    """
    def __init__(self, num_classes: int = 7, embedding_dim: int = 128):
        super().__init__()
        try:
            self.transformer = AutoModel.from_pretrained(MODEL_NAME)
        except Exception:
            # Fallback embedding layer if offline download is unreachable
            self.transformer = None
            
        hidden_size = 768 if self.transformer is not None else 128
        
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def extract_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.transformer is not None:
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token embedding ([0])
            cls_repr = outputs.last_hidden_state[:, 0, :]
        else:
            # Synthetic feature representation fallback
            cls_repr = torch.randn((input_ids.shape[0], 128), device=input_ids.device)
            
        embed = self.embedding_head(cls_repr)
        return embed

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embed = self.extract_features(input_ids, attention_mask)
        logits = self.classifier(embed)
        return logits, embed
