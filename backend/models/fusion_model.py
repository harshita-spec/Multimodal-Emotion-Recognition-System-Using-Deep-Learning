import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class MultimodalFusionModel(nn.Module):
    """
    Multimodal Emotion Recognition Fusion Model.
    Supports Feature Fusion, Late Fusion, and Multi-Head Cross-Attention Fusion.
    Gracefully handles missing modalities via dynamic modality masking.
    """
    def __init__(self, embedding_dim: int = 128, num_classes: int = 7, num_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Multi-Head Attention Fusion layer
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        
        # Feature-level Concatenation Fusion Classifier (3 * 128 = 384)
        self.feature_fusion_head = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Attention Fusion Classifier (128d fused state)
        self.attn_fusion_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        v_embed: Optional[torch.Tensor] = None,
        a_embed: Optional[torch.Tensor] = None,
        t_embed: Optional[torch.Tensor] = None,
        fusion_method: str = "attention_fusion"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass handling 1, 2, or 3 modalities.
        Returns:
            logits (Tensor): (batch_size, num_classes)
            contributions (Dict[str, float]): percentage split of modality contribution
        """
        device = next(self.parameters()).device
        batch_size = 1
        for emb in (v_embed, a_embed, t_embed):
            if emb is not None:
                batch_size = emb.shape[0]
                device = emb.device
                break
                
        # Zero placeholders for missing modalities
        zero_embed = torch.zeros((batch_size, self.embedding_dim), device=device)
        
        has_v = v_embed is not None
        has_a = a_embed is not None
        has_t = t_embed is not None
        
        v_feat = v_embed if has_v else zero_embed
        a_feat = a_embed if has_a else zero_embed
        t_feat = t_embed if has_t else zero_embed
        
        # Compute mean modality norm contributions across batch
        norm_v = float(torch.norm(v_feat, dim=1).mean().item()) if has_v else 0.0
        norm_a = float(torch.norm(a_feat, dim=1).mean().item()) if has_a else 0.0
        norm_t = float(torch.norm(t_feat, dim=1).mean().item()) if has_t else 0.0
        total_norm = norm_v + norm_a + norm_t
        
        if total_norm > 0:
            c_v = round((norm_v / total_norm) * 100, 1)
            c_a = round((norm_a / total_norm) * 100, 1)
            c_t = round((norm_t / total_norm) * 100, 1)
        else:
            c_v, c_a, c_t = 33.3, 33.3, 33.4
            
        contributions = {"vision": c_v, "audio": c_a, "text": c_t}

        if fusion_method == "feature_fusion":
            concat_feat = torch.cat([v_feat, a_feat, t_feat], dim=1)
            logits = self.feature_fusion_head(concat_feat)
            return logits, contributions

        elif fusion_method == "attention_fusion":
            # Stack modalities into sequence tensor (batch_size, 3, 128)
            seq = torch.stack([v_feat, a_feat, t_feat], dim=1)
            
            # Key padding mask: True indicates position SHOULD BE IGNORED
            padding_mask_single = [not has_v, not has_a, not has_t]
            key_padding_mask = torch.tensor(
                [padding_mask_single] * batch_size,
                device=device,
                dtype=torch.bool
            )
            
            attn_out, attn_weights = self.attention_fusion(
                seq, seq, seq, key_padding_mask=key_padding_mask
            )
            attn_out = self.norm(seq + attn_out) # Residual connection
            
            # Pool sequence dimension (mean pool over active modalities)
            valid_mask = ~key_padding_mask.unsqueeze(-1) # (batch_size, 3, 1)
            masked_out = attn_out * valid_mask.float()
            active_count = valid_mask.float().sum(dim=1).clamp(min=1.0)
            fused_vector = masked_out.sum(dim=1) / active_count
            
            logits = self.attn_fusion_head(fused_vector)
            return logits, contributions

        else:
            # Fallback to simple concatenation
            concat_feat = torch.cat([v_feat, a_feat, t_feat], dim=1)
            logits = self.feature_fusion_head(concat_feat)
            return logits, contributions

def compute_late_fusion(
    v_probs: Optional[Dict[str, float]] = None,
    a_probs: Optional[Dict[str, float]] = None,
    t_probs: Optional[Dict[str, float]] = None,
    emotions: List[str] = None
) -> Tuple[Dict[str, float], str, float]:
    """
    Computes late fusion as a baseline by taking weighted average of modality probability vectors.
    """
    if emotions is None:
        emotions = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"]
        
    active_probs = []
    if v_probs is not None:
        active_probs.append(v_probs)
    if a_probs is not None:
        active_probs.append(a_probs)
    if t_probs is not None:
        active_probs.append(t_probs)
        
    if not active_probs:
        uniform = 1.0 / len(emotions)
        fused = {e: uniform for e in emotions}
        return fused, "Neutral", uniform
        
    fused_probs = {e: 0.0 for e in emotions}
    weight = 1.0 / len(active_probs)
    
    for probs in active_probs:
        for e in emotions:
            fused_probs[e] += probs.get(e, 0.0) * weight
            
    # Normalize sum to 1.0
    total = sum(fused_probs.values())
    if total > 0:
        fused_probs = {e: round(p / total, 4) for e, p in fused_probs.items()}
        
    top_emotion = max(fused_probs, key=fused_probs.get)
    confidence = fused_probs[top_emotion]
    return fused_probs, top_emotion, confidence
