import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.fusion_model import MultimodalFusionModel
from training.config import FUSION_CONFIG, DEVICE, NUM_CLASSES

def train_fusion_model():
    print(f"=== Training Multimodal Attention Fusion Network on {DEVICE} ===")
    
    model = MultimodalFusionModel(num_classes=NUM_CLASSES, num_heads=FUSION_CONFIG["num_heads"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FUSION_CONFIG["lr"], weight_decay=FUSION_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic dataset of tri-modal embeddings (200, 128d each)
    torch.manual_seed(42)
    v_feats = torch.randn(200, 128)
    a_feats = torch.randn(200, 128)
    t_feats = torch.randn(200, 128)
    labels = torch.randint(0, NUM_CLASSES, (200,))
    
    dataset = TensorDataset(v_feats, a_feats, t_feats, labels)
    train_loader = DataLoader(dataset, batch_size=FUSION_CONFIG["batch_size"], shuffle=True)
    
    model.train()
    for epoch in range(1, 4):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for vf, af, tf, lbls in train_loader:
            vf, af, tf, lbls = vf.to(DEVICE), af.to(DEVICE), tf.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(v_embed=vf, a_embed=af, t_embed=tf, fusion_method="attention_fusion")
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * vf.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += vf.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}/3 - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")
        
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "fusion"))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multimodal_fusion.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Fusion model checkpoint saved to: {save_path}\n")

if __name__ == "__main__":
    train_fusion_model()
