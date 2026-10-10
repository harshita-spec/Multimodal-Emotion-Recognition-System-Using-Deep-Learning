import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.vision_model import VisionEmotionModel
from training.config import VISION_CONFIG, DEVICE, NUM_CLASSES, EMOTION_CLASSES

def train_vision_model():
    print(f"=== Training Vision Emotion Model (ResNet50) on {DEVICE} ===")
    
    model = VisionEmotionModel(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=VISION_CONFIG["lr"], weight_decay=VISION_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Synthetic Dataset placeholder representing preprocessed FER2013 facial images
    torch.manual_seed(42)
    dummy_images = torch.randn(200, 3, 224, 224)
    dummy_labels = torch.randint(0, NUM_CLASSES, (200,))
    
    dataset = TensorDataset(dummy_images, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=VISION_CONFIG["batch_size"], shuffle=True)
    
    model.train()
    for epoch in range(1, 4): # Demonstration epochs
        total_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += imgs.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch}/3 - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")
        
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "vision"))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vision_emotion.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Vision model checkpoint saved to: {save_path}\n")

if __name__ == "__main__":
    train_vision_model()
