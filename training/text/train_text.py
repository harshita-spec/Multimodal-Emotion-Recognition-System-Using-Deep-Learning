import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.text_model import TextEmotionModel
from training.config import TEXT_CONFIG, DEVICE, NUM_CLASSES

def train_text_model():
    print(f"=== Training Text Emotion Model (DistilBERT) on {DEVICE} ===")
    
    model = TextEmotionModel(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEXT_CONFIG["lr"], weight_decay=TEXT_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic dataset representing GoEmotions tokenized text
    torch.manual_seed(42)
    dummy_input_ids = torch.randint(100, 2000, (200, 128))
    dummy_masks = torch.ones((200, 128))
    dummy_labels = torch.randint(0, NUM_CLASSES, (200,))
    
    dataset = TensorDataset(dummy_input_ids, dummy_masks, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=TEXT_CONFIG["batch_size"], shuffle=True)
    
    model.train()
    for epoch in range(1, 4):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for ids, masks, lbls in train_loader:
            ids, masks, lbls = ids.to(DEVICE), masks.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(ids, masks)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * ids.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += ids.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}/3 - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")
        
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "text"))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "text_emotion.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Text model checkpoint saved to: {save_path}\n")

if __name__ == "__main__":
    train_text_model()
