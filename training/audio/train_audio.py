import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.models.audio_model import AudioEmotionModel
from training.config import AUDIO_CONFIG, DEVICE, NUM_CLASSES

def train_audio_model():
    print(f"=== Training Audio Emotion Model (2D CNN Spectrogram) on {DEVICE} ===")
    
    model = AudioEmotionModel(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=AUDIO_CONFIG["lr"], weight_decay=AUDIO_CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic dataset representing RAVDESS / TESS Mel-Spectrograms (128x128)
    torch.manual_seed(42)
    dummy_specs = torch.randn(200, 1, 128, 128)
    dummy_labels = torch.randint(0, NUM_CLASSES, (200,))
    
    dataset = TensorDataset(dummy_specs, dummy_labels)
    train_loader = DataLoader(dataset, batch_size=AUDIO_CONFIG["batch_size"], shuffle=True)
    
    model.train()
    for epoch in range(1, 4):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for specs, lbls in train_loader:
            specs, lbls = specs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(specs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * specs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == lbls).sum().item()
            total += specs.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}/3 - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")
        
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "audio"))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "audio_emotion.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Audio model checkpoint saved to: {save_path}\n")

if __name__ == "__main__":
    train_audio_model()
