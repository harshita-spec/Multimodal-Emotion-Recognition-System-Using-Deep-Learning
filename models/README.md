# Models Directory

This directory contains trained PyTorch model checkpoints.

## Download / Train Instructions

After cloning the repository, run the training scripts to generate model checkpoints:

```bash
# Set Python path
export PYTHONPATH=$(pwd)

# Train Vision Model (ResNet50) on FER2013
python3 training/vision/train_vision.py

# Train Audio Model (Mel-CNN) on RAVDESS/TESS
python3 training/audio/train_audio.py

# Train Text Model (DistilBERT) on GoEmotions
python3 training/text/train_text.py

# Train Fusion Model (Attention Fusion)
python3 training/fusion/train_fusion.py
```

Checkpoints will be saved to:
- `models/vision/vision_emotion.pt`
- `models/audio/audio_emotion.pt`
- `models/text/text_emotion.pt`
- `models/fusion/multimodal_fusion.pt`
