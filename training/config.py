import torch

# Central training configuration parameters
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NUM_CLASSES = 7
EMOTION_CLASSES = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"]

VISION_CONFIG = {
    "batch_size": 32,
    "epochs": 15,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "embedding_dim": 128
}

AUDIO_CONFIG = {
    "batch_size": 32,
    "epochs": 20,
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "n_mels": 128,
    "embedding_dim": 128
}

TEXT_CONFIG = {
    "batch_size": 16,
    "epochs": 10,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "max_len": 128,
    "embedding_dim": 128
}

FUSION_CONFIG = {
    "batch_size": 32,
    "epochs": 25,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "dropout": 0.3,
    "num_heads": 4,
    "embedding_dim": 128
}
