import io
import numpy as np
import librosa
import torch
import cv2

TARGET_SR = 16000
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048

def load_and_preprocess_audio(audio_bytes: bytes, target_sr: int = TARGET_SR):
    """
    Load raw audio bytes (wav/mp3), resample to 16kHz, remove silence, and normalize.
    """
    buffer = io.BytesIO(audio_bytes)
    try:
        y, sr = librosa.load(buffer, sr=target_sr, mono=True)
    except Exception as e:
        # Fallback using scipy or raw float conversion if librosa load fails
        import soundfile as sf
        buffer.seek(0)
        y, sr = sf.read(buffer)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Silence trimming
    yt, _ = librosa.effects.trim(y, top_db=25)
    if len(yt) < target_sr * 0.5:
        yt = y # Keep full audio if trimming cut off too much
        
    # Peak normalization
    max_val = np.max(np.abs(yt))
    if max_val > 0:
        yt = yt / max_val
        
    return yt, target_sr

def compute_mel_spectrogram(y: np.ndarray, sr: int = TARGET_SR, n_mels: int = N_MELS):
    """
    Computes log Mel-Spectrogram matrix (128, T) normalized to [0, 1].
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Normalize to [0, 1] range for CNN model input
    s_min, s_max = S_dB.min(), S_dB.max()
    if s_max > s_min:
        norm_S = (S_dB - s_min) / (s_max - s_min)
    else:
        norm_S = np.zeros_like(S_dB)
        
    return norm_S, S_dB

def preprocess_audio_tensor(mel_spec: np.ndarray, fixed_width: int = 128):
    """
    Resize/pad Mel-spectrogram to fixed 2D dimensions (128 x 128) and return 4D tensor (1, 1, 128, 128).
    """
    spec_resized = cv2.resize(mel_spec, (fixed_width, N_MELS), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(spec_resized).float().unsqueeze(0).unsqueeze(0)
    return tensor
