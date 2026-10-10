from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

EMOTION_CLASSES = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"]

class ModalityPrediction(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]

class XaiExplanation(BaseModel):
    vision_gradcam_base64: Optional[str] = None
    audio_spectrogram_base64: Optional[str] = None
    text_token_attributions: Optional[List[Dict[str, Any]]] = None
    modality_contributions: Dict[str, float] = Field(
        default_factory=lambda: {"vision": 0.0, "audio": 0.0, "text": 0.0}
    )

class MultimodalAnalysisResponse(BaseModel):
    id: str
    timestamp: str
    final_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    fusion_method: str
    used_modalities: List[str]
    transcript: Optional[str] = None
    modality_predictions: Dict[str, ModalityPrediction]
    explanation: XaiExplanation
    processing_time_ms: float

class HistoryItem(BaseModel):
    id: str
    timestamp: str
    final_emotion: str
    confidence: float
    used_modalities: List[str]
    fusion_method: str

class ModelStatus(BaseModel):
    vision_model_loaded: bool
    audio_model_loaded: bool
    text_model_loaded: bool
    fusion_model_loaded: bool
    device: str
    num_classes: int = 7
    emotions: List[str] = EMOTION_CLASSES
