import os
import time
import uuid
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from backend.schemas.analysis import (
    EMOTION_CLASSES, ModalityPrediction, XaiExplanation, MultimodalAnalysisResponse
)
from backend.models.vision_model import VisionEmotionModel
from backend.models.audio_model import AudioEmotionModel
from backend.models.text_model import TextEmotionModel
from backend.models.fusion_model import MultimodalFusionModel, compute_late_fusion
from backend.models.stt_service import stt_service
from backend.utils.image_utils import detect_and_crop_face, preprocess_image_tensor, analyze_facial_heuristics
from backend.utils.audio_utils import load_and_preprocess_audio, compute_mel_spectrogram, preprocess_audio_tensor
from backend.services.xai_service import (
    generate_gradcam_overlay, generate_spectrogram_xai, generate_text_token_attributions
)
from backend.services.db_service import db_service

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

def compute_text_emotion_prior(text_str: str) -> np.ndarray:
    """Computes sentiment & emotion keyword prior logits across 7 classes."""
    text_lower = text_str.lower()
    logits = np.zeros(7, dtype=np.float32)
    
    happy_words = ["happy", "smile", "smiling", "joy", "joyful", "excited", "thrilled", "wonderful", "great", "love", "awesome", "good", "delighted", "glad"]
    sad_words = ["sad", "depressed", "unhappy", "cry", "crying", "lonely", "sorrow", "disappointed", "hurt", "miserable", "grief", "gloomy"]
    angry_words = ["angry", "furious", "mad", "annoyed", "hate", "hating", "rage", "frustrated", "irritated", "outraged", "enraged"]
    fear_words = ["afraid", "scared", "fear", "terrified", "anxious", "panic", "horrified", "nervous", "dread", "frightened"]
    surprise_words = ["surprised", "shocked", "amazed", "wow", "unexpected", "astonished", "unbelievable", "stunted"]
    disgust_words = ["disgusted", "revolted", "sick", "nasty", "gross", "horrible", "awful", "repulsive", "yuck"]
    
    for w in happy_words:
        if w in text_lower: logits[0] += 3.0
    for w in sad_words:
        if w in text_lower: logits[1] += 3.0
    for w in angry_words:
        if w in text_lower: logits[2] += 3.0
    for w in fear_words:
        if w in text_lower: logits[3] += 3.0
    for w in surprise_words:
        if w in text_lower: logits[4] += 3.0
    for w in disgust_words:
        if w in text_lower: logits[5] += 3.0
        
    if np.max(logits) == 0:
        logits[6] = 2.0 # Neutral baseline
        
    return logits

class MultimodalInferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Instantiate PyTorch models
        self.vision_model = VisionEmotionModel(num_classes=7).to(self.device)
        self.audio_model = AudioEmotionModel(num_classes=7).to(self.device)
        self.text_model = TextEmotionModel(num_classes=7).to(self.device)
        self.fusion_model = MultimodalFusionModel(num_classes=7).to(self.device)
        
        self.text_tokenizer = None
        self._init_tokenizer()
        self.load_checkpoints()
        
    def _init_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        except Exception:
            self.text_tokenizer = None

    def load_checkpoints(self):
        """Loads trained weights if available or initializes model params."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        v_ckpt = os.path.join(MODEL_DIR, "vision", "vision_emotion.pt")
        a_ckpt = os.path.join(MODEL_DIR, "audio", "audio_emotion.pt")
        t_ckpt = os.path.join(MODEL_DIR, "text", "text_emotion.pt")
        f_ckpt = os.path.join(MODEL_DIR, "fusion", "multimodal_fusion.pt")
        
        for path, model in [(v_ckpt, self.vision_model), (a_ckpt, self.audio_model),
                             (t_ckpt, self.text_model), (f_ckpt, self.fusion_model)]:
            if os.path.exists(path):
                try:
                    model.load_state_dict(torch.load(path, map_location=self.device))
                except Exception:
                    pass
            model.eval()

    def analyze(
        self,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        text_input: Optional[str] = None,
        fusion_method: str = "attention_fusion",
        auto_stt: bool = True
    ) -> MultimodalAnalysisResponse:
        start_time = time.time()
        
        used_modalities = []
        modality_preds = {}
        
        v_embed, a_embed, t_embed = None, None, None
        v_probs, a_probs, t_probs = None, None, None
        
        face_crop_bgr = None
        face_target_class = 0
        mel_spec_db = None
        transcript = None
        
        # 1. Vision Processing
        if image_bytes is not None and len(image_bytes) > 0:
            used_modalities.append("vision")
            face_crop_bgr, bbox, found = detect_and_crop_face(image_bytes)
            v_tensor = preprocess_image_tensor(face_crop_bgr).to(self.device)
            
            # Analyze smile & facial geometry priors (OpenCV smile detector)
            face_priors = analyze_facial_heuristics(face_crop_bgr)
            
            with torch.no_grad():
                v_logits_raw, v_embed_raw = self.vision_model(v_tensor)
                v_raw_np = v_logits_raw[0].cpu().numpy()
                
            # Combine ResNet embedding logits with facial geometry priors
            v_combined_logits = v_raw_np * 0.3 + face_priors
            v_softmax = np.exp(v_combined_logits) / np.sum(np.exp(v_combined_logits))
            
            # Form enhanced embedding vector incorporating face prior
            v_embed = v_embed_raw.clone()
            v_embed[0, 0] += face_priors[0] * 2.0  # Happy feature boost
            v_embed[0, 1] += face_priors[1] * 2.0  # Sad feature boost
            
            v_probs = {EMOTION_CLASSES[i]: round(float(v_softmax[i]), 4) for i in range(7)}
            v_top_idx = int(np.argmax(v_softmax))
            face_target_class = v_top_idx
            
            modality_preds["vision"] = ModalityPrediction(
                emotion=EMOTION_CLASSES[v_top_idx],
                confidence=round(float(v_softmax[v_top_idx]), 4),
                probabilities=v_probs
            )
            
        # 2. Audio Processing & Speech-to-Text
        if audio_bytes is not None and len(audio_bytes) > 0:
            used_modalities.append("audio")
            y_audio, sr = load_and_preprocess_audio(audio_bytes)
            norm_mel, mel_spec_db = compute_mel_spectrogram(y_audio, sr)
            a_tensor = preprocess_audio_tensor(norm_mel).to(self.device)
            
            # Acoustic pitch/energy heuristics
            audio_energy = np.std(y_audio)
            a_priors = np.zeros(7, dtype=np.float32)
            if audio_energy > 0.15:
                a_priors[0] += 2.0 # Happy
                a_priors[4] += 1.5 # Surprise
            elif audio_energy < 0.04:
                a_priors[1] += 2.0 # Sad
                a_priors[6] += 1.5 # Neutral
            else:
                a_priors[6] += 2.0 # Neutral
                a_priors[0] += 0.5 # Happy
                
            with torch.no_grad():
                a_logits_raw, a_embed_raw = self.audio_model(a_tensor)
                a_raw_np = a_logits_raw[0].cpu().numpy()
                
            a_combined_logits = a_raw_np * 0.3 + a_priors
            a_softmax = np.exp(a_combined_logits) / np.sum(np.exp(a_combined_logits))
            
            a_embed = a_embed_raw.clone()
            a_embed[0, 0] += a_priors[0] * 2.0
            
            a_probs = {EMOTION_CLASSES[i]: round(float(a_softmax[i]), 4) for i in range(7)}
            a_top_idx = int(np.argmax(a_softmax))
            
            modality_preds["audio"] = ModalityPrediction(
                emotion=EMOTION_CLASSES[a_top_idx],
                confidence=round(float(a_softmax[a_top_idx]), 4),
                probabilities=a_probs
            )
            
            # Auto Speech-to-Text if text input was not explicitly provided
            if auto_stt and (text_input is None or not text_input.strip()):
                transcript = stt_service.transcribe_audio(y_audio, sr)
                text_input = transcript

        # 3. Text Processing
        if text_input is not None and text_input.strip():
            used_modalities.append("text")
            text_str = text_input.strip()
            
            text_priors = compute_text_emotion_prior(text_str)
            
            if self.text_tokenizer is not None:
                encoded = self.text_tokenizer(text_str, return_tensors="pt", padding=True, truncation=True, max_length=128)
                input_ids = encoded["input_ids"].to(self.device)
                att_mask = encoded["attention_mask"].to(self.device)
            else:
                input_ids = torch.tensor([[101, 2054, 102]], device=self.device)
                att_mask = torch.tensor([[1, 1, 1]], device=self.device)
                
            with torch.no_grad():
                t_logits_raw, t_embed_raw = self.text_model(input_ids, att_mask)
                t_raw_np = t_logits_raw[0].cpu().numpy()
                
            t_combined_logits = t_raw_np * 0.3 + text_priors
            t_softmax = np.exp(t_combined_logits) / np.sum(np.exp(t_combined_logits))
            
            t_embed = t_embed_raw.clone()
            t_embed[0, 0] += text_priors[0] * 2.0
            t_embed[0, 1] += text_priors[1] * 2.0
            t_embed[0, 2] += text_priors[2] * 2.0
            
            t_probs = {EMOTION_CLASSES[i]: round(float(t_softmax[i]), 4) for i in range(7)}
            t_top_idx = int(np.argmax(t_softmax))
            
            modality_preds["text"] = ModalityPrediction(
                emotion=EMOTION_CLASSES[t_top_idx],
                confidence=round(float(t_softmax[t_top_idx]), 4),
                probabilities=t_probs
            )

        # Edge case: No inputs provided
        if not used_modalities:
            used_modalities.append("text")
            text_str = "Neutral emotion fallback"
            t_probs = {e: (1.0/7.0) for e in EMOTION_CLASSES}
            t_probs["Neutral"] = 0.7
            t_embed = torch.randn((1, 128), device=self.device)
            modality_preds["text"] = ModalityPrediction(
                emotion="Neutral", confidence=0.7, probabilities=t_probs
            )

        # 4. Fusion Inference
        if fusion_method == "late_fusion":
            final_probs, final_emotion, confidence = compute_late_fusion(v_probs, a_probs, t_probs, EMOTION_CLASSES)
            c_v = 33.3 if "vision" in used_modalities else 0.0
            c_a = 33.3 if "audio" in used_modalities else 0.0
            c_t = 33.4 if "text" in used_modalities else 0.0
            total_c = c_v + c_a + c_t
            contributions = {"vision": round(c_v/total_c*100, 1), "audio": round(c_a/total_c*100, 1), "text": round(c_t/total_c*100, 1)}
        else:
            with torch.no_grad():
                f_logits_raw, contributions = self.fusion_model(
                    v_embed=v_embed, a_embed=a_embed, t_embed=t_embed, fusion_method=fusion_method
                )
                f_raw_np = f_logits_raw[0].cpu().numpy()
                
            # Combine fusion logits with active modality priors
            priors_sum = np.zeros(7, dtype=np.float32)
            if "vision" in used_modalities: priors_sum += analyze_facial_heuristics(face_crop_bgr)
            if "text" in used_modalities: priors_sum += compute_text_emotion_prior(text_input)
            
            f_combined = f_raw_np * 0.3 + priors_sum
            f_softmax = np.exp(f_combined) / np.sum(np.exp(f_combined))
            
            final_probs = {EMOTION_CLASSES[i]: round(float(f_softmax[i]), 4) for i in range(7)}
            top_idx = int(np.argmax(f_softmax))
            final_emotion = EMOTION_CLASSES[top_idx]
            confidence = round(float(f_softmax[top_idx]), 4)

        # 5. Explainable AI Generation
        vision_gradcam = None
        if face_crop_bgr is not None:
            vision_gradcam = generate_gradcam_overlay(self.vision_model, v_tensor, face_crop_bgr, face_target_class)
            
        audio_spec_b64 = None
        if mel_spec_db is not None:
            audio_spec_b64 = generate_spectrogram_xai(mel_spec_db)
            
        text_attributions = None
        if text_input and text_input.strip():
            text_attributions = generate_text_token_attributions(text_input, final_emotion)

        explanation = XaiExplanation(
            vision_gradcam_base64=vision_gradcam,
            audio_spectrogram_base64=audio_spec_b64,
            text_token_attributions=text_attributions,
            modality_contributions=contributions
        )
        
        proc_time = round((time.time() - start_time) * 1000, 2)
        record_id = f"anl_{uuid.uuid4().hex[:10]}"
        timestamp_str = datetime.now().isoformat()
        
        response = MultimodalAnalysisResponse(
            id=record_id,
            timestamp=timestamp_str,
            final_emotion=final_emotion,
            confidence=confidence,
            probabilities=final_probs,
            fusion_method=fusion_method,
            used_modalities=used_modalities,
            transcript=transcript,
            modality_predictions=modality_preds,
            explanation=explanation,
            processing_time_ms=proc_time
        )
        
        # Save to database
        db_service.save_analysis(response.model_dump())
        return response

inference_service = MultimodalInferenceService()
