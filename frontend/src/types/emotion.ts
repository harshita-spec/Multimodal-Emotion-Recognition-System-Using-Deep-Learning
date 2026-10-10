export type EmotionClass = 'Happy' | 'Sad' | 'Angry' | 'Fear' | 'Surprise' | 'Disgust' | 'Neutral';

export interface ModalityPrediction {
  emotion: EmotionClass;
  confidence: number;
  probabilities: Record<EmotionClass, number>;
}

export interface TokenAttribution {
  word: string;
  score: number;
}

export interface XaiExplanation {
  vision_gradcam_base64?: string;
  audio_spectrogram_base64?: string;
  text_token_attributions?: TokenAttribution[];
  modality_contributions: {
    vision: number;
    audio: number;
    text: number;
  };
}

export interface MultimodalAnalysisResponse {
  id: string;
  timestamp: string;
  final_emotion: EmotionClass;
  confidence: number;
  probabilities: Record<EmotionClass, number>;
  fusion_method: 'attention_fusion' | 'feature_fusion' | 'late_fusion';
  used_modalities: ('vision' | 'audio' | 'text')[];
  transcript?: string;
  modality_predictions: {
    vision?: ModalityPrediction;
    audio?: ModalityPrediction;
    text?: ModalityPrediction;
  };
  explanation: XaiExplanation;
  processing_time_ms: number;
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  final_emotion: EmotionClass;
  confidence: number;
  used_modalities: string[];
  fusion_method: string;
}

export interface ModelStatus {
  vision_model_loaded: boolean;
  audio_model_loaded: boolean;
  text_model_loaded: boolean;
  fusion_model_loaded: boolean;
  device: string;
  num_classes: number;
  emotions: string[];
}

export interface EvaluationMetrics {
  modalities: Record<string, {
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    precision: number;
    recall: number;
  }>;
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
}
