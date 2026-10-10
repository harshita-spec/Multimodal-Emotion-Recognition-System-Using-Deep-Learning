import { MultimodalAnalysisResponse, HistoryItem, ModelStatus, EvaluationMetrics } from '../types/emotion';

const API_BASE = '/api';

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/models/status`);
  return res.json();
}

export async function fetchHistory(): Promise<HistoryItem[]> {
  const res = await fetch(`${API_BASE}/history?limit=50`);
  return res.json();
}

export async function fetchRecordById(id: string): Promise<MultimodalAnalysisResponse> {
  const res = await fetch(`${API_BASE}/history/${id}`);
  return res.json();
}

export async function fetchEvaluationMetrics(): Promise<EvaluationMetrics> {
  const res = await fetch(`${API_BASE}/evaluation/metrics`);
  return res.json();
}

export async function analyzeMultimodal(
  imageFile: File | null,
  audioBlob: Blob | File | null,
  text: string | null,
  fusionMethod: string = 'attention_fusion',
  autoStt: boolean = true
): Promise<MultimodalAnalysisResponse> {
  const formData = new FormData();

  if (imageFile) {
    formData.append('image', imageFile, imageFile.name || 'image.jpg');
  }

  if (audioBlob) {
    formData.append('audio', audioBlob, audioBlob instanceof File ? audioBlob.name : 'audio.wav');
  }

  if (text && text.trim()) {
    formData.append('text', text.trim());
  }

  formData.append('fusion_method', fusionMethod);
  formData.append('auto_stt', autoStt ? 'true' : 'false');

  const res = await fetch(`${API_BASE}/analyze/multimodal`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Analysis request failed' }));
    throw new Error(errorData.detail || 'Multimodal analysis failed');
  }

  return res.json();
}
