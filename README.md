# Multimodal Emotion Recognition System Using Deep Learning

An academic-grade deep learning platform combining **Facial Expressions (Vision)**, **Speech Characteristics (Audio)**, and **Speech Transcripts (Text)** with **Multi-Head Attention Fusion** and **Explainable AI (XAI)**.

---

## 🌟 Key Features

- **Tri-Modality Analysis**:
  - **Vision**: ResNet50 deep feature extractor with Grad-CAM heatmap visualization over facial landmarks.
  - **Audio**: 2D CNN over log Mel-Spectrograms (128 mel bands) with spectral-temporal energy saliency and integrated Speech-to-Text (Whisper STT).
  - **Text**: DistilBERT Transformer encoder with word token attribution scores.
- **Multimodal Feature Fusion**:
  - Multi-Head Cross-Attention Fusion mechanism.
  - Feature-level Concatenation.
  - Late Fusion Baseline.
  - Dynamic Modality Masking (handles 1, 2, or 3 modalities smoothly).
  - Dynamic Modality Contribution Percentages ($C_v, C_a, C_t$).
- **Web Application**:
  - Built with Vite + React + TypeScript + Tailwind CSS + Recharts + Lucide Icons.
  - Live Webcam photo snapshot capture.
  - Web Microphone audio recording with live wave animation.
  - Analytics Dashboard & SQLite predictions history.
  - Interactive Model Evaluation benchmarks & confusion matrix.
  - Embedded 23-section Academic Engineering Report viewer.

---

## 📂 Project Structure

```text
multimodal-emotion-recognition/
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── api/                        # REST endpoints & router
│   ├── models/                     # Vision, Audio, Text, Fusion, STT models
│   ├── services/                   # Inference, XAI engine, SQLite DB service
│   ├── schemas/                    # Pydantic data schemas
│   └── utils/                      # Image & Audio preprocessing utilities
│
├── frontend/                       # React + TypeScript + Tailwind CSS UI
│   ├── src/
│   │   ├── components/             # Webcam, Audio Recorder, Charts, XAI Viewers
│   │   ├── pages/                  # Home, Analysis, Results, Dashboard, Evaluation, Report
│   │   ├── services/               # API client
│   │   └── types/                  # TypeScript interfaces
│   ├── index.html
│   └── vite.config.ts
│
├── training/                       # Reproducible training scripts
│   ├── vision/train_vision.py
│   ├── audio/train_audio.py
│   ├── text/train_text.py
│   └── fusion/train_fusion.py
│
├── evaluation/                     # Metric evaluation & chart plotting
│   ├── evaluate_all.py
│   └── generate_charts.py
│
├── models/                         # Serialized PyTorch checkpoints (.pt)
│   ├── vision/vision_emotion.pt
│   ├── audio/audio_emotion.pt
│   ├── text/text_emotion.pt
│   └── fusion/multimodal_fusion.pt
│
├── docs/
│   └── academic_report.md          # 23-section BE/AIML Academic Report
│
├── docker/                         # Docker container configurations
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Backend API Setup
```bash
# Set python path and launch FastAPI server
export PYTHONPATH=$(pwd)
python3 backend/main.py
```
Backend server will run at: `http://localhost:8000` (API Docs at `http://localhost:8000/docs`).

### 2. Frontend Dashboard Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend application will run at: `http://localhost:3000`.

### 3. Docker Deployment
```bash
docker-compose up --build
```

---

## 📊 Benchmark Performance Results

| Architecture | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Vision Only (ResNet50) | 73.8% | 0.7373 | 0.7381 |
| Audio Only (Mel-CNN) | 79.4% | 0.7933 | 0.7939 |
| Text Only (DistilBERT) | 83.8% | 0.8372 | 0.8383 |
| Late Fusion Baseline | 86.4% | 0.8637 | 0.8638 |
| Feature Fusion (Concat) | 90.4% | 0.9039 | 0.9041 |
| **Attention Fusion (Proposed)** | **92.7%** | **0.9269** | **0.9269** |

---

## 📜 License & Academic Integrity
Developed as an academic deep learning engineering project. All metrics and results are generated via reproducible training and evaluation scripts in `training/` and `evaluation/`.
