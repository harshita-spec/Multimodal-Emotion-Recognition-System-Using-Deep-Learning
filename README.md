
🧠 Multimodal Emotion Recognition System Using Deep Learning

<p align="center">
  <strong>AURA-Emotion AI</strong><br>
  An Explainable AI Platform for Vision, Audio, and Text-Based Emotion Recognition
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-TypeScript-61DAFB?logo=react" alt="React">
  <img src="https://img.shields.io/badge/Explainable%20AI-Grad--CAM%20%7C%20Attribution-purple" alt="Explainable AI">
</p>

📌 Overview

The **Multimodal Emotion Recognition System Using Deep Learning** is an academic deep learning platform designed to recognize emotional expressions from three complementary modalities: **facial expressions, speech characteristics, and text**.

The system uses specialized deep learning models for each modality and combines their learned representations through a proposed **Multi-Head Cross-Attention Fusion** mechanism. It also incorporates Explainable AI (XAI) techniques to provide visual and feature-level insights into model predictions.

The platform includes an interactive web application for analyzing inputs, viewing emotion predictions, comparing model performance, inspecting explanations, and reviewing prediction history.

✨ Key Highlights

- Three-modality emotion recognition: Vision, Audio, and Text.
- Attention-based multimodal feature fusion.
- Support for missing modalities through modality-aware masking.
- Explainability using Grad-CAM, audio saliency, and token attribution.
- Interactive dashboard with emotion probability charts.
- Model benchmarks and confusion matrix visualization.
- Prediction history stored using SQLite.
- FastAPI backend and React + TypeScript frontend.
- Docker-based deployment configuration.

🏗️ System Architecture

The system follows a modular architecture:

1. Input Acquisition: Accepts an image, audio recording, text input, or a combination of modalities.
2. Preprocessing: Detects and aligns faces, converts audio into log Mel-spectrograms, and tokenizes text.
3. Feature Extraction: Extracts modality-specific representations using ResNet50, a 2D CNN, and DistilBERT.
4. Multimodal Fusion: Combines available modality representations using attention-based fusion or baseline fusion approaches.
5. Emotion Prediction: Produces a predicted emotion and associated confidence/probability scores.
6. Explainability: Generates visual, audio, textual, and modality-contribution explanations.
7. Visualization and Storage: Displays results in the web interface and stores prediction history.

## 🧩 Core Modules

1. Vision Modality

- Model: ResNet50 with transfer learning.
- Face Detection: OpenCV Haar Cascade.
- Preprocessing: Face detection, alignment, resizing, and ImageNet normalization.
- Output: Emotion prediction and learned visual representation.
- Explainability: Grad-CAM heatmaps based on the ResNet50 `layer4` activations.

2. Audio Modality

- Model: Two-dimensional Convolutional Neural Network (2D CNN).
- Audio Processing: Librosa-based preprocessing at 16 kHz.
- Feature Extraction: Log Mel-spectrogram with 128 Mel bands.
- Preprocessing: Silence trimming and peak-amplitude normalization.
- Output: Emotion prediction and audio feature representation.
- Explainability: Spectral-temporal energy visualization.
- Speech-to-Text: Whisper integration for converting speech into text.

3. Text Modality

- Model: DistilBERT Transformer encoder.
- Feature Extraction: Contextual representation from the CLS token.
- Output: Emotion prediction and text feature representation.
- Explainability: Token-level attribution to highlight words relevant to the prediction.

4. Multimodal Fusion

The system provides three fusion strategies:

| Fusion Method | Description |
|---|---|
| Attention Fusion | Multi-Head Cross-Attention to combine modality representations. |
| Feature Fusion | Concatenation of modality feature representations. |
| Late Fusion | Combines modality-level prediction information. |

Dynamic Modality Masking enables the fusion model to handle one, two, or three available modalities.

The application also displays a modality contribution breakdown based on the implemented dynamic norm-based calculation. These percentages are model-derived indicators and should not be interpreted as proof of causal importance.

😊 Emotion Categories

The system recognizes seven emotion classes:

- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

🖥️ Web Application Features

The web interface is built using Vite, React, TypeScript, and Tailwind CSS.

| Page | Functionality |
|---|---|
| Home | Project overview and modality introduction. |
| Emotion Analysis | Image upload, webcam capture, audio recording, and text input. |
| Results | Predicted emotion, confidence, probability distribution, and modality breakdown. |
| Explainable AI | Grad-CAM, audio saliency, token attribution, and modality contribution visualizations. |
| Analytics Dashboard | Prediction statistics and recent history. |
| Model Benchmarks | Performance comparison and confusion matrix. |
| Academic Report | Embedded engineering project report. |

🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python, TypeScript |
| Deep Learning | PyTorch, ResNet50, 2D CNN, DistilBERT |
| Computer Vision | OpenCV |
| Audio Processing | Librosa, Whisper |
| Natural Language Processing | Hugging Face Transformers |
| Backend | FastAPI |
| Frontend | React, Vite, Tailwind CSS |
| Data Visualization | Recharts, Lucide Icons |
| Database | SQLite |
| Deployment | Docker, Docker Compose |
| Evaluation | Accuracy, Macro F1-score, Weighted F1-score |

📂 Project Structure

```text
multimodal-emotion-recognition/
├── backend/
│   ├── main.py
│   ├── api/
│   ├── models/
│   ├── services/
│   ├── schemas/
│   └── utils/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── types/
│   ├── index.html
│   └── vite.config.ts
├── training/
│   ├── vision/train_vision.py
│   ├── audio/train_audio.py
│   ├── text/train_text.py
│   └── fusion/train_fusion.py
├── evaluation/
│   ├── evaluate_all.py
│   └── generate_charts.py
├── models/
│   ├── vision/vision_emotion.pt
│   ├── audio/audio_emotion.pt
│   ├── text/text_emotion.pt
│   └── fusion/multimodal_fusion.pt
├── docs/
│   └── academic_report.md
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

⚙️ Installation and Setup

Prerequisites

Install the following before running the project:

- Python 3.x compatible with the project's dependencies.
- Node.js and npm.
- Git.
- A compatible environment for PyTorch and the required model checkpoints.
- Webcam and microphone for live input features, if needed.

1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
```

Replace `YOUR_USERNAME` and the repository name with your actual GitHub details.

2. Set Up the Backend

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the Python module path and start the backend:

```bash
export PYTHONPATH="$PWD"
python3 backend/main.py
```

Backend URL:

`http://localhost:8000`

Interactive API documentation:

`http://localhost:8000/docs`

> On Windows, activate the virtual environment using `.venv\Scripts\activate` and set `PYTHONPATH` using the appropriate shell syntax.

3. Set Up the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the frontend URL displayed by Vite. If configured for port 3000, the application will be available at:

`http://localhost:3000`

4. Run with Docker

From the project root:

```bash
docker-compose up --build
```

Ensure that the Docker configuration, model checkpoints, environment variables, and required ports are correctly configured.

🔌 API Endpoints

The backend exposes the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/analyze/image` | Analyze an image. |
| POST | `/api/analyze/audio` | Analyze an audio recording. |
| POST | `/api/analyze/text` | Analyze text input. |
| POST | `/api/analyze/multimodal` | Analyze combined modality inputs. |
| GET | `/api/history` | Retrieve prediction history. |
| GET | `/api/history/{id}` | Retrieve a specific history record. |
| GET | `/api/models/status` | Check model loading status. |
| GET | `/api/evaluation/metrics` | Retrieve evaluation metrics. |
| GET | `/api/health` | Check backend health. |

Actual request formats and response schemas can be inspected through the FastAPI documentation.

📊 Benchmark Results

The following results are reported by the project's evaluation scripts on a 1,000-sample evaluation set.

| Architecture | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Vision Only (ResNet50) | 73.8% | 0.7373 | 0.7381 |
| Audio Only (Mel-CNN) | 79.4% | 0.7933 | 0.7939 |
| Text Only (DistilBERT) | 83.8% | 0.8372 | 0.8383 |
| Late Fusion Baseline | 86.4% | 0.8637 | 0.8638 |
| Feature Fusion (Concat) | 90.4% | 0.9039 | 0.9041 |
| **Attention Fusion (Proposed)** | **92.7%** | **0.9269** | **0.9269** |

Key Observation

The proposed Attention Fusion model achieves the highest reported accuracy of **92.7%**, compared with 90.4% for Feature Fusion and 86.4% for Late Fusion.

These metrics describe the reported evaluation setup. For reproducibility, document the dataset, class distribution, data split, random seeds, and evaluation procedure used to obtain them.

🔍 Explainable AI

The project includes several techniques to help users inspect model predictions:

- Vision: Grad-CAM heatmaps visualize image regions associated with a model prediction.
- Audio: Spectral-temporal saliency highlights relevant regions of the audio representation.
- Text: Token attribution assigns importance scores to individual tokens.
- Multimodal Fusion: Contribution percentages summarize the relative representation contributions produced by the implemented calculation.

These explanations provide insight into model behavior but do not guarantee that a prediction is correct or that a highlighted feature caused the predicted emotion.

🧪 Evaluation

Run the evaluation script from the project root:

```bash
python evaluation/evaluate_all.py
```

Generate evaluation charts:

```bash
python evaluation/generate_charts.py
```

The evaluation module is intended to compare individual modality models and fusion strategies using classification metrics and visualizations. Verify the configured dataset paths and model checkpoints before execution.

🔐 Privacy and Responsible Use

- Obtain appropriate consent before recording or analyzing a person's face, voice, or text.
- Avoid using emotion predictions as definitive evidence of a person's internal emotional state.
- Consider bias, cultural differences, recording conditions, and model limitations.
- Protect stored prediction history and any personal data.
- Do not use the system as the sole basis for high-impact decisions.

🚧 Limitations

- Predictions can be affected by poor lighting, background noise, and low-quality input.
- Facial expressions, speech patterns, and written text may not reliably represent a person's actual emotional state.
- Performance depends on dataset quality, diversity, and evaluation methodology.
- Deep learning inference may require significant computational resources.
- Modality contribution values are descriptive model outputs, not causal explanations.

🔮 Future Enhancements

Potential extensions include:

- Support for additional languages and accents.
- Real-time continuous emotion analysis.
- Evaluation on larger and more diverse datasets.
- Improved fusion and modality weighting strategies.
- Enhanced explanation methods.
- Optimized inference and cloud deployment.
- More extensive robustness, fairness, and cross-dataset evaluation.

📚 References

1. K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016.
2. A. Vaswani et al., “Attention Is All You Need,” *Advances in Neural Information Processing Systems*, 2017.
3. V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter,” 2020.
4. A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” 2022.
5. R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” *Proceedings of the IEEE International Conference on Computer Vision*, 2017.

👩‍💻 Academic Project

This project demonstrates the integration of deep learning, multimodal feature fusion, explainable AI, REST API development, and interactive web technologies in an academic engineering application.

Project: Multimodal Emotion Recognition System Using Deep Learning  
Platform: AURA-Emotion AI  
Focus Areas: Artificial Intelligence, Machine Learning, Deep Learning, Multimodal Learning, Explainable AI


