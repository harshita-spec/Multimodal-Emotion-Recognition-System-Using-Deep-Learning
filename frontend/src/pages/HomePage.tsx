import React from 'react';
import { BrainCircuit, Sparkles, Layers, ShieldCheck, Cpu, ArrowRight, Eye, Mic, MessageSquareCode } from 'lucide-react';

interface HomePageProps {
  setActiveTab: (tab: string) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ setActiveTab }) => {
  return (
    <div className="flex flex-col gap-16 py-8">
      {/* Hero Section */}
      <section className="relative overflow-hidden glass-card p-8 md:p-12 border border-indigo-500/20 rounded-3xl bg-gradient-to-b from-slate-900/90 via-slate-950/80 to-slate-950">
        <div className="absolute top-0 right-0 -translate-y-12 translate-x-12 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 translate-y-12 -translate-x-12 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-3xl space-y-6 relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-indigo-500/10 border border-indigo-500/30 rounded-full text-indigo-300 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" />
            BE/AIML Academic Deep Learning Project
          </div>

          <h1 className="text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight">
            Multimodal Emotion Recognition System Using Deep Learning
          </h1>

          <p className="text-slate-300 text-base md:text-lg leading-relaxed">
            A state-of-the-art AI architecture that unifies <strong className="text-indigo-300">Facial Expressions (ResNet50)</strong>, <strong className="text-pink-300">Speech Characteristics (Mel-Spectrogram 2D CNN)</strong>, and <strong className="text-teal-300">Speech Transcripts (DistilBERT)</strong> using multi-head attention feature fusion and Explainable AI (Grad-CAM, Spectrogram Saliency, Token Attribution).
          </p>

          <div className="flex flex-wrap gap-4 pt-2">
            <button
              onClick={() => setActiveTab('analyze')}
              className="flex items-center gap-2 px-6 py-3.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-base rounded-2xl shadow-xl shadow-indigo-500/25 transition-all hover:scale-105"
            >
              Start Emotion Analysis
              <ArrowRight className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('report')}
              className="flex items-center gap-2 px-6 py-3.5 bg-slate-900 hover:bg-slate-800 text-slate-200 font-semibold text-base rounded-2xl border border-slate-700 transition-all"
            >
              View Academic Report
            </button>
          </div>
        </div>
      </section>

      {/* 3 Modalities Overview */}
      <section className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold text-white">Three Complementary Modalities</h2>
          <p className="text-slate-400 text-sm max-w-xl mx-auto">
            Combines spatial visual features, spectral audio characteristics, and semantic text representations.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="glass-card-hover p-6 flex flex-col gap-4">
            <div className="p-3 bg-indigo-500/20 border border-indigo-500/30 rounded-xl w-fit">
              <Eye className="w-6 h-6 text-indigo-400" />
            </div>
            <h3 className="text-lg font-bold text-white">Vision Modality</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              OpenCV face detection & alignment followed by ResNet50 deep feature extraction and Grad-CAM spatial activation mapping.
            </p>
            <span className="text-[11px] font-mono text-indigo-400 bg-indigo-950/60 border border-indigo-800/40 px-2.5 py-1 rounded-lg w-fit">
              ResNet50 (224x224) → 128d
            </span>
          </div>

          <div className="glass-card-hover p-6 flex flex-col gap-4">
            <div className="p-3 bg-pink-500/20 border border-pink-500/30 rounded-xl w-fit">
              <Mic className="w-6 h-6 text-pink-400" />
            </div>
            <h3 className="text-lg font-bold text-white">Audio Modality</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Librosa 16kHz resampling, peak normalization, log Mel-Spectrogram extraction (128 mel bands) fed into a 2D CNN backbone.
            </p>
            <span className="text-[11px] font-mono text-pink-400 bg-pink-950/60 border border-pink-800/40 px-2.5 py-1 rounded-lg w-fit">
              Mel-CNN (128x128) → 128d
            </span>
          </div>

          <div className="glass-card-hover p-6 flex flex-col gap-4">
            <div className="p-3 bg-teal-500/20 border border-teal-500/30 rounded-xl w-fit">
              <MessageSquareCode className="w-6 h-6 text-teal-400" />
            </div>
            <h3 className="text-lg font-bold text-white">Text Modality</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              DistilBERT transformer tokenization with Speech-to-Text (Whisper STT integration) and token-level attribution saliency.
            </p>
            <span className="text-[11px] font-mono text-teal-400 bg-teal-950/60 border border-teal-800/40 px-2.5 py-1 rounded-lg w-fit">
              DistilBERT (CLS 768d) → 128d
            </span>
          </div>
        </div>
      </section>

      {/* System Architecture Flow Diagram */}
      <section className="glass-card p-8 border border-slate-800 space-y-6">
        <div className="flex items-center gap-3">
          <Cpu className="w-6 h-6 text-indigo-400" />
          <h2 className="text-xl font-bold text-white">End-to-End System Pipeline</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
          <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl space-y-2">
            <span className="text-[10px] uppercase font-bold text-slate-500">Stage 1</span>
            <h4 className="text-sm font-semibold text-indigo-300">Multimodal Input</h4>
            <p className="text-[11px] text-slate-400">Webcam snapshot, microphone audio, typed text</p>
          </div>

          <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl space-y-2">
            <span className="text-[10px] uppercase font-bold text-slate-500">Stage 2</span>
            <h4 className="text-sm font-semibold text-purple-300">Feature Encoders</h4>
            <p className="text-[11px] text-slate-400">ResNet50 + Mel-CNN + DistilBERT</p>
          </div>

          <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl space-y-2">
            <span className="text-[10px] uppercase font-bold text-slate-500">Stage 3</span>
            <h4 className="text-sm font-semibold text-pink-300">Attention Fusion</h4>
            <p className="text-[11px] text-slate-400">Cross-Attention + Dynamic Modality Masking</p>
          </div>

          <div className="bg-slate-950 border border-slate-800 p-4 rounded-xl space-y-2">
            <span className="text-[10px] uppercase font-bold text-slate-500">Stage 4</span>
            <h4 className="text-sm font-semibold text-emerald-300">Emotion + XAI</h4>
            <p className="text-[11px] text-slate-400">7-Class Probabilities + Grad-CAM / Spectrogram</p>
          </div>
        </div>
      </section>
    </div>
  );
};
