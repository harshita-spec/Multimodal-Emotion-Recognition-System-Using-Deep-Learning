import React from 'react';
import { MultimodalAnalysisResponse, EmotionClass } from '../types/emotion';
import { EmotionProbabilityChart } from '../components/EmotionProbabilityChart';
import { ModalityContributionChart } from '../components/ModalityContributionChart';
import { XaiVisualizer } from '../components/XaiVisualizer';
import { ArrowLeft, Clock, Layers, Sparkles, CheckCircle2 } from 'lucide-react';

interface ResultsPageProps {
  result: MultimodalAnalysisResponse;
  onBack: () => void;
}

const EMOTION_EMOJIS: Record<EmotionClass, string> = {
  Happy: '😊',
  Sad: '😢',
  Angry: '😡',
  Fear: '😨',
  Surprise: '😲',
  Disgust: '🤢',
  Neutral: '😐',
};

const EMOTION_BADGE_COLORS: Record<EmotionClass, string> = {
  Happy: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
  Sad: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
  Angry: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
  Fear: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
  Surprise: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
  Disgust: 'bg-lime-500/20 text-lime-300 border-lime-500/40',
  Neutral: 'bg-slate-500/20 text-slate-300 border-slate-500/40',
};

export const ResultsPage: React.FC<ResultsPageProps> = ({ result, onBack }) => {
  const emoji = EMOTION_EMOJIS[result.final_emotion] || '😐';
  const badgeColor = EMOTION_BADGE_COLORS[result.final_emotion] || EMOTION_BADGE_COLORS.Neutral;
  const confPercent = Math.round(result.confidence * 1000) / 10;

  return (
    <div className="flex flex-col gap-8 py-4 max-w-6xl mx-auto">
      {/* Top Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-white transition-all bg-slate-900 px-3.5 py-2 rounded-xl border border-slate-800"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Analysis
        </button>

        <div className="flex items-center gap-3 text-xs text-slate-400 bg-slate-950 px-3.5 py-2 rounded-xl border border-slate-800">
          <Clock className="w-3.5 h-3.5 text-indigo-400" />
          Latency: <strong className="text-slate-200">{result.processing_time_ms} ms</strong>
          <span className="text-slate-600">|</span>
          <Layers className="w-3.5 h-3.5 text-purple-400" />
          Fusion: <strong className="text-slate-200 capitalize">{result.fusion_method.replace('_', ' ')}</strong>
        </div>
      </div>

      {/* Main Final Emotion Card */}
      <div className="glass-card p-8 border border-indigo-500/30 bg-gradient-to-r from-slate-900 via-indigo-950/20 to-slate-900 flex flex-col md:flex-row items-center justify-between gap-8">
        <div className="flex items-center gap-6">
          <div className="text-6xl md:text-7xl p-4 bg-slate-950/80 border border-slate-800 rounded-3xl shadow-2xl animate-bounce">
            {emoji}
          </div>
          <div className="space-y-2">
            <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Predicted Final Emotion
            </span>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl md:text-4xl font-black text-white tracking-tight">
                {result.final_emotion.toUpperCase()}
              </h1>
              <span className={`px-3 py-1 rounded-full text-xs font-bold border ${badgeColor}`}>
                {confPercent}% Confidence
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Modalities Used:{' '}
              <strong className="text-indigo-300 uppercase">
                {result.used_modalities.join(' + ')}
              </strong>
            </p>
          </div>
        </div>

        {/* Speech Transcript Display */}
        {result.transcript && (
          <div className="w-full md:w-80 bg-slate-950/80 border border-slate-800 p-4 rounded-2xl text-xs space-y-1">
            <span className="text-[10px] uppercase font-bold text-pink-400 flex items-center gap-1">
              <Sparkles className="w-3 h-3" /> Auto Speech-to-Text Transcript
            </span>
            <p className="text-slate-300 italic font-mono">"{result.transcript}"</p>
          </div>
        )}
      </div>

      {/* 2-Column Dashboard Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left Column: Probability Bar Chart */}
        <div className="glass-card p-6 border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white">Emotion Probability Distribution</h3>
            <span className="text-[10px] text-slate-500 uppercase font-semibold">7 Classes</span>
          </div>
          <EmotionProbabilityChart probabilities={result.probabilities} topEmotion={result.final_emotion} />
        </div>

        {/* Right Column: Individual Modality Predictions & Fusion Weights */}
        <div className="glass-card p-6 border border-slate-800 flex flex-col gap-6">
          <h3 className="text-base font-bold text-white">Individual Modality Breakdown</h3>

          <div className="grid grid-cols-1 gap-3">
            {(['vision', 'audio', 'text'] as const).map((mod) => {
              const pred = result.modality_predictions[mod];
              const isUsed = result.used_modalities.includes(mod);

              return (
                <div
                  key={mod}
                  className={`p-3.5 rounded-xl border flex items-center justify-between transition-all ${
                    isUsed
                      ? 'bg-slate-950 border-slate-800 text-slate-200'
                      : 'bg-slate-950/30 border-slate-900/50 text-slate-600 opacity-60'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xs font-bold uppercase tracking-wider text-indigo-400 w-16">
                      {mod}
                    </span>
                    {isUsed && pred ? (
                      <span className="text-xs font-semibold text-slate-200">
                        {pred.emotion} ({Math.round(pred.confidence * 100)}%)
                      </span>
                    ) : (
                      <span className="text-xs text-slate-500 italic">Not Provided (Masked)</span>
                    )}
                  </div>
                  {isUsed && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                </div>
              );
            })}
          </div>

          <div className="pt-2 border-t border-slate-800">
            <h4 className="text-xs font-bold text-slate-300 mb-2">Multimodal Fusion Weights</h4>
            <ModalityContributionChart contributions={result.explanation.modality_contributions} />
          </div>
        </div>
      </div>

      {/* Explainable AI Visualizer Section */}
      <div className="glass-card p-6 border border-slate-800 space-y-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-indigo-400" />
          Explainable AI (XAI) Visual Inspection
        </h2>
        <XaiVisualizer explanation={result.explanation} />
      </div>
    </div>
  );
};
