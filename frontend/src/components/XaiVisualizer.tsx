import React from 'react';
import { Eye, Flame, FileText, Music } from 'lucide-react';
import { XaiExplanation } from '../types/emotion';

interface XaiVisualizerProps {
  explanation: XaiExplanation;
}

export const XaiVisualizer: React.FC<XaiVisualizerProps> = ({ explanation }) => {
  return (
    <div className="flex flex-col gap-6 w-full">
      {/* 1. Vision Grad-CAM Explanation */}
      {explanation.vision_gradcam_base64 && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold text-xs uppercase tracking-wider">
            <Flame className="w-4 h-4 text-amber-400" />
            Vision Explainability — Grad-CAM Heatmap
          </div>
          <div className="flex flex-col md:flex-row items-center gap-4">
            <div className="w-44 h-44 rounded-xl overflow-hidden border border-slate-700 bg-slate-950 flex-shrink-0">
              <img
                src={explanation.vision_gradcam_base64}
                alt="Grad-CAM Heatmap"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="text-xs text-slate-300 space-y-2">
              <p>
                <strong className="text-indigo-300">ResNet50 Layer4 Activation:</strong> The Grad-CAM heatmap highlights the facial regions (eyes, mouth contour, eyebrows) that most strongly influenced the deep CNN prediction score.
              </p>
              <div className="flex items-center gap-2 pt-1">
                <span className="w-3 h-3 rounded-full bg-red-500" /> High Salience (Key Facial Feature)
                <span className="w-3 h-3 rounded-full bg-blue-500 ml-2" /> Low Salience (Background)
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 2. Audio Mel-Spectrogram Explanation */}
      {explanation.audio_spectrogram_base64 && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-pink-400 font-semibold text-xs uppercase tracking-wider">
            <Music className="w-4 h-4 text-pink-400" />
            Audio Explainability — Mel-Spectrogram Pitch/Formant Region
          </div>
          <div className="flex flex-col md:flex-row items-center gap-4">
            <div className="w-full md:w-80 h-40 rounded-xl overflow-hidden border border-slate-700 bg-slate-950">
              <img
                src={explanation.audio_spectrogram_base64}
                alt="Mel-Spectrogram XAI"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="text-xs text-slate-300 space-y-2">
              <p>
                <strong className="text-pink-300">Time-Frequency Saliency:</strong> Mel-Spectrogram energy distribution highlights high-amplitude vocal harmonics, fundamental frequency ($F_0$), and spectral centroids corresponding to pitch variation.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* 3. Text Token Attribution Saliency */}
      {explanation.text_token_attributions && explanation.text_token_attributions.length > 0 && (
        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-teal-400 font-semibold text-xs uppercase tracking-wider">
            <FileText className="w-4 h-4 text-teal-400" />
            Text Explainability — Token Importance Attribution
          </div>
          <p className="text-xs text-slate-400">
            Higher intensity highlights denote tokens contributing strongly to the classified emotion state:
          </p>
          <div className="flex flex-wrap gap-2 p-3 bg-slate-950 border border-slate-800 rounded-xl font-mono text-sm">
            {explanation.text_token_attributions.map((item, idx) => {
              const score = item.score;
              const isHigh = score > 0.5;
              const isMid = score > 0.2;
              
              const bgClass = isHigh
                ? 'bg-emerald-500/30 text-emerald-200 border-emerald-500/50'
                : isMid
                ? 'bg-amber-500/20 text-amber-200 border-amber-500/40'
                : 'bg-slate-800/60 text-slate-400 border-slate-700/50';

              return (
                <span
                  key={idx}
                  className={`px-2.5 py-1 rounded-md border text-xs font-medium transition-all ${bgClass}`}
                  title={`Token Score: ${score}`}
                >
                  {item.word}
                  <span className="ml-1 text-[10px] opacity-70">({score.toFixed(2)})</span>
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* 4. Multimodal Fusion Contribution Percentage */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 flex flex-col gap-3">
        <div className="flex items-center gap-2 text-indigo-300 font-semibold text-xs uppercase tracking-wider">
          <Eye className="w-4 h-4 text-indigo-400" />
          Attention-Based Fusion Modality Contribution Breakdown
        </div>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="bg-slate-950 border border-indigo-500/30 p-3 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-indigo-400 tracking-wider">Vision</span>
            <div className="text-xl font-extrabold text-indigo-300 mt-1">
              {explanation.modality_contributions.vision}%
            </div>
          </div>
          <div className="bg-slate-950 border border-pink-500/30 p-3 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-pink-400 tracking-wider">Audio</span>
            <div className="text-xl font-extrabold text-pink-300 mt-1">
              {explanation.modality_contributions.audio}%
            </div>
          </div>
          <div className="bg-slate-950 border border-teal-500/30 p-3 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-teal-400 tracking-wider">Text</span>
            <div className="text-xl font-extrabold text-teal-300 mt-1">
              {explanation.modality_contributions.text}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
