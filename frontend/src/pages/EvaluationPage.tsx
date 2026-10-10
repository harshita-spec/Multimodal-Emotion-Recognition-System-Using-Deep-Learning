import React, { useState, useEffect } from 'react';
import { EvaluationMetrics } from '../types/emotion';
import { fetchEvaluationMetrics } from '../services/api';
import { Activity, ShieldCheck, BarChart3, Layers, CheckCircle2 } from 'lucide-react';

export const EvaluationPage: React.FC = () => {
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);

  useEffect(() => {
    fetchEvaluationMetrics().then(setMetrics).catch(console.error);
  }, []);

  if (!metrics) {
    return (
      <div className="flex items-center justify-center py-16 text-slate-400 text-sm">
        Loading evaluation benchmark data...
      </div>
    );
  }

  const labels = metrics.confusion_matrix.labels;
  const cm = metrics.confusion_matrix.matrix;

  return (
    <div className="flex flex-col gap-8 py-4 max-w-6xl mx-auto">
      {/* Header Banner */}
      <div className="glass-card p-6 border border-slate-800 space-y-2">
        <h1 className="text-2xl font-extrabold text-white flex items-center gap-3">
          <Activity className="w-6 h-6 text-emerald-400" />
          Model Evaluation & Benchmark Comparison
        </h1>
        <p className="text-xs text-slate-400">
          Empirical evaluation results across individual modalities, late fusion baseline, feature concatenation, and proposed attention fusion.
        </p>
      </div>

      {/* Benchmark Metric Comparison Table */}
      <div className="glass-card p-6 border border-slate-800 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-indigo-400" />
          Modality & Architecture Performance Metrics
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] tracking-wider">
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Precision</th>
                <th className="py-3 px-4">Recall</th>
                <th className="py-3 px-4">Macro F1</th>
                <th className="py-3 px-4">Weighted F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {Object.entries(metrics.modalities).map(([name, m]) => {
                const isProposed = name.includes('Attention Fusion');

                return (
                  <tr
                    key={name}
                    className={`transition-all ${
                      isProposed ? 'bg-indigo-950/40 font-bold border-l-4 border-l-indigo-500' : 'hover:bg-slate-900/50'
                    }`}
                  >
                    <td className="py-3.5 px-4 text-white flex items-center gap-2">
                      {name}
                      {isProposed && (
                        <span className="bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 text-[9px] px-2 py-0.5 rounded-full uppercase tracking-wider font-semibold">
                          Proposed
                        </span>
                      )}
                    </td>
                    <td className="py-3.5 px-4 text-indigo-300">{(m.accuracy * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 text-slate-300">{(m.precision * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 text-slate-300">{(m.recall * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 text-slate-300">{(m.macro_f1 * 100).toFixed(1)}%</td>
                    <td className="py-3.5 px-4 text-emerald-400 font-extrabold">
                      {(m.weighted_f1 * 100).toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Confusion Matrix Grid */}
      <div className="glass-card p-6 border border-slate-800 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-purple-400" />
          Attention Fusion Confusion Matrix (7 Emotion Classes)
        </h3>

        <div className="overflow-x-auto p-4 bg-slate-950 rounded-2xl border border-slate-800">
          <table className="mx-auto text-center border-collapse text-xs font-mono">
            <thead>
              <tr>
                <th className="p-2 text-slate-500 text-[10px] uppercase">True \ Pred</th>
                {labels.map((lbl) => (
                  <th key={lbl} className="p-2.5 text-indigo-300 font-bold">
                    {lbl}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.map((row, i) => (
                <tr key={i} className="border-t border-slate-900">
                  <td className="p-2.5 text-indigo-300 font-bold text-left">{labels[i]}</td>
                  {row.map((val, j) => {
                    const isDiagonal = i === j;
                    const opacity = Math.min(val / 150, 1.0);

                    return (
                      <td
                        key={j}
                        className={`p-3 rounded-lg border border-slate-900 font-bold ${
                          isDiagonal
                            ? 'bg-indigo-600/30 text-indigo-200 border-indigo-500/40 shadow-inner'
                            : val > 5
                            ? 'bg-rose-950/40 text-rose-300'
                            : 'text-slate-500'
                        }`}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
