import React, { useState, useEffect } from 'react';
import { HistoryItem } from '../types/emotion';
import { fetchHistory } from '../services/api';
import { BarChart3, Clock, Layers, Sparkles, CheckCircle2, RefreshCw } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from 'recharts';

export const AnalyticsDashboard: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const loadHistoryData = async () => {
    setIsLoading(true);
    try {
      const data = await fetchHistory();
      setHistory(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadHistoryData();
  }, []);

  // Compute metrics
  const totalCount = history.length;
  const avgConf = totalCount > 0
    ? Math.round((history.reduce((sum, h) => sum + h.confidence, 0) / totalCount) * 1000) / 10
    : 0;

  const emotionCounts: Record<string, number> = {};
  history.forEach((h) => {
    emotionCounts[h.final_emotion] = (emotionCounts[h.final_emotion] || 0) + 1;
  });

  const topEmotion = Object.keys(emotionCounts).length > 0
    ? Object.keys(emotionCounts).reduce((a, b) => (emotionCounts[a] > emotionCounts[b] ? a : b))
    : 'N/A';

  const emotionChartData = Object.entries(emotionCounts).map(([emotion, count]) => ({
    name: emotion,
    count,
  }));

  const modalityUsageCounts: Record<string, number> = { vision: 0, audio: 0, text: 0 };
  history.forEach((h) => {
    h.used_modalities.forEach((m) => {
      const k = m.toLowerCase();
      if (modalityUsageCounts[k] !== undefined) modalityUsageCounts[k]++;
    });
  });

  const modalityChartData = [
    { name: 'Vision', value: modalityUsageCounts.vision, color: '#6366f1' },
    { name: 'Audio', value: modalityUsageCounts.audio, color: '#ec4899' },
    { name: 'Text', value: modalityUsageCounts.text, color: '#14b8a6' },
  ];

  return (
    <div className="flex flex-col gap-8 py-4 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between glass-card p-6 border border-slate-800">
        <div>
          <h1 className="text-2xl font-extrabold text-white flex items-center gap-3">
            <BarChart3 className="w-6 h-6 text-indigo-400" />
            System Analytics & Storage History
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Real-time telemetry and prediction history persisted in SQLite database.
          </p>
        </div>
        <button
          onClick={loadHistoryData}
          className="flex items-center gap-2 px-3.5 py-2 bg-slate-900 hover:bg-slate-800 text-slate-200 text-xs font-semibold rounded-xl border border-slate-800 transition-all"
        >
          <RefreshCw className="w-3.5 h-3.5 text-indigo-400" />
          Refresh Data
        </button>
      </div>

      {/* 4 Key Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-card p-5 border border-slate-800 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-500">Total Inferences</span>
          <div className="text-3xl font-black text-indigo-400">{totalCount}</div>
        </div>

        <div className="glass-card p-5 border border-slate-800 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-500">Dominant Emotion</span>
          <div className="text-3xl font-black text-purple-400">{topEmotion}</div>
        </div>

        <div className="glass-card p-5 border border-slate-800 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-500">Average Confidence</span>
          <div className="text-3xl font-black text-emerald-400">{avgConf}%</div>
        </div>

        <div className="glass-card p-5 border border-slate-800 space-y-1">
          <span className="text-[10px] uppercase font-bold text-slate-500">Database Engine</span>
          <div className="text-xl font-bold text-slate-300 mt-1">SQLite 3</div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Emotion Distribution Bar Chart */}
        <div className="glass-card p-6 border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white">Detected Emotion Frequency</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={emotionChartData}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#f8fafc' }}
                />
                <Bar dataKey="count" fill="#6366f1" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Modality Usage Distribution */}
        <div className="glass-card p-6 border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white">Modality Utilization Breakdown</h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={modalityChartData} dataKey="value" cx="50%" cy="50%" outerRadius={70} label>
                  {modalityChartData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#f8fafc' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent History Table */}
      <div className="glass-card p-6 border border-slate-800 space-y-4">
        <h3 className="text-base font-bold text-white">Recent Analysis Log</h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px] tracking-wider">
                <th className="py-3 px-4">Record ID</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Final Emotion</th>
                <th className="py-3 px-4">Confidence</th>
                <th className="py-3 px-4">Modalities</th>
                <th className="py-3 px-4">Fusion Method</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {history.length > 0 ? (
                history.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-900/50 transition-all">
                    <td className="py-3 px-4 text-indigo-400 font-semibold">{item.id}</td>
                    <td className="py-3 px-4 text-slate-400">
                      {new Date(item.timestamp).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 font-bold text-white">{item.final_emotion}</td>
                    <td className="py-3 px-4 text-emerald-400 font-semibold">
                      {Math.round(item.confidence * 100)}%
                    </td>
                    <td className="py-3 px-4 text-slate-300 uppercase">
                      {item.used_modalities.join(', ')}
                    </td>
                    <td className="py-3 px-4 text-purple-300 capitalize">
                      {item.fusion_method.replace('_', ' ')}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-500 italic">
                    No analysis history records stored yet. Run an emotion analysis to view log entries!
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
