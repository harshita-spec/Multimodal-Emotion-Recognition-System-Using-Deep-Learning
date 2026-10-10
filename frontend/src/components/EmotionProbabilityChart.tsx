import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';
import { EmotionClass } from '../types/emotion';

interface EmotionProbabilityChartProps {
  probabilities: Record<EmotionClass, number>;
  topEmotion: EmotionClass;
}

const EMOTION_COLORS: Record<EmotionClass, string> = {
  Happy: '#10b981',    // Emerald
  Sad: '#3b82f6',      // Blue
  Angry: '#ef4444',    // Red
  Fear: '#a855f7',     // Purple
  Surprise: '#f59e0b', // Amber
  Disgust: '#84cc16',  // Lime
  Neutral: '#64748b'   // Slate
};

export const EmotionProbabilityChart: React.FC<EmotionProbabilityChartProps> = ({
  probabilities,
  topEmotion
}) => {
  const data = Object.entries(probabilities).map(([emotion, prob]) => ({
    name: emotion,
    value: Math.round(prob * 1000) / 10,
    color: EMOTION_COLORS[emotion as EmotionClass] || '#6366f1'
  })).sort((a, b) => b.value - a.value);

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <XAxis type="number" domain={[0, 100]} unit="%" stroke="#64748b" fontSize={11} />
          <YAxis dataKey="name" type="category" stroke="#cbd5e1" fontSize={12} width={70} />
          <Tooltip
            formatter={(val: number) => [`${val}%`, 'Probability']}
            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#f8fafc' }}
          />
          <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={16}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.name === topEmotion ? entry.color : `${entry.color}88`}
                stroke={entry.name === topEmotion ? '#ffffff' : undefined}
                strokeWidth={entry.name === topEmotion ? 1.5 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
