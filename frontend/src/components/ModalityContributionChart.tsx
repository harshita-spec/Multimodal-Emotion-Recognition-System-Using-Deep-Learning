import React from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts';

interface ModalityContributionChartProps {
  contributions: {
    vision: number;
    audio: number;
    text: number;
  };
}

const MODALITY_COLORS = {
  vision: '#6366f1', // Indigo
  audio: '#ec4899',  // Pink
  text: '#14b8a6'    // Teal
};

export const ModalityContributionChart: React.FC<ModalityContributionChartProps> = ({ contributions }) => {
  const data = [
    { name: 'Vision', value: contributions.vision, color: MODALITY_COLORS.vision },
    { name: 'Audio', value: contributions.audio, color: MODALITY_COLORS.audio },
    { name: 'Text', value: contributions.text, color: MODALITY_COLORS.text },
  ].filter((d) => d.value > 0);

  return (
    <div className="w-full h-56 flex flex-col items-center justify-center">
      {data.length > 0 ? (
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={45}
              outerRadius={70}
              paddingAngle={4}
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              formatter={(val: number) => [`${val}%`, 'Contribution']}
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#f8fafc' }}
            />
            <Legend verticalAlign="bottom" height={36} iconType="circle" />
          </PieChart>
        </ResponsiveContainer>
      ) : (
        <p className="text-xs text-slate-500">No contribution data</p>
      )}
    </div>
  );
};
