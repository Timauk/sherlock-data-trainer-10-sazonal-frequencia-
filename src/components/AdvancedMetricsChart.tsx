import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface AdvancedMetricsChartProps {
  frequencyAnalysis: { range: number; frequencies: { [key: number]: number } }[];
  seasonalTrends: number[];
  dayOfWeekTrends: number[];
}

const AdvancedMetricsChart: React.FC<AdvancedMetricsChartProps> = ({ 
  frequencyAnalysis, 
  seasonalTrends, 
  dayOfWeekTrends 
}) => {
  const frequencyData = frequencyAnalysis.map(analysis => ({
    name: `Últimos ${analysis.range}`,
    ...Object.fromEntries(
      Object.entries(analysis.frequencies).map(([key, value]) => [
        `Número ${key}`,
        value
      ])
    ),
  }));

  const seasonalData = [
    { name: 'Primavera', valor: seasonalTrends[0] },
    { name: 'Verão', valor: seasonalTrends[1] },
    { name: 'Outono', valor: seasonalTrends[2] },
    { name: 'Inverno', valor: seasonalTrends[3] },
  ];

  const dayOfWeekData = [
    { name: 'Domingo', valor: dayOfWeekTrends[0] },
    { name: 'Segunda', valor: dayOfWeekTrends[1] },
    { name: 'Terça', valor: dayOfWeekTrends[2] },
    { name: 'Quarta', valor: dayOfWeekTrends[3] },
    { name: 'Quinta', valor: dayOfWeekTrends[4] },
    { name: 'Sexta', valor: dayOfWeekTrends[5] },
    { name: 'Sábado', valor: dayOfWeekTrends[6] },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold mb-2">Análise de Frequência</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={frequencyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            {Object.keys(frequencyData[0] || {}).filter(key => key !== 'name').map((key, index) => (
              <Bar key={key} dataKey={key} fill={`hsl(${index * 30}, 70%, 50%)`} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-2">Tendências Sazonais</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={seasonalData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="valor" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h3 className="text-lg font-semibold mb-2">Tendências por Dia da Semana</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={dayOfWeekData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="valor" fill="#82ca9d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AdvancedMetricsChart;