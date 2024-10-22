import React from 'react';
import DataUploader from '@/components/DataUploader';
import GameControls from '@/components/GameControls';
import GameBoard from '@/components/GameBoard';
import EnhancedLogDisplay from '@/components/EnhancedLogDisplay';
import NeuralNetworkVisualization from '@/components/NeuralNetworkVisualization';
import ModelMetrics from '@/components/ModelMetrics';
import AdvancedMetricsChart from '@/components/AdvancedMetricsChart';
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { useGameLogic } from '@/hooks/useGameLogic';

interface PlayPageContentProps {
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
  onReset: () => void;
  onThemeToggle: () => void;
  onCsvUpload: (file: File) => void;
  onModelUpload: (jsonFile: File, weightsFile: File) => void;
  onSaveModel: () => void;
  progress: number;
  generation: number;
  gameLogic: ReturnType<typeof useGameLogic>;
}

export const PlayPageContent: React.FC<PlayPageContentProps> = ({
  isPlaying,
  onPlay,
  onPause,
  onReset,
  onThemeToggle,
  onCsvUpload,
  onModelUpload,
  onSaveModel,
  progress,
  generation,
  gameLogic
}) => {
  return (
    <div className="flex flex-wrap gap-4">
      <div className="flex-1">
        <DataUploader onCsvUpload={onCsvUpload} onModelUpload={onModelUpload} onSaveModel={onSaveModel} />

        <GameControls
          isPlaying={isPlaying}
          onPlay={onPlay}
          onPause={onPause}
          onReset={onReset}
          onThemeToggle={onThemeToggle}
        />

        <Button onClick={gameLogic.toggleInfiniteMode} className="mt-2">
          {gameLogic.isInfiniteMode ? 'Desativar' : 'Ativar'} Modo Infinito
        </Button>

        <div className="mb-4">
          <h3 className="text-lg font-semibold mb-2">Progresso da Geração {generation}</h3>
          <Progress value={progress} className="w-full" />
        </div>

        <ModelMetrics
          accuracy={gameLogic.modelMetrics.accuracy}
          randomAccuracy={gameLogic.modelMetrics.randomAccuracy}
          totalPredictions={gameLogic.modelMetrics.totalPredictions}
        />

        <GameBoard
          boardNumbers={gameLogic.boardNumbers}
          concursoNumber={gameLogic.concursoNumber}
          players={gameLogic.players}
          evolutionData={gameLogic.evolutionData}
        />
        
        <EnhancedLogDisplay logs={gameLogic.logs} />
      </div>

      <div className="flex-1">
        <NeuralNetworkVisualization 
          layers={[37, 64, 32, 15]} 
          inputData={gameLogic.neuralNetworkVisualization?.input}
          outputData={gameLogic.neuralNetworkVisualization?.output}
        />

        <AdvancedMetricsChart
          frequencyAnalysis={gameLogic.frequencyAnalysis}
          seasonalTrends={gameLogic.seasonalTrends}
          dayOfWeekTrends={gameLogic.dayOfWeekTrends}
        />
      </div>
    </div>
  );
};