import { useState, useCallback, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { updateModel, normalizeData } from '../utils/aiModel';

interface Player {
  id: number;
  score: number;
  predictions: number[];
  weights: number[];
}

interface ModelMetrics {
  accuracy: number;
  randomAccuracy: number;
  totalPredictions: number;
}

export const useGameLogic = (csvData: number[][], trainedModel: tf.LayersModel | null) => {
  const [players, setPlayers] = useState<Player[]>([]);
  const [generation, setGeneration] = useState(1);
  const [evolutionData, setEvolutionData] = useState<any[]>([]);
  const [boardNumbers, setBoardNumbers] = useState<number[]>([]);
  const [concursoNumber, setConcursoNumber] = useState(0);
  const [isInfiniteMode, setIsInfiniteMode] = useState(false);
  const [neuralNetworkVisualization, setNeuralNetworkVisualization] = useState<{ input: number[], output: number[], weights: number[][] } | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics>({
    accuracy: 0,
    randomAccuracy: 0,
    totalPredictions: 0,
  });
  const [logs, setLogs] = useState<{ message: string; matches?: number }[]>([]);

  const [trainingData, setTrainingData] = useState<number[][]>([]);
  const [updateInterval, setUpdateInterval] = useState(10);
  const [maxConcurso, setMaxConcurso] = useState(0);

  const addLog = useCallback((message: string, matches?: number) => {
    setLogs(prevLogs => [...prevLogs, { message, matches }]);
  }, []);

  const initializePlayers = useCallback(() => {
    const newPlayers = Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      score: 0,
      predictions: [],
      weights: Array.from({ length: 22 }, () => Math.floor(Math.random() * 1001))
    }));
    setPlayers(newPlayers);
  }, []);

  useEffect(() => {
    initializePlayers();
    if (csvData.length > 0) {
      setMaxConcurso(Math.max(...csvData.map(row => row[0])));
    }
  }, [initializePlayers, csvData]);

  const makePrediction = (inputData: number[], playerWeights: number[]): number[] => {
    if (!trainedModel) return [];
    
    const normalizedData = normalizeData([inputData])[0];
    const weightedInput = normalizedData.map((value, index) => value * (playerWeights[index] / 1000));
    const inputTensor = tf.tensor2d([weightedInput], [1, 22]);
    
    const predictions = trainedModel.predict(inputTensor) as tf.Tensor;
    const result = Array.from(predictions.dataSync());
    
    inputTensor.dispose();
    predictions.dispose();
    
    setNeuralNetworkVisualization({ input: weightedInput, output: result, weights: trainedModel.getWeights().map(w => Array.from(w.dataSync())) });
    
    // Ensure 15 unique numbers
    const uniqueNumbers = new Set<number>();
    while (uniqueNumbers.size < 15) {
      const num = Math.floor(Math.random() * 25) + 1;
      uniqueNumbers.add(num);
    }
    return Array.from(uniqueNumbers);
  };

  const gameLoop = useCallback(() => {
    if (csvData.length === 0 || !trainedModel) return;

    const currentBoardNumbers = csvData[concursoNumber % csvData.length];
    setBoardNumbers(currentBoardNumbers);

    let totalMatches = 0;
    let totalRandomMatches = 0;

    const updatedPlayers = players.map(player => {
      const playerPredictions = makePrediction(currentBoardNumbers, player.weights);
      const matches = playerPredictions.filter(num => currentBoardNumbers.includes(num)).length;
      const randomPredictions = Array.from({ length: 15 }, () => Math.floor(Math.random() * 25) + 1);
      const randomMatches = randomPredictions.filter(num => currentBoardNumbers.includes(num)).length;
      
      totalMatches += matches;
      totalRandomMatches += randomMatches;

      const reward = calculateDynamicReward(matches);

      if (matches >= 13) {
        addLog(`Jogador ${player.id} acertou ${matches} números!`, matches);
      }

      return {
        ...player,
        score: player.score + reward,
        predictions: playerPredictions
      };
    });

    setPlayers(updatedPlayers);
    setEvolutionData(prev => [
      ...prev,
      ...updatedPlayers.map(player => ({
        generation,
        playerId: player.id,
        score: player.score
      }))
    ]);

    // Collect training data
    setTrainingData(prev => [...prev, [...currentBoardNumbers, ...updatedPlayers[0].predictions]]);

    // Update model metrics
    setModelMetrics(prev => ({
      accuracy: (prev.accuracy * prev.totalPredictions + totalMatches / (15 * players.length)) / (prev.totalPredictions + 1),
      randomAccuracy: (prev.randomAccuracy * prev.totalPredictions + totalRandomMatches / (15 * players.length)) / (prev.totalPredictions + 1),
      totalPredictions: prev.totalPredictions + 1,
    }));

    setConcursoNumber(prev => prev + 1);

    // Update model if necessary
    if (concursoNumber % updateInterval === 0 && trainingData.length > 0) {
      updateModelWithNewData();
    }
  }, [players, csvData, concursoNumber, generation, trainedModel, addLog, updateInterval, trainingData]);

  const updateModelWithNewData = useCallback(async () => {
    if (!trainedModel || trainingData.length === 0) return;

    try {
      const updatedModel = await updateModel(trainedModel, trainingData);
      // We can't directly set the trainedModel state here as it's passed as a prop
      // Instead, we'll need to handle this update in the parent component
      addLog(`Modelo atualizado com ${trainingData.length} novos registros.`);
      setTrainingData([]); // Clear training data after update
    } catch (error) {
      addLog(`Erro ao atualizar o modelo: ${error instanceof Error ? error.message : 'Erro desconhecido'}`);
    }
  }, [trainedModel, trainingData, addLog]);

  useEffect(() => {
    // Set update interval based on CSV data length
    setUpdateInterval(Math.max(10, Math.floor(csvData.length / 10)));
  }, [csvData]);

  const evolveGeneration = useCallback(() => {
    setGeneration(prev => prev + 1);
    // Implement evolution logic here if needed
    addLog(`Geração ${generation} concluída. Iniciando geração ${generation + 1}.`);
  }, [generation, addLog]);

  const calculateDynamicReward = (matches: number): number => {
    return matches > 12 ? Math.pow(2, matches - 12) : -Math.pow(2, 12 - matches);
  };

  const toggleInfiniteMode = useCallback(() => {
    setIsInfiniteMode(prev => !prev);
    addLog(`Modo infinito ${isInfiniteMode ? 'desativado' : 'ativado'}.`);
  }, [isInfiniteMode, addLog]);

  return {
    players,
    generation,
    evolutionData,
    boardNumbers,
    concursoNumber,
    isInfiniteMode,
    setIsInfiniteMode,
    initializePlayers,
    gameLoop,
    evolveGeneration,
    neuralNetworkVisualization,
    modelMetrics,
    logs,
    addLog,
    toggleInfiniteMode,
    maxConcurso
  };
};