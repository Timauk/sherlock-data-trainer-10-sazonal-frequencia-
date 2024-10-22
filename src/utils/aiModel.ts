import * as tf from '@tensorflow/tfjs';

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  validationSplit: number;
  earlyStoppingPatience: number;
}

function createModel(): tf.LayersModel {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [21] })); // Aumentado para 21 inputs
  model.add(tf.layers.lstm({ units: 32, returnSequences: false }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 15, activation: 'sigmoid' }));
  
  model.compile({ 
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

async function trainModel(
  model: tf.LayersModel,
  data: number[][],
  config: TrainingConfig
): Promise<tf.History> {
  const processedData = processData(data);
  const xs = tf.tensor2d(processedData.map(row => row.slice(0, -15)));
  const ys = tf.tensor2d(processedData.map(row => row.slice(-15)));

  const history = await model.fit(xs, ys, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    validationSplit: config.validationSplit,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: config.earlyStoppingPatience }),
      // Removido o callback tensorBoard que causava o erro
    ]
  });

  xs.dispose();
  ys.dispose();

  return history;
}

function normalizeDate(date: Date): number {
  const startDate = new Date('2003-09-29');
  const timeDiff = date.getTime() - startDate.getTime();
  return timeDiff / (1000 * 60 * 60 * 24 * 365);
}

function processData(data: number[][]): number[][] {
  const frequencyMap = new Map<number, number>();
  const sumIndices: number[] = [];
  const seasonalTrends: number[] = [];

  // Análise de frequência e soma dos números
  data.forEach(row => {
    const balls = row.slice(2, 17);
    const sum = balls.reduce((a, b) => a + b, 0);
    sumIndices.push(sum);
    
    balls.forEach(ball => {
      frequencyMap.set(ball, (frequencyMap.get(ball) || 0) + 1);
    });
  });

  // Detecção de tendências sazonais (simplificada)
  for (let i = 0; i < data.length; i++) {
    const seasonIndex = (i % 4) + 1; // Assumindo 4 estações por ano
    seasonalTrends.push(seasonIndex);
  }

  return data.map((row, index) => {
    const normalizedBalls = row.slice(2, 17).map(ball => ball / 25);
    const normalizedDate = normalizeDate(new Date(row[1]));
    const normalizedConcurso = row[0] / 10000;
    const normalizedSum = sumIndices[index] / 375; // 375 é a soma máxima possível (15 * 25)
    const normalizedFrequencies = row.slice(2, 17).map(ball => (frequencyMap.get(ball) || 0) / data.length);
    
    return [
      ...normalizedBalls,
      normalizedDate,
      normalizedConcurso,
      normalizedSum,
      seasonalTrends[index] / 4,
      ...normalizedFrequencies
    ];
  });
}

export function normalizeData(data: number[][]): number[][] {
  return processData(data);
}

export function denormalizeData(data: number[][]): number[][] {
  return data.map(row => [
    ...row.slice(0, 15).map(n => Math.round(n * 25)),
    row[15], // Mantém a data normalizada
    Math.round(row[16] * 10000), // Desnormaliza o número do concurso
    // Outros campos permanecem normalizados
  ]);
}

export async function updateModel(model: tf.LayersModel, newData: number[][]): Promise<tf.LayersModel> {
  const processedData = processData(newData);
  const xs = tf.tensor2d(processedData.map(row => row.slice(0, -15)));
  const ys = tf.tensor2d(processedData.map(row => row.slice(-15)));

  await model.fit(xs, ys, {
    epochs: 1,
    batchSize: 32,
  });

  xs.dispose();
  ys.dispose();

  return model;
}

export { createModel, trainModel };