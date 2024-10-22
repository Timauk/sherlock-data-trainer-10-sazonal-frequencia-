import * as tf from '@tensorflow/tfjs';

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  validationSplit: number;
  earlyStoppingPatience: number;
  lstmUnits: number;
  lstmDropout: number;
}

function createModel(config: TrainingConfig): tf.LayersModel {
  const model = tf.sequential();
  
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [37] })); // Aumentado para 37 inputs
  model.add(tf.layers.lstm({ 
    units: config.lstmUnits, 
    returnSequences: false,
    dropout: config.lstmDropout 
  }));
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
  const frequencyMaps: Map<number, number>[] = [
    new Map(), // últimos 3 jogos
    new Map(), // últimos 5 jogos
    new Map(), // últimos 7 jogos
    new Map(), // últimos 10 jogos
    new Map(), // últimos 15 jogos
    new Map(), // últimos 20 jogos
    new Map(), // últimos 50 jogos
    new Map()  // últimos 100 jogos
  ];
  const sumIndices: number[] = [];
  const seasonalTrends: number[] = [];
  const dayOfWeekTrends: number[] = [];

  // Análise de frequência e soma dos números
  data.forEach((row, index) => {
    const balls = row.slice(2, 17);
    const sum = balls.reduce((a, b) => a + b, 0);
    sumIndices.push(sum);
    
    const ranges = [3, 5, 7, 10, 15, 20, 50, 100];
    ranges.forEach((range, mapIndex) => {
      if (index < range) {
        balls.forEach(ball => {
          frequencyMaps[mapIndex].set(ball, (frequencyMaps[mapIndex].get(ball) || 0) + 1);
        });
      }
    });
  });

  // Detecção de tendências sazonais e dia da semana
  data.forEach((row, index) => {
    const date = new Date(row[1]);
    const seasonIndex = Math.floor(((date.getMonth() + 1) % 12) / 3); // 0-3 para as estações
    const dayOfWeek = date.getDay(); // 0-6 para os dias da semana
    seasonalTrends.push(seasonIndex);
    dayOfWeekTrends.push(dayOfWeek);
  });

  const maxConcurso = Math.max(...data.map(row => row[0]));

  return data.map((row, index) => {
    const normalizedBalls = row.slice(2, 17).map(ball => ball / 25);
    const normalizedDate = normalizeDate(new Date(row[1]));
    const normalizedConcurso = row[0] / maxConcurso;
    const normalizedSum = sumIndices[index] / 375; // 375 é a soma máxima possível (15 * 25)
    
    const frequencyFeatures = frequencyMaps.map(map => 
      row.slice(2, 17).map(ball => (map.get(ball) || 0) / map.size)
    ).flat();

    return [
      ...normalizedBalls,
      normalizedDate,
      normalizedConcurso,
      normalizedSum,
      seasonalTrends[index] / 3,
      dayOfWeekTrends[index] / 6,
      ...frequencyFeatures
    ];
  });
}

export function normalizeData(data: number[][]): number[][] {
  return processData(data);
}

export function denormalizeData(data: number[][], maxConcurso: number): number[][] {
  return data.map(row => [
    ...row.slice(0, 15).map(n => Math.round(n * 25)),
    row[15], // Mantém a data normalizada
    Math.round(row[16] * maxConcurso), // Desnormaliza o número do concurso
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