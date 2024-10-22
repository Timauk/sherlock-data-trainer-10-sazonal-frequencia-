import * as tf from '@tensorflow/tfjs';

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  validationSplit: number;
  earlyStoppingPatience: number;
}

export function createModel(): tf.LayersModel {
  const model = tf.sequential();
  
  // Input layer
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [17] }));
  
  // LSTM layer for capturing temporal patterns
  model.add(tf.layers.lstm({ units: 32, returnSequences: false }));
  
  // Dense layers
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  
  // Output layer
  model.add(tf.layers.dense({ units: 15, activation: 'sigmoid' }));
  
  model.compile({ 
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });
  
  return model;
}

export async function trainModel(
  model: tf.LayersModel,
  data: number[][],
  config: TrainingConfig
): Promise<tf.History> {
  const xs = tf.tensor2d(data.map(row => [
    ...row.slice(0, 15), // 15 bolas
    normalizeDate(new Date(row[15])), // Data normalizada
    row[16] / 10000 // Número do concurso normalizado
  ]));
  const ys = tf.tensor2d(data.map(row => row.slice(0, 15)));

  const history = await model.fit(xs, ys, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    validationSplit: config.validationSplit,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: config.earlyStoppingPatience }),
      tf.callbacks.tensorBoard('logs')
    ]
  });

  xs.dispose();
  ys.dispose();

  return history;
}

function normalizeDate(date: Date): number {
  const startDate = new Date('2003-09-29'); // Data do primeiro concurso
  const timeDiff = date.getTime() - startDate.getTime();
  return timeDiff / (1000 * 60 * 60 * 24 * 365); // Normaliza para anos
}

export function normalizeData(data: number[][]): number[][] {
  return data.map(row => [
    ...row.slice(0, 15).map(n => n / 25), // Normaliza as bolas
    normalizeDate(new Date(row[15])), // Normaliza a data
    row[16] / 10000 // Normaliza o número do concurso
  ]);
}

export function denormalizeData(data: number[][]): number[][] {
  return data.map(row => [
    ...row.slice(0, 15).map(n => Math.round(n * 25)),
    row[15], // Mantém a data normalizada
    Math.round(row[16] * 10000) // Desnormaliza o número do concurso
  ]);
}

export function addDerivedFeatures(data: number[][]): number[][] {
  const frequencyMap = new Map<number, number>();
  data.forEach(row => {
    row.forEach(n => {
      frequencyMap.set(n, (frequencyMap.get(n) || 0) + 1);
    });
  });

  return data.map(row => {
    const frequencies = row.map(n => frequencyMap.get(n) || 0);
    return [...row, ...frequencies];
  });
}

export async function updateModel(model: tf.LayersModel, newData: number[][]): Promise<tf.LayersModel> {
  const xs = tf.tensor2d(newData.map(row => row.slice(0, -15)));
  const ys = tf.tensor2d(newData.map(row => row.slice(-15)));

  await model.fit(xs, ys, {
    epochs: 1,
    batchSize: 32,
  });

  xs.dispose();
  ys.dispose();

  return model;
}
