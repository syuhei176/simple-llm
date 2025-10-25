import { Optimizer } from '../optimizer';

export type InitializationType = 'xavier' | 'he' | 'normal';

export class EmbeddingLayer {
  weights: number[][];
  embeddingDim: number;
  vocabSize: number;
  learningRate: number = 0.001; // 勾配爆発防止のため学習率を下げる

  // Gradient accumulator
  gradWeights: Map<number, number[]> = new Map();

  constructor(
    vocabSize: number,
    embeddingDim: number,
    initialization: InitializationType = 'xavier'
  ) {
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;

    // 初期化方法の選択
    let scale: number;
    switch (initialization) {
      case 'xavier':
        // Xavier/Glorot initialization - 一般的なニューラルネットワークに適している
        scale = Math.sqrt(1.0 / embeddingDim);
        break;
      case 'he':
        // He initialization - ReLU活性化関数に最適
        scale = Math.sqrt(2.0 / embeddingDim);
        break;
      case 'normal':
        // 小さな正規分布
        scale = 0.01;
        break;
      default:
        scale = Math.sqrt(1.0 / embeddingDim);
    }

    // メモリ効率のため、一度に全ての重みを初期化
    this.weights = Array.from({ length: vocabSize }, () =>
      Array.from({ length: embeddingDim }, () => (Math.random() * 2 - 1) * scale)
    );
  }

  forward(index: number): number[] {
    if (index < 0 || index >= this.weights.length) {
      return Array(this.embeddingDim).fill(0);
    }
    return [...this.weights[index]]; // コピーを返す
  }

  // 勾配を受け取って蓄積
  backward(index: number, gradient: number[]): void {
    if (index < 0 || index >= this.weights.length) {
      return;
    }

    // Accumulate gradients
    if (!this.gradWeights.has(index)) {
      this.gradWeights.set(index, new Array(this.embeddingDim).fill(0));
    }
    const grad = this.gradWeights.get(index)!;
    for (let i = 0; i < this.embeddingDim; i++) {
      grad[i] += gradient[i];
    }
  }

  // Manual update (for backward compatibility)
  updateParametersManual(): void {
    const clipValue = 5.0;
    for (const [index, grad] of this.gradWeights.entries()) {
      for (let i = 0; i < this.embeddingDim; i++) {
        const clippedGrad = Math.max(-clipValue, Math.min(clipValue, grad[i]));
        this.weights[index][i] += this.learningRate * clippedGrad;
      }
    }
    this.gradWeights.clear();
  }

  // Update using optimizer
  updateWithOptimizer(optimizer: Optimizer): void {
    const clipValue = 5.0;
    for (const [index, grad] of this.gradWeights.entries()) {
      // Clip gradients
      const clippedGrad = grad.map(g => Math.max(-clipValue, Math.min(clipValue, g)));

      // Create a 2D array view for the optimizer
      const paramView = [this.weights[index]];
      const gradView = [clippedGrad];

      // Update using optimizer
      optimizer.update(`embedding_${index}`, paramView, gradView);
    }
    this.gradWeights.clear();
  }

  // Zero gradients
  zeroGrad(): void {
    this.gradWeights.clear();
  }
}