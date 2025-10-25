import { Optimizer } from './optimizer';

// Layer Normalization with complete backpropagation
export class LayerNorm {
  epsilon: number;
  gamma: number[]; // スケールパラメータ
  beta: number[];  // シフトパラメータ
  dim: number;
  learningRate: number = 0.001;

  // 勾配アキュムレータ
  gradGamma: number[];
  gradBeta: number[];

  // キャッシュ（逆伝播用）
  lastMean: number[] = [];
  lastVar: number[] = [];
  lastStd: number[] = [];
  lastNormalized: number[][] = [];
  lastInput: number[][] = [];

  constructor(dim: number, epsilon: number = 1e-8) {
    this.dim = dim;
    this.epsilon = epsilon;
    // 初期値: gamma=1, beta=0
    this.gamma = Array(dim).fill(1);
    this.beta = Array(dim).fill(0);
    // 勾配を0で初期化
    this.gradGamma = Array(dim).fill(0);
    this.gradBeta = Array(dim).fill(0);
  }

  /**
   * 順伝播: 正規化を適用
   */
  forward(x: number[]): number[] {
    // 平均と分散を計算
    const mean = x.reduce((sum, val) => sum + val, 0) / this.dim;
    const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / this.dim;
    const std = Math.sqrt(variance + this.epsilon);

    // 正規化してスケール・シフト
    const normalized = x.map((val) => (val - mean) / std);
    return normalized.map((val, i) => this.gamma[i] * val + this.beta[i]);
  }

  /**
   * バッチ処理用の順伝播
   */
  forwardBatch(batch: number[][]): number[][] {
    this.lastInput = batch.map(row => [...row]);
    this.lastMean = [];
    this.lastVar = [];
    this.lastStd = [];
    this.lastNormalized = [];

    return batch.map((x, idx) => {
      // 平均と分散を計算
      const mean = x.reduce((sum, val) => sum + val, 0) / this.dim;
      const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / this.dim;
      const std = Math.sqrt(variance + this.epsilon);

      // キャッシュに保存
      this.lastMean[idx] = mean;
      this.lastVar[idx] = variance;
      this.lastStd[idx] = std;

      // 正規化してスケール・シフト
      const normalized = x.map((val) => (val - mean) / std);
      this.lastNormalized[idx] = [...normalized];

      return normalized.map((val, i) => this.gamma[i] * val + this.beta[i]);
    });
  }

  /**
   * 完全な逆伝播
   * https://arxiv.org/pdf/1502.03167.pdf の式に基づく
   */
  backward(x: number[], gradOutput: number[], batchIdx: number): number[] {
    const N = this.dim;
    const mean = this.lastMean[batchIdx];
    const std = this.lastStd[batchIdx];
    const normalized = this.lastNormalized[batchIdx];

    // gamma と beta の勾配を累積
    for (let i = 0; i < N; i++) {
      this.gradGamma[i] += gradOutput[i] * normalized[i];
      this.gradBeta[i] += gradOutput[i];
    }

    // 入力への勾配
    // dy/dx = gamma/std * (dy_norm - mean(dy_norm) - normalized * mean(dy_norm * normalized))
    const gradNormalized = gradOutput.map((g, i) => g * this.gamma[i]);

    const meanGrad = gradNormalized.reduce((sum, val) => sum + val, 0) / N;
    const meanGradNorm = gradNormalized.reduce((sum, val, i) =>
      sum + val * normalized[i], 0) / N;

    const gradInput = gradNormalized.map((g, i) =>
      (g - meanGrad - normalized[i] * meanGradNorm) / std
    );

    return gradInput;
  }

  /**
   * 勾配をゼロにリセット
   */
  zeroGrad(): void {
    this.gradGamma.fill(0);
    this.gradBeta.fill(0);
  }

  /**
   * 累積した勾配でパラメータを更新
   */
  updateParameters(): void {
    const clipValue = 5.0;
    for (let i = 0; i < this.dim; i++) {
      const clippedGradGamma = Math.max(-clipValue, Math.min(clipValue, this.gradGamma[i]));
      const clippedGradBeta = Math.max(-clipValue, Math.min(clipValue, this.gradBeta[i]));

      this.gamma[i] += this.learningRate * clippedGradGamma;
      this.beta[i] += this.learningRate * clippedGradBeta;
    }
  }

  /**
   * Update parameters using optimizer
   */
  updateWithOptimizer(optimizer: Optimizer, layerKey: string): void {
    const clipValue = 5.0;

    // Clip gradients
    const clippedGradGamma = this.gradGamma.map(g => Math.max(-clipValue, Math.min(clipValue, g)));
    const clippedGradBeta = this.gradBeta.map(g => Math.max(-clipValue, Math.min(clipValue, g)));

    // Update gamma and beta using optimizer
    optimizer.update1D(`${layerKey}_gamma`, this.gamma, clippedGradGamma);
    optimizer.update1D(`${layerKey}_beta`, this.beta, clippedGradBeta);
  }

  /**
   * バッチ処理用の逆伝播
   */
  backwardBatch(batch: number[][], gradOutputBatch: number[][]): number[][] {
    return batch.map((x, i) => this.backward(x, gradOutputBatch[i], i));
  }
}

/**
 * ベクトルのバッチに対してLayer Normalizationを適用
 */
export function layerNormalize(
  vector: number[],
  epsilon: number = 1e-8
): { normalized: number[], mean: number, std: number } {
  const mean = vector.reduce((sum, val) => sum + val, 0) / vector.length;
  const variance = vector.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / vector.length;
  const std = Math.sqrt(variance + epsilon);

  const normalized = vector.map(val => (val - mean) / std);

  return { normalized, mean, std };
}

/**
 * バッチに対してLayer Normalizationを適用（パラメータなし版）
 */
export function batchLayerNormalize(batch: number[][], epsilon: number = 1e-8): number[][] {
  return batch.map(vector => layerNormalize(vector, epsilon).normalized);
}
