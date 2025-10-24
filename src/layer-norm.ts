// Layer Normalization
// 各層の出力を正規化して学習を安定化させる

export class LayerNorm {
  epsilon: number;
  gamma: number[]; // スケールパラメータ
  beta: number[];  // シフトパラメータ
  dim: number;
  learningRate: number = 0.01; // 勾配爆発防止のため学習率を下げる

  constructor(dim: number, epsilon: number = 1e-8) {
    this.dim = dim;
    this.epsilon = epsilon;
    // 初期値: gamma=1, beta=0
    this.gamma = Array(dim).fill(1);
    this.beta = Array(dim).fill(0);
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
    return x.map((val, i) => {
      const normalized = (val - mean) / std;
      return this.gamma[i] * normalized + this.beta[i];
    });
  }

  /**
   * バッチ処理用の順伝播
   */
  forwardBatch(batch: number[][]): number[][] {
    return batch.map(x => this.forward(x));
  }

  /**
   * 逆伝播（簡略版）
   */
  backward(x: number[], gradOutput: number[]): number[] {
    // 簡略化された逆伝播
    // 実際にはより複雑な計算が必要だが、教育目的のため簡略化

    const mean = x.reduce((sum, val) => sum + val, 0) / this.dim;
    const variance = x.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / this.dim;
    const std = Math.sqrt(variance + this.epsilon);

    // パラメータの更新
    for (let i = 0; i < this.dim; i++) {
      const normalized = (x[i] - mean) / std;

      // gammaとbetaの勾配
      const gradGamma = gradOutput[i] * normalized;
      const gradBeta = gradOutput[i];

      // パラメータ更新
      this.gamma[i] += this.learningRate * gradGamma;
      this.beta[i] += this.learningRate * gradBeta;
    }

    // 入力への勾配（簡略版）
    return gradOutput.map((grad, i) => grad * this.gamma[i] / std);
  }

  /**
   * バッチ処理用の逆伝播
   */
  backwardBatch(batch: number[][], gradOutputBatch: number[][]): number[][] {
    return batch.map((x, i) => this.backward(x, gradOutputBatch[i]));
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
