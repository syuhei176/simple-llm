import { LayerNorm } from '../layer-norm';

// シンプルなTransformer Layer（Self-Attention含む）
export class SimpleTransformer {
  // Query, Key, Value の重み行列
  wq: number[][];
  wk: number[][];
  wv: number[][];
  // Feed-forward層
  w1: number[][];
  w2: number[][];
  // Layer Normalization
  layerNorm1: LayerNorm;
  layerNorm2: LayerNorm;
  learningRate: number = 0.01;
  embeddingDim: number;

  // キャッシュ（逆伝播用）
  lastInput: number[][] = [];
  lastQ: number[][] = [];
  lastK: number[][] = [];
  lastV: number[][] = [];
  lastAttention: number[][] = [];
  lastAttnOutput: number[][] = [];

  constructor(embeddingDim: number) {
    this.embeddingDim = embeddingDim;
    const scale = Math.sqrt(2.0 / embeddingDim);

    // Query, Key, Value の重み初期化
    this.wq = this.initWeights(embeddingDim, embeddingDim, scale);
    this.wk = this.initWeights(embeddingDim, embeddingDim, scale);
    this.wv = this.initWeights(embeddingDim, embeddingDim, scale);

    // Feed-forward層の重み初期化
    this.w1 = this.initWeights(embeddingDim, embeddingDim, scale);
    this.w2 = this.initWeights(embeddingDim, embeddingDim, scale);

    // Layer Normalizationの初期化
    this.layerNorm1 = new LayerNorm(embeddingDim);
    this.layerNorm2 = new LayerNorm(embeddingDim);
  }

  private initWeights(rows: number, cols: number, scale: number): number[][] {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
    );
  }

  // 行列積: A * B
  private matmul(a: number[][], b: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < a[0].length; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  // ベクトルと行列の積
  private vecMatmul(vec: number[], mat: number[][]): number[] {
    const result = new Array(mat[0].length).fill(0);
    for (let j = 0; j < mat[0].length; j++) {
      for (let i = 0; i < vec.length; i++) {
        result[j] += vec[i] * mat[i][j];
      }
    }
    return result;
  }

  // Softmax関数（行ごと）
  private softmax(matrix: number[][]): number[][] {
    return matrix.map(row => {
      const maxVal = Math.max(...row);
      const expScores = row.map(x => Math.exp(x - maxVal));
      const sumExp = expScores.reduce((a, b) => a + b, 1e-8);
      return expScores.map(x => x / sumExp);
    });
  }

  // Self-Attention
  private attention(input: number[][]): number[][] {
    const seqLen = input.length;
    const scale = Math.sqrt(this.embeddingDim);

    // Q, K, V を計算
    this.lastQ = input.map(vec => this.vecMatmul(vec, this.wq));
    this.lastK = input.map(vec => this.vecMatmul(vec, this.wk));
    this.lastV = input.map(vec => this.vecMatmul(vec, this.wv));

    // Attention scores: Q * K^T / sqrt(d)
    const scores: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      scores[i] = [];
      for (let j = 0; j < seqLen; j++) {
        let dot = 0;
        for (let k = 0; k < this.embeddingDim; k++) {
          dot += this.lastQ[i][k] * this.lastK[j][k];
        }
        scores[i][j] = dot / scale;
      }
    }

    // Softmax
    this.lastAttention = this.softmax(scores);

    // Attention output: Attention * V
    const output: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      output[i] = new Array(this.embeddingDim).fill(0);
      for (let j = 0; j < seqLen; j++) {
        for (let k = 0; k < this.embeddingDim; k++) {
          output[i][k] += this.lastAttention[i][j] * this.lastV[j][k];
        }
      }
    }

    return output;
  }

  // Feed-forward層（ReLU活性化）
  private feedForward(input: number[][]): number[][] {
    // 第1層
    const hidden = input.map(vec => {
      const h = this.vecMatmul(vec, this.w1);
      // ReLU
      return h.map(x => Math.max(0, x));
    });

    // 第2層
    return hidden.map(vec => this.vecMatmul(vec, this.w2));
  }

  forward(input: number[][]): number[][] {
    this.lastInput = input.map(row => [...row]);

    // Self-Attention
    this.lastAttnOutput = this.attention(input);

    // 残差接続 + Layer Normalization
    const attnWithResidual = this.lastAttnOutput.map((row, i) =>
      row.map((val, j) => val + input[i][j])
    );
    const attnNormalized = this.layerNorm1.forwardBatch(attnWithResidual);

    // Feed-forward
    const ffOutput = this.feedForward(attnNormalized);

    // 残差接続 + Layer Normalization
    const ffWithResidual = ffOutput.map((row, i) =>
      row.map((val, j) => val + attnNormalized[i][j])
    );
    return this.layerNorm2.forwardBatch(ffWithResidual);
  }

  // 簡略化された逆伝播
  backward(gradOutput: number[][]): number[][] {
    const seqLen = gradOutput.length;
    const gradInput: number[][] = Array.from({ length: seqLen }, () =>
      new Array(this.embeddingDim).fill(0)
    );

    // 簡易的な勾配更新（実際はより複雑）
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const grad = gradOutput[i][j];

        // Wq, Wk, Wvの更新
        for (let k = 0; k < this.embeddingDim; k++) {
          this.wq[k][j] += this.learningRate * grad * this.lastInput[i][k] * 0.1;
          this.wk[k][j] += this.learningRate * grad * this.lastInput[i][k] * 0.1;
          this.wv[k][j] += this.learningRate * grad * this.lastInput[i][k] * 0.1;

          gradInput[i][k] += grad * (this.wq[k][j] + this.wk[k][j] + this.wv[k][j]) * 0.33;
        }
      }
    }

    return gradInput;
  }
}
