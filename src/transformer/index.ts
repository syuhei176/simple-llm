import { LayerNorm } from '../layer-norm';

// 完全な逆伝播を実装したTransformer Layer
export class SimpleTransformer {
  // Query, Key, Value の重み行列
  wq: number[][];
  wk: number[][];
  wv: number[][];
  // Feed-forward層
  w1: number[][];
  w2: number[][];
  // 勾配アキュムレータ
  gradWq: number[][];
  gradWk: number[][];
  gradWv: number[][];
  gradW1: number[][];
  gradW2: number[][];
  // Layer Normalization
  layerNorm1: LayerNorm;
  layerNorm2: LayerNorm;
  learningRate: number = 0.001;
  embeddingDim: number;

  // キャッシュ（逆伝播用）
  lastInput: number[][] = [];
  lastQ: number[][] = [];
  lastK: number[][] = [];
  lastV: number[][] = [];
  lastScores: number[][] = [];
  lastAttentionWeights: number[][] = [];
  lastAttnOutput: number[][] = [];
  lastAttnResidual: number[][] = [];
  lastAttnNorm: number[][] = [];
  lastFF1: number[][] = [];
  lastFFOutput: number[][] = [];
  lastFFResidual: number[][] = [];

  constructor(embeddingDim: number) {
    this.embeddingDim = embeddingDim;
    // Xavier初期化
    const scale = Math.sqrt(1.0 / embeddingDim);

    // Query, Key, Value の重み初期化
    this.wq = this.initWeights(embeddingDim, embeddingDim, scale);
    this.wk = this.initWeights(embeddingDim, embeddingDim, scale);
    this.wv = this.initWeights(embeddingDim, embeddingDim, scale);

    // Feed-forward層の重み初期化
    this.w1 = this.initWeights(embeddingDim, embeddingDim, scale);
    this.w2 = this.initWeights(embeddingDim, embeddingDim, scale);

    // 勾配を0で初期化
    this.gradWq = this.initWeights(embeddingDim, embeddingDim, 0);
    this.gradWk = this.initWeights(embeddingDim, embeddingDim, 0);
    this.gradWv = this.initWeights(embeddingDim, embeddingDim, 0);
    this.gradW1 = this.initWeights(embeddingDim, embeddingDim, 0);
    this.gradW2 = this.initWeights(embeddingDim, embeddingDim, 0);

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

  // 行列の転置
  private transpose(mat: number[][]): number[][] {
    const rows = mat.length;
    const cols = mat[0].length;
    const result: number[][] = Array.from({ length: cols }, () => new Array(rows));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j][i] = mat[i][j];
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

  // Softmaxの逆伝播
  private softmaxBackward(gradOutput: number[][], softmaxOutput: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < gradOutput.length; i++) {
      const s = softmaxOutput[i];
      const grad = gradOutput[i];

      // Jacobian行列を使った逆伝播
      const sum = grad.reduce((acc, g, j) => acc + g * s[j], 0);
      result[i] = s.map((si, j) => si * (grad[j] - sum));
    }
    return result;
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
    this.lastScores = [];
    for (let i = 0; i < seqLen; i++) {
      this.lastScores[i] = [];
      for (let j = 0; j < seqLen; j++) {
        let dot = 0;
        for (let k = 0; k < this.embeddingDim; k++) {
          dot += this.lastQ[i][k] * this.lastK[j][k];
        }
        this.lastScores[i][j] = dot / scale;
      }
    }

    // Softmax
    this.lastAttentionWeights = this.softmax(this.lastScores);

    // Attention output: Attention * V
    const output: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      output[i] = new Array(this.embeddingDim).fill(0);
      for (let j = 0; j < seqLen; j++) {
        for (let k = 0; k < this.embeddingDim; k++) {
          output[i][k] += this.lastAttentionWeights[i][j] * this.lastV[j][k];
        }
      }
    }

    return output;
  }

  // Feed-forward層（ReLU活性化）
  private feedForward(input: number[][]): number[][] {
    // 第1層
    this.lastFF1 = input.map(vec => {
      const h = this.vecMatmul(vec, this.w1);
      // ReLU
      return h.map(x => Math.max(0, x));
    });

    // 第2層
    return this.lastFF1.map(vec => this.vecMatmul(vec, this.w2));
  }

  forward(input: number[][]): number[][] {
    this.lastInput = input.map(row => [...row]);

    // Self-Attention
    this.lastAttnOutput = this.attention(input);

    // 残差接続
    this.lastAttnResidual = this.lastAttnOutput.map((row, i) =>
      row.map((val, j) => val + input[i][j])
    );

    // Layer Normalization
    this.lastAttnNorm = this.layerNorm1.forwardBatch(this.lastAttnResidual);

    // Feed-forward
    this.lastFFOutput = this.feedForward(this.lastAttnNorm);

    // 残差接続
    this.lastFFResidual = this.lastFFOutput.map((row, i) =>
      row.map((val, j) => val + this.lastAttnNorm[i][j])
    );

    // Layer Normalization
    return this.layerNorm2.forwardBatch(this.lastFFResidual);
  }

  // 完全な逆伝播
  backward(gradOutput: number[][]): number[][] {
    const seqLen = gradOutput.length;
    const scale = Math.sqrt(this.embeddingDim);

    // Layer Norm 2 の逆伝播
    let grad = this.layerNorm2.backwardBatch(this.lastFFResidual, gradOutput);

    // 残差接続の逆伝播（勾配を2つに分配）
    const gradFFOutput = grad.map(row => [...row]);
    const gradAttnNorm1 = grad.map(row => [...row]);

    // Feed-forward 第2層の逆伝播
    const gradFF1: number[][] = Array.from({ length: seqLen }, () =>
      new Array(this.embeddingDim).fill(0)
    );

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const g = gradFFOutput[i][j];

        for (let k = 0; k < this.embeddingDim; k++) {
          // 勾配を累積
          this.gradW2[k][j] += g * this.lastFF1[i][k];
          gradFF1[i][k] += g * this.w2[k][j];
        }
      }
    }

    // ReLUの逆伝播
    const gradAttnNorm2: number[][] = gradFF1.map((row, i) =>
      row.map((val, j) => this.lastFF1[i][j] > 0 ? val : 0)
    );

    // Feed-forward 第1層の逆伝播
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const g = gradAttnNorm2[i][j];

        for (let k = 0; k < this.embeddingDim; k++) {
          // 勾配を累積
          this.gradW1[k][j] += g * this.lastAttnNorm[i][k];
          gradAttnNorm1[i][k] += g * this.w1[k][j];
        }
      }
    }

    // Layer Norm 1 の逆伝播
    const gradAttnResidual = this.layerNorm1.backwardBatch(this.lastAttnResidual, gradAttnNorm1);

    // 残差接続の逆伝播
    const gradAttnOutput = gradAttnResidual.map(row => [...row]);
    const gradInput1 = gradAttnResidual.map(row => [...row]);

    // Attention の逆伝播
    // output = attn_weights @ V
    const gradAttnWeights: number[][] = Array.from({ length: seqLen }, () =>
      new Array(seqLen).fill(0)
    );
    const gradV: number[][] = Array.from({ length: seqLen }, () =>
      new Array(this.embeddingDim).fill(0)
    );

    for (let i = 0; i < seqLen; i++) {
      for (let k = 0; k < this.embeddingDim; k++) {
        const g = gradAttnOutput[i][k];
        for (let j = 0; j < seqLen; j++) {
          gradAttnWeights[i][j] += g * this.lastV[j][k];
          gradV[j][k] += g * this.lastAttentionWeights[i][j];
        }
      }
    }

    // Softmax の逆伝播
    const gradScores = this.softmaxBackward(gradAttnWeights, this.lastAttentionWeights);

    // スケーリングの逆伝播
    const gradScoresScaled = gradScores.map(row =>
      row.map(val => val / scale)
    );

    // scores = Q @ K^T の逆伝播
    const gradQ: number[][] = Array.from({ length: seqLen }, () =>
      new Array(this.embeddingDim).fill(0)
    );
    const gradK: number[][] = Array.from({ length: seqLen }, () =>
      new Array(this.embeddingDim).fill(0)
    );

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        const g = gradScoresScaled[i][j];
        for (let k = 0; k < this.embeddingDim; k++) {
          gradQ[i][k] += g * this.lastK[j][k];
          gradK[j][k] += g * this.lastQ[i][k];
        }
      }
    }

    // Q, K, V = input @ W の逆伝播
    const gradInput2: number[][] = Array.from({ length: seqLen }, () =>
      new Array(this.embeddingDim).fill(0)
    );

    // Q の逆伝播
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const g = gradQ[i][j];

        for (let k = 0; k < this.embeddingDim; k++) {
          // 勾配を累積
          this.gradWq[k][j] += g * this.lastInput[i][k];
          gradInput2[i][k] += g * this.wq[k][j];
        }
      }
    }

    // K の逆伝播
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const g = gradK[i][j];

        for (let k = 0; k < this.embeddingDim; k++) {
          // 勾配を累積
          this.gradWk[k][j] += g * this.lastInput[i][k];
          gradInput2[i][k] += g * this.wk[k][j];
        }
      }
    }

    // V の逆伝播
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const g = gradV[i][j];

        for (let k = 0; k < this.embeddingDim; k++) {
          // 勾配を累積
          this.gradWv[k][j] += g * this.lastInput[i][k];
          gradInput2[i][k] += g * this.wv[k][j];
        }
      }
    }

    // すべての勾配を合計
    const finalGrad = gradInput1.map((row, i) =>
      row.map((val, j) => val + gradInput2[i][j])
    );

    return finalGrad;
  }

  /**
   * 勾配をゼロにリセット
   */
  zeroGrad(): void {
    for (let i = 0; i < this.embeddingDim; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        this.gradWq[i][j] = 0;
        this.gradWk[i][j] = 0;
        this.gradWv[i][j] = 0;
        this.gradW1[i][j] = 0;
        this.gradW2[i][j] = 0;
      }
    }
  }

  /**
   * 累積した勾配でパラメータを更新
   */
  updateParameters(): void {
    const clipValue = 5.0;
    for (let i = 0; i < this.embeddingDim; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        // Wq の更新
        const clippedGradWq = Math.max(-clipValue, Math.min(clipValue, this.gradWq[i][j]));
        this.wq[i][j] += this.learningRate * clippedGradWq;

        // Wk の更新
        const clippedGradWk = Math.max(-clipValue, Math.min(clipValue, this.gradWk[i][j]));
        this.wk[i][j] += this.learningRate * clippedGradWk;

        // Wv の更新
        const clippedGradWv = Math.max(-clipValue, Math.min(clipValue, this.gradWv[i][j]));
        this.wv[i][j] += this.learningRate * clippedGradWv;

        // W1 の更新
        const clippedGradW1 = Math.max(-clipValue, Math.min(clipValue, this.gradW1[i][j]));
        this.w1[i][j] += this.learningRate * clippedGradW1;

        // W2 の更新
        const clippedGradW2 = Math.max(-clipValue, Math.min(clipValue, this.gradW2[i][j]));
        this.w2[i][j] += this.learningRate * clippedGradW2;
      }
    }
  }
}
