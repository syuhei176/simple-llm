// 出力層: Embedding次元からVocabularyサイズへの線形変換
export class OutputLayer {
  weights: number[][];
  bias: number[];
  gradWeights: number[][];
  gradBias: number[];
  learningRate: number = 0.001; // 勾配爆発防止のため学習率を下げる
  vocabSize: number;
  embeddingDim: number;

  constructor(embeddingDim: number, vocabSize: number) {
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
    // Xavier初期化
    const scale = Math.sqrt(1.0 / (embeddingDim + vocabSize));
    this.weights = Array.from({ length: embeddingDim }, () =>
      Array.from({ length: vocabSize }, () => (Math.random() * 2 - 1) * scale)
    );
    this.bias = Array.from({ length: vocabSize }, () => 0);
    // 勾配を0で初期化
    this.gradWeights = Array.from({ length: embeddingDim }, () =>
      Array.from({ length: vocabSize }, () => 0)
    );
    this.gradBias = Array.from({ length: vocabSize }, () => 0);
  }

  // 順伝播: embedding -> logits
  forward(input: number[]): number[] {
    const output = new Array(this.vocabSize).fill(0);
    for (let i = 0; i < this.vocabSize; i++) {
      let sum = this.bias[i];
      for (let j = 0; j < this.embeddingDim; j++) {
        sum += input[j] * this.weights[j][i];
      }
      output[i] = sum;
    }
    return output;
  }

  // Softmax関数
  softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expScores = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expScores.reduce((a, b) => a + b, 1e-8);
    return expScores.map(x => x / sumExp);
  }

  // 逆伝播
  backward(input: number[], gradOutput: number[]): number[] {
    const gradInput = new Array(this.embeddingDim).fill(0);

    // 勾配を累積
    for (let i = 0; i < this.embeddingDim; i++) {
      for (let j = 0; j < this.vocabSize; j++) {
        this.gradWeights[i][j] += gradOutput[j] * input[i];
        gradInput[i] += gradOutput[j] * this.weights[i][j];
      }
    }

    for (let j = 0; j < this.vocabSize; j++) {
      this.gradBias[j] += gradOutput[j];
    }

    return gradInput;
  }

  /**
   * 勾配をゼロにリセット
   */
  zeroGrad(): void {
    for (let i = 0; i < this.embeddingDim; i++) {
      for (let j = 0; j < this.vocabSize; j++) {
        this.gradWeights[i][j] = 0;
      }
    }
    for (let j = 0; j < this.vocabSize; j++) {
      this.gradBias[j] = 0;
    }
  }

  /**
   * 累積した勾配でパラメータを更新
   */
  updateParameters(): void {
    const clipValue = 5.0;
    for (let i = 0; i < this.embeddingDim; i++) {
      for (let j = 0; j < this.vocabSize; j++) {
        const clippedGrad = Math.max(-clipValue, Math.min(clipValue, this.gradWeights[i][j]));
        this.weights[i][j] += this.learningRate * clippedGrad;
      }
    }

    for (let j = 0; j < this.vocabSize; j++) {
      const clippedGrad = Math.max(-clipValue, Math.min(clipValue, this.gradBias[j]));
      this.bias[j] += this.learningRate * clippedGrad;
    }
  }
}
