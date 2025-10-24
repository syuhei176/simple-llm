// 出力層: Embedding次元からVocabularyサイズへの線形変換
export class OutputLayer {
  weights: number[][];
  bias: number[];
  learningRate: number = 0.01;
  vocabSize: number;
  embeddingDim: number;

  constructor(embeddingDim: number, vocabSize: number) {
    this.embeddingDim = embeddingDim;
    this.vocabSize = vocabSize;
    // Xavier初期化
    const scale = Math.sqrt(2.0 / (embeddingDim + vocabSize));
    this.weights = Array.from({ length: embeddingDim }, () =>
      Array.from({ length: vocabSize }, () => (Math.random() * 2 - 1) * scale)
    );
    this.bias = Array.from({ length: vocabSize }, () => 0);
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

    // 重みとバイアスの更新
    for (let i = 0; i < this.embeddingDim; i++) {
      for (let j = 0; j < this.vocabSize; j++) {
        this.weights[i][j] += this.learningRate * gradOutput[j] * input[i];
        gradInput[i] += gradOutput[j] * this.weights[i][j];
      }
    }

    for (let j = 0; j < this.vocabSize; j++) {
      this.bias[j] += this.learningRate * gradOutput[j];
    }

    return gradInput;
  }
}
