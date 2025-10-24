export class EmbeddingLayer {
  weights: number[][];
  embeddingDim: number;
  learningRate: number = 0.01;

  constructor(vocabSize: number, embeddingDim: number) {
    this.embeddingDim = embeddingDim;
    // Xavier初期化
    const scale = Math.sqrt(2.0 / embeddingDim);
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

  // 勾配を受け取って重みを更新
  backward(index: number, gradient: number[]): void {
    if (index < 0 || index >= this.weights.length) {
      return;
    }
    for (let i = 0; i < this.embeddingDim; i++) {
      this.weights[index][i] += this.learningRate * gradient[i];
    }
  }
}