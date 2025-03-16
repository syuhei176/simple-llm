export class EmbeddingLayer {
  weights: number[][];

  constructor(vocabSize: number, embeddingDim: number) {
    this.weights = Array.from({ length: vocabSize }, () =>
      Array.from({ length: embeddingDim }, () => Math.random() * 0.1 - 0.05)
    );
  }

  forward(index: number): number[] {
    return this.weights[index] ?? this.weights[0];
  }
}