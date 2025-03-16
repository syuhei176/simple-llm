import { transpose } from '../transpose';

// シンプルなTransformer Layer（Attention含む）
export class SimpleTransformer {
  weights: number[][];
  learningRate: number = 0.01;
  embeddingDim: number;

  constructor(embeddingDim: number) {
    this.embeddingDim = embeddingDim;
    this.weights = Array.from({ length: embeddingDim }, () =>
      Array.from({ length: embeddingDim }, () => Math.random() * 0.1 - 0.05)
    );
  }

  attention(input: number[][]): number[][] {
    const inputT = transpose(input);
    const scale = Math.sqrt(this.embeddingDim);
    return inputT.map((vec, i) => {
      const scores = vec.map((v, j) => v * this.weights[i][j]);
      const expScores = scores.map(Math.exp);
      const sumExp = expScores.reduce((a, b) => a + b, 1e-8);
      return expScores.map(v => v / sumExp);
    });
  }

  forward(input: number[][]): number[][] {
    const attn = this.attention(input);
    return transpose(attn);
  }

  backward(input: number[][], output: number[][], target: number[][]): void {
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const error = target[i][j] - output[i][j];
        this.weights[i][j] += this.learningRate * error * input[i][j];
      }
    }
  }
}
