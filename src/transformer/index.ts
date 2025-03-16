import { transpose } from '../transpose';

// シンプルなTransformer Layer（Attention含む）
export class SimpleTransformer {
  weights: number[][];
  learningRate: number = 0.01;

  constructor(embeddingDim: number) {
    this.weights = Array.from({ length: embeddingDim }, () =>
      Array.from({ length: embeddingDim }, () => Math.random() * 0.1 - 0.05)
    );
  }

  attention(input: number[][]): number[][] {
    const transposedInput = transpose(input);
    const scale = Math.sqrt(transposedInput.length);

    return transposedInput.map((vec, i) => {
      const scores = vec.map((v, j) => v * this.weights[i][j] / scale);
      const expScores = scores.map(Math.exp);
      const sumExp = expScores.reduce((a, b) => a + b, 1e-8);
      return expScores.map(v => v / sumExp);
    });
  }

  forward(input: number[][]): number[][] {
    const attn = this.attention(input);
    return attn.map((a, i) => input[i].map((v, j) => v * a[j]));
  }

  backward(input: number[][], output: number[][], target: number[][]): void {
    for (let i = 0; i < input.length; i++) {
      for (let j = 0; j < input[i].length; j++) {
        const error = target[i][j] - output[i][j];
        this.weights[i][j] += this.learningRate * error * input[i][j];
      }
    }
  }

  quantize() {
    this.weights = this.weights.map(row => row.map(v => parseFloat(v.toFixed(4))));
  }
}
