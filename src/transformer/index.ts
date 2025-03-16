// シンプルなTransformer Layer（Attention含む）
export class SimpleTransformer {
  weights: number[][];
  learningRate: number = 0.01;

  constructor(size: number) {
    this.weights = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => Math.random() * 0.1 - 0.05) // 小さな初期値を使用
    );
  }

  attention(input: number[]): number[] {
    const scale = Math.sqrt(input.length);
    const scores = input.map((_, i) => input.reduce((acc, v) => acc + v * this.weights[i][i], 0) / scale);

    const expScores = scores.map(Math.exp);
    const sumExp = expScores.reduce((a, b) => a + b, 1e-8); // ゼロ割り防止

    return expScores.map(v => v / sumExp);
  }

  forward(input: number[]): number[] {
    const attn = this.attention(input);
    return attn.map((a, i) => input[i] * a);
  }

  backward(input: number[], output: number[], target: number[]): void {
    const error = output.map((o, i) => target[i] - o);
    this.weights = this.weights.map((row, i) =>
      row.map((w, j) => {
        const newWeight = w + this.learningRate * error[j] * input[i];
        return isNaN(newWeight) ? w : newWeight; // NaNを防ぐ
      })
    );
  }

  quantize() {
    this.weights = this.weights.map(row => row.map(v => parseFloat(v.toFixed(2))));
  }
}