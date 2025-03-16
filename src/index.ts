// 簡易的なTokenizer
class SimpleTokenizer {
  vocab: Record<string, number> = {};
  invVocab: Record<number, string> = {};

  constructor(vocabList: string[]) {
    vocabList.forEach((word, idx) => {
      this.vocab[word] = idx;
      this.invVocab[idx] = word;
    });
  }

  encode(text: string): number[] {
    return text.split(' ').map(w => this.vocab[w] || 0);
  }

  decode(tokens: number[]): string {
    return tokens.map(t => this.invVocab[t] || '[UNK]').join(' ');
  }
}

// シンプルなTransformer Layer（Attention含む）
class SimpleTransformer {
  weights: number[][];

  constructor(size: number) {
    this.weights = Array.from({ length: size }, () =>
      Array.from({ length: size }, () => Math.random() - 0.5)
    );
  }

  attention(input: number[]): number[] {
    const scale = Math.sqrt(input.length);
    const scores = input.map((_, i) => input.reduce((acc, v) => acc + v * this.weights[i][i], 0) / scale);
    const expScores = scores.map(Math.exp);
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(v => v / sumExp);
  }

  forward(input: number[]): number[] {
    const attn = this.attention(input);
    return attn.map((a, i) => input[i] * a);
  }

  quantize() {
    this.weights = this.weights.map(row => row.map(v => parseFloat(v.toFixed(2))));
  }
}

// 簡易LLM
class SimpleLLM {
  tokenizer: SimpleTokenizer;
  transformer: SimpleTransformer;

  constructor(vocab: string[]) {
    this.tokenizer = new SimpleTokenizer(vocab);
    this.transformer = new SimpleTransformer(vocab.length);
  }

  predict(text: string, maxLen = 5): string {
    let tokens = this.tokenizer.encode(text);
    for (let i = 0; i < maxLen; i++) {
      const output = this.transformer.forward(tokens);
      const nextToken = output.indexOf(Math.max(...output));
      tokens.push(nextToken);
    }
    return this.tokenizer.decode(tokens);
  }

  train(input: string[], target: string[]) {
    input.forEach((text, idx) => {
      const inputTokens = this.tokenizer.encode(text);
      const targetTokens = this.tokenizer.encode(target[idx]);
      const outputTokens = this.transformer.forward(inputTokens);
      // シンプルな重み更新（実際はバックプロパゲーションが必要）
      this.transformer.weights = this.transformer.weights.map((row, i) =>
        row.map((w, j) => w + 0.01 * (targetTokens[j] - outputTokens[j]))
      );
    });
  }

  quantize() {
    this.transformer.quantize();
  }
}

// 動作例
const vocab = ['hello', 'world', 'I', 'am', 'AI', '[UNK]'];
const llm = new SimpleLLM(vocab);

// 学習（簡易的な更新）
llm.train(['hello world'], ['I am AI']);

// 量子化
llm.quantize();

// 推論
console.log(llm.predict('hello'));
