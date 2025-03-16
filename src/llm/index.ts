import { SimpleTokenizer } from '../tokenizer';
import { SimpleTransformer } from '../transformer';

// 簡易LLM
export class SimpleLLM {
  tokenizer: SimpleTokenizer;
  transformer: SimpleTransformer;
  temperature: number = 1;

  constructor(vocab: string[]) {
    this.tokenizer = new SimpleTokenizer(vocab);
    this.transformer = new SimpleTransformer(vocab.length);
  }

  predict(text: string, maxLen = 5): string {
    let tokens = this.tokenizer.encode(text);
    for (let i = 0; i < maxLen; i++) {
      const output = this.transformer.forward(tokens);
      
      // ★ 確率分布を正規化（softmax の簡易版）
      const expOutput = output.map(v => Math.exp(v / this.temperature));
      const sumExp = expOutput.reduce((a, b) => a + b, 1e-8);
      const probabilities = expOutput.map(v => v / sumExp);
      
      // ★ 確率分布から次の単語をサンプリング
      let randomVal = Math.random();
      let cumulativeProb = 0;
      let nextToken = 0;
      for (let j = 0; j < probabilities.length; j++) {
        cumulativeProb += probabilities[j];
        if (randomVal < cumulativeProb) {
          nextToken = j;
          break;
        }
      }
      
      tokens.push(nextToken);
    }
    return this.tokenizer.decode(tokens);
  }

  train(trainingData: { input: string; target: string }[], epochs: number = 10) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      trainingData.forEach(({ input, target }) => {
        const inputTokens = this.tokenizer.encode(input);
        const targetTokens = this.tokenizer.encode(target);
        const outputTokens = this.transformer.forward(inputTokens);
        this.transformer.backward(inputTokens, outputTokens, targetTokens);
      });
      console.log(`Epoch ${epoch + 1}/${epochs} completed.`);
    }
  }

  quantize() {
    this.transformer.quantize();
  }
}