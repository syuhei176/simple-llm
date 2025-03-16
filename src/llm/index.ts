import { SimpleTokenizer } from '../tokenizer';
import { SimpleTransformer } from '../transformer';
import { EmbeddingLayer } from '../embedding';
import { transpose } from '../transpose';

// 簡易LLM
export class SimpleLLM {
  tokenizer: SimpleTokenizer;
  embedding: EmbeddingLayer;
  transformer: SimpleTransformer;

  constructor(vocab: string[], embeddingDim = 4) {
    this.tokenizer = new SimpleTokenizer(vocab);
    this.embedding = new EmbeddingLayer(vocab.length, embeddingDim);
    this.transformer = new SimpleTransformer(embeddingDim);
  }

  predict(text: string, maxLen = 5): string {
    let tokens = this.tokenizer.encode(text);
    let embeddedInputs = tokens.map(token => this.embedding.forward(token));
    for (let i = 0; i < maxLen; i++) {
      const output = this.transformer.forward(embeddedInputs);
      const nextToken = output[output.length - 1].indexOf(Math.max(...output[output.length - 1]));
      embeddedInputs.push(this.embedding.forward(nextToken));
      tokens.push(nextToken);
    }
    return this.tokenizer.decode(tokens);
  }

  train(trainingData: { input: string; target: string }[], epochs: number = 10) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      trainingData.forEach(({ input, target }) => {
        const inputTokens = this.tokenizer.encode(input);
        const targetTokens = this.tokenizer.encode(target);
        const inputVectors = inputTokens.map(token => this.embedding.forward(token));
        const targetVectors = targetTokens.map(token => this.embedding.forward(token));
        const outputVectors = this.transformer.forward(inputVectors);
        this.transformer.backward(inputVectors, outputVectors, targetVectors);
      });
      console.log(`Epoch ${epoch + 1}/${epochs} completed.`);
    }
  }
}

function printMatrix(name: string, matrix: number[][]) {
  console.log(name, matrix.length, matrix[0].length);
}