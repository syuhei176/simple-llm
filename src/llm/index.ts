import { SimpleTokenizer } from '../tokenizer';
import { SimpleTransformer } from '../transformer';
import { EmbeddingLayer } from '../embedding';
import { OutputLayer } from '../output';
import { PositionalEncodingCache } from '../positional-encoding';

// 簡易LLM
export class SimpleLLM {
  tokenizer: SimpleTokenizer;
  embedding: EmbeddingLayer;
  transformers: SimpleTransformer[]; // 複数層に変更
  outputLayer: OutputLayer;
  positionalEncoding: PositionalEncodingCache;
  vocabSize: number;
  embeddingDim: number;
  numLayers: number;

  constructor(vocab: string[], embeddingDim = 16, numLayers = 2) {
    this.vocabSize = vocab.length;
    this.embeddingDim = embeddingDim;
    this.numLayers = numLayers;
    this.tokenizer = new SimpleTokenizer(vocab);
    this.embedding = new EmbeddingLayer(vocab.length, embeddingDim);

    // 複数のTransformerレイヤーを作成
    this.transformers = Array.from(
      { length: numLayers },
      () => new SimpleTransformer(embeddingDim)
    );

    this.outputLayer = new OutputLayer(embeddingDim, vocab.length);
    this.positionalEncoding = new PositionalEncodingCache(embeddingDim);
  }

  // 予測（自己回帰生成）
  predict(text: string, maxLen = 10): string {
    let tokens = this.tokenizer.encode(text);

    for (let i = 0; i < maxLen; i++) {
      // Embedding + Positional Encoding
      const embeddedInputs = tokens.map(token => this.embedding.forward(token));
      let output = this.positionalEncoding.addToEmbeddings(embeddedInputs);

      // 複数のTransformerレイヤーを順次適用
      for (const transformer of this.transformers) {
        output = transformer.forward(output);
      }

      const transformerOutput = output;

      // 最後のトークンの出力から次のトークンを予測
      const lastOutput = transformerOutput[transformerOutput.length - 1];
      const logits = this.outputLayer.forward(lastOutput);
      const probs = this.outputLayer.softmax(logits);

      // 最も確率の高いトークンを選択
      const nextToken = probs.indexOf(Math.max(...probs));
      tokens.push(nextToken);

      // 終了条件（パディングやEOSトークンがあれば）
      if (nextToken === 0) break;
    }

    return this.tokenizer.decode(tokens);
  }

  // 学習
  train(trainingData: { input: string; target: string }[], epochs: number = 10) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      trainingData.forEach(({ input, target }) => {
        const inputTokens = this.tokenizer.encode(input);
        const targetTokens = this.tokenizer.encode(target);

        // 順伝播: Embedding + Positional Encoding
        const inputEmbeddings = inputTokens.map(token => this.embedding.forward(token));
        let output = this.positionalEncoding.addToEmbeddings(inputEmbeddings);

        // 複数のTransformerレイヤーを順次適用
        for (const transformer of this.transformers) {
          output = transformer.forward(output);
        }

        const transformerOutput = output;

        // 各位置での損失計算と逆伝播
        const seqLen = Math.min(transformerOutput.length, targetTokens.length);
        const gradTransformer: number[][] = Array.from({ length: transformerOutput.length }, () =>
          new Array(this.embeddingDim).fill(0)
        );

        for (let i = 0; i < seqLen; i++) {
          const logits = this.outputLayer.forward(transformerOutput[i]);
          const probs = this.outputLayer.softmax(logits);

          // Cross-entropy loss の勾配
          const gradLogits = [...probs];
          gradLogits[targetTokens[i]] -= 1; // probs - one_hot(target)

          // 損失の計算（表示用）
          totalLoss -= Math.log(probs[targetTokens[i]] + 1e-10);

          // 出力層の逆伝播
          const gradHidden = this.outputLayer.backward(transformerOutput[i], gradLogits);

          // Transformerへの勾配を蓄積
          for (let j = 0; j < this.embeddingDim; j++) {
            gradTransformer[i][j] = gradHidden[j];
          }
        }

        // Transformerの逆伝播（逆順に適用）
        let gradOutput = gradTransformer;
        for (let i = this.transformers.length - 1; i >= 0; i--) {
          gradOutput = this.transformers[i].backward(gradOutput);
        }
        const gradEmbedding = gradOutput;

        // Embeddingの更新
        for (let i = 0; i < inputTokens.length && i < gradEmbedding.length; i++) {
          this.embedding.backward(inputTokens[i], gradEmbedding[i]);
        }
      });

      const avgLoss = totalLoss / trainingData.length;
      console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${avgLoss.toFixed(4)}`);
    }
  }
}