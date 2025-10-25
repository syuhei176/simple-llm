import { SimpleTokenizer } from '../tokenizer';
import { SimpleTransformer, MultiHeadTransformer } from '../transformer';
import { EmbeddingLayer } from '../embedding';
import { OutputLayer } from '../output';
import { PositionalEncodingCache } from '../positional-encoding';
import { Optimizer, AdamOptimizer, SGDOptimizer } from '../optimizer';
import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack';

// 簡易LLM
export class SimpleLLM {
  tokenizer: SimpleTokenizer;
  embedding: EmbeddingLayer;
  transformers: (SimpleTransformer | MultiHeadTransformer)[]; // 複数層に変更
  outputLayer: OutputLayer;
  positionalEncoding: PositionalEncodingCache;
  vocabSize: number;
  embeddingDim: number;
  numLayers: number;
  numHeads: number;
  optimizer?: Optimizer;

  constructor(vocab: string[], embeddingDim = 16, numLayers = 2, numHeads = 1) {
    this.vocabSize = vocab.length;
    this.embeddingDim = embeddingDim;
    this.numLayers = numLayers;
    this.numHeads = numHeads;
    this.tokenizer = new SimpleTokenizer(vocab);
    this.embedding = new EmbeddingLayer(vocab.length, embeddingDim);

    // 複数のTransformerレイヤーを作成
    // numHeads > 1 の場合はMultiHeadTransformerを使用
    this.transformers = Array.from(
      { length: numLayers },
      () => numHeads > 1
        ? new MultiHeadTransformer(embeddingDim, numHeads)
        : new SimpleTransformer(embeddingDim)
    );

    this.outputLayer = new OutputLayer(embeddingDim, vocab.length);
    this.positionalEncoding = new PositionalEncodingCache(embeddingDim);
  }

  // 特殊トークンのインデックスを取得
  private getSpecialTokenIndices(): Set<number> {
    const special = new Set<number>();
    const specialTokens = ['[PAD]', '[UNK]', '[EOS]'];
    specialTokens.forEach(token => {
      const idx = this.tokenizer.vocabMap[token];
      if (idx !== undefined) special.add(idx);
    });
    return special;
  }

  // 予測（自己回帰生成）
  predict(text: string, maxLen = 10): string {
    let tokens = this.tokenizer.encode(text);
    const specialTokens = this.getSpecialTokenIndices();
    const eosIndex = this.tokenizer.vocabMap['[EOS]'];

    console.log('Input text:', text);
    console.log('Encoded tokens:', tokens);
    console.log('Special token indices:', Array.from(specialTokens));
    console.log('Vocab size:', this.vocabSize);

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

      // デバッグ：上位5つの確率を表示
      const topProbs = probs
        .map((p, idx) => ({ prob: p, idx, word: this.tokenizer.vocab[idx] }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);
      console.log(`Step ${i} - Top 5 probs (before filter):`, topProbs);

      // 特殊トークンの確率を0にする
      const filteredProbs = probs.map((p, idx) =>
        specialTokens.has(idx) ? 0 : p
      );

      // 再正規化
      const sum = filteredProbs.reduce((a, b) => a + b, 1e-10);
      const normalizedProbs = filteredProbs.map(p => p / sum);

      // 最も確率の高いトークンを選択
      const maxProb = Math.max(...normalizedProbs);
      const nextToken = normalizedProbs.indexOf(maxProb);

      console.log(`Step ${i}: nextToken=${nextToken}, maxProb=${maxProb.toFixed(4)}, word="${this.tokenizer.vocab[nextToken]}"`);

      // デバッグ：上位5つのフィルタ後の確率を表示
      const topFilteredProbs = normalizedProbs
        .map((p, idx) => ({ prob: p, idx, word: this.tokenizer.vocab[idx] }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 5);
      console.log(`Step ${i} - Top 5 probs (after filter):`, topFilteredProbs);

      tokens.push(nextToken);

      // EOSトークンが生成されたら終了
      if (nextToken === eosIndex) break;
    }

    const result = this.tokenizer.decode(tokens);
    console.log('Final result:', result);
    return result;
  }

  // Set optimizer
  setOptimizer(optimizer: Optimizer): void {
    this.optimizer = optimizer;
  }

  // 学習
  train(
    trainingData: { input: string; target: string }[],
    epochs: number = 10,
    optimizerType: 'sgd' | 'adam' = 'adam',
    learningRate: number = 0.001
  ) {
    // Create optimizer if not set
    if (!this.optimizer) {
      if (optimizerType === 'adam') {
        this.optimizer = new AdamOptimizer(learningRate);
      } else {
        this.optimizer = new SGDOptimizer(learningRate);
      }
    }
    for (let epoch = 0; epoch < epochs; epoch++) {
      console.log(`Epoch ${epoch + 1}/${epochs}`);
      let totalLoss = 0;

      trainingData.forEach(({ input, target }) => {
        // 勾配をゼロリセット
        this.embedding.zeroGrad();
        for (const transformer of this.transformers) {
          transformer.zeroGrad();
          transformer.layerNorm1.zeroGrad();
          transformer.layerNorm2.zeroGrad();
        }
        this.outputLayer.zeroGrad();

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

          // Cross-entropy loss の勾配 (negative for gradient descent)
          const gradLogits = probs.map(p => -p);
          gradLogits[targetTokens[i]] += 1; // one_hot(target) - probs

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

        // パラメータ更新
        if (this.optimizer) {
          // Use optimizer for parameter updates
          this.embedding.updateWithOptimizer(this.optimizer);
          this.outputLayer.updateWithOptimizer(this.optimizer);
          for (let i = 0; i < this.transformers.length; i++) {
            const transformer = this.transformers[i];
            if (transformer instanceof MultiHeadTransformer) {
              transformer.updateWithOptimizer(this.optimizer, `transformer_${i}`);
            } else {
              transformer.updateWithOptimizer(this.optimizer, `transformer_${i}`);
            }
          }
        } else {
          // Fallback to manual updates
          this.embedding.updateParametersManual();
          this.outputLayer.updateParameters();
          for (const transformer of this.transformers) {
            transformer.updateParameters();
            transformer.layerNorm1.updateParameters();
            transformer.layerNorm2.updateParameters();
          }
        }
      });

      const avgLoss = totalLoss / trainingData.length;
      console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${avgLoss.toFixed(4)}`);
    }
  }

  // モデルをシリアライズ（MessagePack形式）
  serialize(): Uint8Array {
    const data = {
      version: '2.0',
      config: {
        vocabSize: this.vocabSize,
        embeddingDim: this.embeddingDim,
        numLayers: this.numLayers,
        numHeads: this.numHeads,
        vocab: this.tokenizer.vocab,
      },
      weights: {
        embedding: this.embedding.weights,
        transformers: this.transformers.map(t => {
          if (t instanceof MultiHeadTransformer) {
            return {
              type: 'multi-head',
              multiHeadAttention: {
                wq: t.multiHeadAttention.wq,
                wk: t.multiHeadAttention.wk,
                wv: t.multiHeadAttention.wv,
                wo: t.multiHeadAttention.wo,
              },
              w1: t.w1,
              w2: t.w2,
              layerNorm1: {
                gamma: t.layerNorm1.gamma,
                beta: t.layerNorm1.beta,
              },
              layerNorm2: {
                gamma: t.layerNorm2.gamma,
                beta: t.layerNorm2.beta,
              },
            };
          } else {
            return {
              type: 'single-head',
              wq: t.wq,
              wk: t.wk,
              wv: t.wv,
              w1: t.w1,
              w2: t.w2,
              layerNorm1: {
                gamma: t.layerNorm1.gamma,
                beta: t.layerNorm1.beta,
              },
              layerNorm2: {
                gamma: t.layerNorm2.gamma,
                beta: t.layerNorm2.beta,
              },
            };
          }
        }),
        output: {
          weights: this.outputLayer.weights,
          bias: this.outputLayer.bias,
        },
      },
    };
    return msgpackEncode(data);
  }

  // モデルをデシリアライズ（MessagePack形式）
  static deserialize(buffer: Uint8Array): SimpleLLM {
    const data = msgpackDecode(buffer) as any;
    const { config, weights } = data;

    const llm = new SimpleLLM(
      config.vocab,
      config.embeddingDim,
      config.numLayers,
      config.numHeads
    );

    // Embeddingの重みを復元
    llm.embedding.weights = weights.embedding;

    // Transformerの重みを復元
    weights.transformers.forEach((tData: any, i: number) => {
      const transformer = llm.transformers[i];

      if (tData.type === 'multi-head' && transformer instanceof MultiHeadTransformer) {
        transformer.multiHeadAttention.wq = tData.multiHeadAttention.wq;
        transformer.multiHeadAttention.wk = tData.multiHeadAttention.wk;
        transformer.multiHeadAttention.wv = tData.multiHeadAttention.wv;
        transformer.multiHeadAttention.wo = tData.multiHeadAttention.wo;
        transformer.w1 = tData.w1;
        transformer.w2 = tData.w2;
        transformer.layerNorm1.gamma = tData.layerNorm1.gamma;
        transformer.layerNorm1.beta = tData.layerNorm1.beta;
        transformer.layerNorm2.gamma = tData.layerNorm2.gamma;
        transformer.layerNorm2.beta = tData.layerNorm2.beta;
      } else if (transformer instanceof SimpleTransformer) {
        transformer.wq = tData.wq;
        transformer.wk = tData.wk;
        transformer.wv = tData.wv;
        transformer.w1 = tData.w1;
        transformer.w2 = tData.w2;
        transformer.layerNorm1.gamma = tData.layerNorm1.gamma;
        transformer.layerNorm1.beta = tData.layerNorm1.beta;
        transformer.layerNorm2.gamma = tData.layerNorm2.gamma;
        transformer.layerNorm2.beta = tData.layerNorm2.beta;
      }
    });

    // Outputレイヤーの重みを復元
    llm.outputLayer.weights = weights.output.weights;
    llm.outputLayer.bias = weights.output.bias;

    return llm;
  }
}