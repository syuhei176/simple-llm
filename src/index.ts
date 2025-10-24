import { SimpleLLM } from './llm';
import * as readline from 'readline';
import * as fs from 'fs';
import { trainingData, createVocab } from './training-data';

const vocab = createVocab(trainingData);

console.log('Vocabulary:', vocab);
console.log('Vocabulary size:', vocab.length);

// Embedding次元を拡大（より豊かな表現が可能に）
const embeddingDim = 64;
const numLayers = 3; // 複数のTransformerレイヤーを使用
const llm = new SimpleLLM(vocab, embeddingDim, numLayers);

// 学習（デバッグ用に短縮）
console.log('Training started...');
llm.train(trainingData, 10);
// ターミナル対話
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function chat() {
  rl.question("You: ", (input) => {
    if (input.toLowerCase() === "exit") {
      console.log("Goodbye!");
      rl.close();
      return;
    }
    const response = llm.predict(input, 5);
    console.log("AI:", response);
    chat();
  });
}

console.log("Chatbot initialized. Type 'exit' to quit.");
chat();
