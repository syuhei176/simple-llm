import { SimpleLLM } from './llm';
import * as readline from 'readline';
import * as fs from 'fs';

// 学習データ
const trainingData = [
  { input: 'hello world', target: 'I am AI' },
  { input: 'color', target: 'red blue green yellow' },
  { input: 'how are you', target: 'thank you and you?' },
  { input: 'hello', target: 'how are you' },
  { input: 'animals', target: 'cat dog bird fish' },
];

function createVocab(data: { input: string, target: string }[]) {
  const vocab = new Set<string>();
  data.forEach(d => {
    d.input.split(' ').forEach(w => vocab.add(w));
    d.target.split(' ').forEach(w => vocab.add(w));
  });
  return Array.from(vocab);
}

const vocab = createVocab(trainingData);


console.log(vocab);

const llm = new SimpleLLM(vocab, vocab.length);

// 学習
llm.train(trainingData, 50);

llm.quantize();

console.log(llm.transformer.weights);
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
