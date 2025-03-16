import { SimpleLLM } from './llm';
import * as readline from 'readline';


// 会話データセット
const conversationData = [
  { input: "Hello", target: "Hi, how can I help you?" },
  { input: "How are you?", target: "I'm doing well, thank you!" },
  { input: "What is your name?", target: "I am an AI assistant." },
  { input: "who are you?", target: "I am an AI assistant." },
  { input: "What is your favorite color?", target: "green" },
  { input: "Tell me a joke", target: "Why did the AI cross the road? To optimize the other side!" },
  { input: "Goodbye", target: "Goodbye! Have a great day!" },
  { input: "I'm hungry", target: "I'm sorry, I can't help with that." },
  { input: "I'm tired", target: "I'm sorry, I can't help with that." },
  { input: "I'm happy", target: "I'm sorry, I can't help with that." },
  { input: "I'm sad", target: "I'm sorry, I can't help with that." },
  { input: "function add(a, b)", target: "{ return a + b; }" }
   
];

// 単語リスト（Vocabulary）を生成する関数
function buildVocabulary(data: { input: string; target: string }[]): string[] {
  const words = new Set<string>();
  data.forEach(({ input, target }) => {
    input.split(/\W+/).forEach(word => words.add(word.toLowerCase()));
    target.split(/\W+/).forEach(word => words.add(word.toLowerCase()));
  });
  return Array.from(words).filter(word => word.length > 0); // 空文字を除外
}

// Vocabularyリストを作成
const vocabulary = buildVocabulary(conversationData);

console.log("Vocabulary List:", vocabulary);

// 動作例
const llm = new SimpleLLM(vocabulary);

// 学習（簡易的な更新）
llm.train(conversationData, 120);

// 量子化
llm.quantize();

// ターミナルでの対話処理
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