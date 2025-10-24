import { SimpleLLM } from './llm';
import { trainingData, createVocab } from './training-data';

console.log('=== Creating Vocabulary ===');
const vocab = createVocab(trainingData);

console.log('\n=== Vocabulary Info ===');
console.log('Vocabulary size:', vocab.length);
console.log('First 20 words:', vocab.slice(0, 20));

console.log('\n=== Initializing Model ===');
// シンプルな構成でテスト
const embeddingDim = 16;
const numLayers = 1;
const llm = new SimpleLLM(vocab, embeddingDim, numLayers);

console.log('\n=== Training Model (10 epochs) ===');
llm.train(trainingData, 50);

console.log('\n=== Testing Predictions ===');

// テスト1: hello
console.log('\n--- Test 1: "hello" ---');
const result1 = llm.predict('hello', 5);
console.log('Result:', result1);

// テスト2: how are you
console.log('\n--- Test 2: "how are you" ---');
const result2 = llm.predict('how are you', 5);
console.log('Result:', result2);

// テスト3: colors
console.log('\n--- Test 3: "colors" ---');
const result3 = llm.predict('colors', 5);
console.log('Result:', result3);

console.log('\n=== Test Complete ===');
