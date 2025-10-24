import { SimpleLLM } from './llm';

console.log('=== Simple Overfitting Test ===');

// Minimal training data - just 2 examples
const simpleData = [
  { input: 'hello', target: 'hi' },
  { input: 'bye', target: 'goodbye' }
];

// Create vocabulary from simple data
const vocab: string[] = ['[PAD]', '[UNK]', '[EOS]'];
const vocabSet = new Set<string>(vocab);

simpleData.forEach(d => {
  d.input.split(' ').concat(d.target.split(' ')).forEach(w => {
    const word = w.trim().toLowerCase();
    if (word && !vocabSet.has(word)) {
      vocab.push(word);
      vocabSet.add(word);
    }
  });
});

console.log('Vocabulary:', vocab);
console.log('Vocab size:', vocab.length);

// Tiny model
const embeddingDim = 8;
const numLayers = 1;
const llm = new SimpleLLM(vocab, embeddingDim, numLayers);

console.log('\n=== Training (100 epochs) ===');
llm.train(simpleData, 100);

console.log('\n=== Testing ===');
console.log('Input: hello');
const result1 = llm.predict('hello', 3);
console.log('Output:', result1);
console.log('\nInput: bye');
const result2 = llm.predict('bye', 3);
console.log('Output:', result2);
