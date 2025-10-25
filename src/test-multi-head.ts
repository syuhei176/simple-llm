import { SimpleLLM } from './llm';

console.log('=== Multi-Head Attention Test ===\n');

// Test 1: Create model with multi-head attention
console.log('Test 1: Creating model with 4 heads...');
const vocab = ['[PAD]', '[UNK]', '[EOS]', 'hello', 'world', 'test', 'multi', 'head', 'attention'];
const llm = new SimpleLLM(vocab, 32, 2, 4); // 32 dim, 2 layers, 4 heads

console.log(`  Vocab size: ${llm.vocabSize}`);
console.log(`  Embedding dim: ${llm.embeddingDim}`);
console.log(`  Num layers: ${llm.numLayers}`);
console.log(`  Num heads: ${llm.numHeads}`);
console.log('  ✓ Model created successfully\n');

// Test 2: Forward pass
console.log('Test 2: Testing forward pass...');
try {
  const result = llm.predict('hello', 3);
  console.log(`  Input: "hello"`);
  console.log(`  Output: "${result}"`);
  console.log('  ✓ Forward pass successful\n');
} catch (error) {
  console.error('  ✗ Forward pass failed:', error);
  process.exit(1);
}

// Test 3: Training
console.log('Test 3: Testing training...');
const trainingData = [
  { input: 'hello', target: 'world' },
  { input: 'test', target: 'multi head' },
];

try {
  llm.train(trainingData, 2);
  console.log('  ✓ Training completed successfully\n');
} catch (error) {
  console.error('  ✗ Training failed:', error);
  process.exit(1);
}

// Test 4: Serialization
console.log('Test 4: Testing serialization...');
try {
  const serialized = llm.serialize();
  console.log(`  Version: ${serialized.version}`);
  console.log(`  Config: ${JSON.stringify(serialized.config)}`);
  console.log('  ✓ Serialization successful\n');
} catch (error) {
  console.error('  ✗ Serialization failed:', error);
  process.exit(1);
}

// Test 5: Deserialization
console.log('Test 5: Testing deserialization...');
try {
  const serialized = llm.serialize();
  const restored = SimpleLLM.deserialize(serialized);
  console.log(`  Restored vocab size: ${restored.vocabSize}`);
  console.log(`  Restored embedding dim: ${restored.embeddingDim}`);
  console.log(`  Restored num layers: ${restored.numLayers}`);
  console.log(`  Restored num heads: ${restored.numHeads}`);
  console.log('  ✓ Deserialization successful\n');
} catch (error) {
  console.error('  ✗ Deserialization failed:', error);
  process.exit(1);
}

// Test 6: Single-head backward compatibility
console.log('Test 6: Testing single-head backward compatibility...');
try {
  const singleHeadLLM = new SimpleLLM(vocab, 32, 2, 1); // 1 head
  singleHeadLLM.predict('hello', 2);
  console.log('  ✓ Single-head model works correctly\n');
} catch (error) {
  console.error('  ✗ Single-head test failed:', error);
  process.exit(1);
}

console.log('=== All tests passed! ===');
