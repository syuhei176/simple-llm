import { SimpleLLM } from './llm';
import * as fs from 'fs';
import * as path from 'path';

console.log('=== Large Vocabulary + Multi-Head Attention Test ===\n');

// Load vocabulary
const vocabPath = path.join(__dirname, '../data/vocab-5000.json');
console.log(`Loading vocabulary from: ${vocabPath}`);
const vocab: string[] = JSON.parse(fs.readFileSync(vocabPath, 'utf-8'));

console.log(`Vocabulary size: ${vocab.length}`);
console.log(`Sample words: ${vocab.slice(4, 14).join(', ')}`); // Skip special tokens
console.log();

// Model configuration
const config = {
  vocabSize: vocab.length,
  embeddingDim: 64,  // Increased from 16 to 64
  numLayers: 2,
  numHeads: 4,       // Multi-head attention
};

console.log('Model configuration:');
console.log(`  Vocabulary size: ${config.vocabSize}`);
console.log(`  Embedding dim: ${config.embeddingDim}`);
console.log(`  Num layers: ${config.numLayers}`);
console.log(`  Num heads: ${config.numHeads}`);
console.log(`  Parameters: ~${estimateParameters(config).toLocaleString()}`);
console.log();

// Create model
console.log('Creating model...');
const llm = new SimpleLLM(vocab, config.embeddingDim, config.numLayers, config.numHeads);
console.log('✓ Model created successfully\n');

// Test 1: Forward pass
console.log('Test 1: Testing forward pass with extended vocabulary...');
const testSentences = [
  'the model learns from data',
  'neural networks are powerful',
  'attention mechanisms process sequences',
];

for (const sentence of testSentences) {
  const result = llm.predict(sentence, 5);
  console.log(`  Input:  "${sentence}"`);
  console.log(`  Output: "${result}"`);
  console.log();
}

// Test 2: Small training
console.log('\nTest 2: Training on sample data...');
const trainingData = [
  { input: 'language models learn', target: 'models learn from data' },
  { input: 'neural networks can', target: 'networks can process text' },
  { input: 'attention mechanisms allow', target: 'mechanisms allow better understanding' },
  { input: 'transformers use multi', target: 'use multi head attention' },
];

console.log(`Training on ${trainingData.length} samples for 3 epochs...`);
llm.train(trainingData, 3);
console.log('✓ Training completed\n');

// Test 3: Post-training inference
console.log('Test 3: Post-training inference...');
for (const sentence of testSentences) {
  const result = llm.predict(sentence, 5);
  console.log(`  Input:  "${sentence}"`);
  console.log(`  Output: "${result}"`);
  console.log();
}

// Test 4: Serialization test
console.log('Test 4: Testing serialization with large vocabulary...');
const serialized = llm.serialize();
console.log(`  Serialized size: ${JSON.stringify(serialized).length.toLocaleString()} characters`);
console.log(`  Version: ${serialized.version}`);
console.log(`  Vocab size in config: ${serialized.config.vocabSize}`);
console.log('  ✓ Serialization successful\n');

// Save model
const modelPath = path.join(__dirname, '../models/large-vocab-multi-head.json');
fs.writeFileSync(modelPath, JSON.stringify(serialized, null, 2));
console.log(`Model saved to: ${modelPath}\n`);

console.log('=== All tests passed! ===');
console.log('\nSummary:');
console.log(`  Vocabulary: ${vocab.length} words (up from 43)`);
console.log(`  Embedding dim: ${config.embeddingDim} (up from 16)`);
console.log(`  Attention heads: ${config.numHeads} (up from 1)`);
console.log(`  Model parameters: ~${estimateParameters(config).toLocaleString()}`);

function estimateParameters(cfg: typeof config): number {
  const { vocabSize, embeddingDim, numLayers, numHeads } = cfg;
  const headDim = embeddingDim / numHeads;

  // Embedding layer
  const embeddingParams = vocabSize * embeddingDim;

  // Each transformer layer
  const qkvParams = 3 * embeddingDim * headDim * numHeads; // Q, K, V for all heads
  const outProjParams = embeddingDim * embeddingDim; // Output projection
  const ffnParams = 2 * embeddingDim * embeddingDim; // Two feed-forward layers
  const layerNormParams = 2 * embeddingDim * 2; // 2 layer norms, each with gamma and beta

  const perLayerParams = qkvParams + outProjParams + ffnParams + layerNormParams;
  const transformerParams = perLayerParams * numLayers;

  // Output layer
  const outputParams = embeddingDim * vocabSize + vocabSize; // weights + bias

  return embeddingParams + transformerParams + outputParams;
}
