/**
 * モデルのロードテスト
 */

import * as fs from 'fs';
import * as path from 'path';
import { SimpleLLM } from '../src/llm';

const modelPath = path.join(process.cwd(), 'models/refactored-test-latest.msgpack');

console.log('Loading model from:', modelPath);

if (!fs.existsSync(modelPath)) {
  console.error('Model file not found');
  process.exit(1);
}

const buffer = fs.readFileSync(modelPath);
const uint8Array = new Uint8Array(buffer);

console.log('Model file size:', uint8Array.length, 'bytes');

const llm = SimpleLLM.deserialize(uint8Array);

console.log('\nModel loaded successfully!');
console.log('Vocab size:', llm.vocabSize);
console.log('Embedding dimension:', llm.embeddingDim);
console.log('Number of layers:', llm.numLayers);
console.log('Number of heads:', llm.numHeads);

console.log('\nTesting prediction...');
const result = llm.predict('hello', 5);
console.log('Input: "hello"');
console.log('Output:', result);

console.log('\n✓ All tests passed!');
