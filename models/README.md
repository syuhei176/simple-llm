# Models Directory

This directory contains trained language models.

## File Naming Convention

Models are saved with the following naming pattern:
```
<model-name>-<timestamp>.json
<model-name>-latest.json
```

- `<model-name>-<timestamp>.json`: Versioned model file
- `<model-name>-latest.json`: Always points to the most recent model

## Model Structure

Each model file contains:
```json
{
  "version": "1.0",
  "config": {
    "vocabSize": 200,
    "embeddingDim": 32,
    "numLayers": 2,
    "vocab": ["[PAD]", "[UNK]", "[EOS]", ...]
  },
  "weights": {
    "embedding": [...],
    "transformers": [...],
    "output": {...}
  },
  "metadata": {
    "name": "model-name",
    "createdAt": "2025-01-01T00:00:00.000Z",
    "trainingTime": 123.45,
    "trainingSamples": 1000,
    "epochs": 50,
    "embeddingDim": 32,
    "numLayers": 2
  }
}
```

## Usage

### In WebUI
The WebUI automatically loads the latest model from this directory.

### Programmatically
```typescript
import { SimpleLLM } from './src/llm';
import * as fs from 'fs';

const modelData = JSON.parse(fs.readFileSync('models/my-model-latest.json', 'utf-8'));
const llm = SimpleLLM.deserialize(modelData);
const output = llm.predict('hello', 10);
```
