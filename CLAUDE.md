# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational Transformer-based language model implementation built from scratch in TypeScript. The project demonstrates core LLM concepts including self-attention, positional encoding, layer normalization, and backpropagation, with both a web interface and a complete training pipeline.

## Common Commands

### Development
```bash
# Install dependencies
pnpm install

# Build TypeScript
npm run build

# Build web bundle (outputs to docs/bundle.js)
npm run build:web

# Run development server with hot reload (http://localhost:8080)
npm run dev
```

### Training Pipeline
```bash
# Run complete pipeline: fetch → prepare → train → deploy
npm run pipeline

# Individual pipeline steps
npm run fetch -- <url> <output-file>           # Fetch training text
npm run prepare -- <input-txt> <output-json>   # Convert to training data
npm run train -- <training-json> <model-name>  # Train and save model

# Custom pipeline with parameters
npm run pipeline:custom -- --corpus ./data/my-text.txt --model my-model --epochs 100
```

### Script Execution
Scripts use a separate TypeScript config (`tsconfig.scripts.json`). Always run scripts with:
```bash
npx ts-node --project tsconfig.scripts.json scripts/<script-name>.ts
```

## Architecture

### Core Components

The LLM is built from composable layers that work together:

**SimpleLLM (src/llm/index.ts)** - Main orchestrator
- Manages the complete model lifecycle: forward pass, backpropagation, training, serialization
- Coordinates multiple transformer layers in sequence
- Handles autoregressive text generation with special token filtering
- Implements gradient flow through all layers during backpropagation

**SimpleTransformer (src/transformer/index.ts)** - Attention layer
- Implements self-attention with separate Q, K, V weight matrices
- Uses scaled dot-product attention: `softmax(QK^T / √d) × V`
- Includes feed-forward network with ReLU activation
- Has residual connections and layer normalization after each sub-layer
- Implements complete backpropagation through attention mechanism

**EmbeddingLayer (src/embedding/index.ts)** - Token to vector conversion
- Maps discrete tokens to continuous embeddings using Xavier initialization
- Embedding dimension is configurable (default: 32-64)
- Includes gradient clipping to prevent exploding gradients

**PositionalEncoding (src/positional-encoding.ts)** - Position information
- Adds sinusoidal position encodings to embeddings
- Uses `sin` and `cos` functions at different frequencies
- Critical for preserving token order in the attention mechanism

**OutputLayer (src/output/index.ts)** - Logits to tokens
- Projects from embedding dimension to vocabulary size
- Applies softmax for probability distribution
- Computes cross-entropy loss gradients during training

**LayerNorm (src/layer-norm.ts)** - Training stabilization
- Normalizes activations to zero mean and unit variance
- Learnable scale (gamma) and shift (beta) parameters
- Applied after attention and feed-forward sub-layers

### Data Flow

**Forward Pass:**
```
Input Tokens → Embedding → + Positional Encoding
    ↓
Transformer Layer 1:
    Self-Attention → + Residual → Layer Norm
    ↓
    Feed-Forward → + Residual → Layer Norm
    ↓
[Repeat for N layers]
    ↓
Output Layer → Softmax → Token Probabilities
```

**Training (Backpropagation):**
```
Loss (Cross-Entropy)
    ↓
Output Layer Gradients
    ↓
[For each Transformer layer in reverse]:
    Layer Norm 2 → Residual → Feed-Forward → Layer Norm 1 → Residual → Attention
    ↓
Embedding Gradients → Weight Updates
```

### Key Implementation Details

**Attention Mechanism:**
- The transformer implements proper scaled dot-product attention, not simplified element-wise operations
- Attention scores are scaled by `1/√d` to prevent gradient vanishing
- Softmax is applied row-wise to get attention weights
- Gradients flow through the entire attention computation graph during backpropagation

**Gradient Management:**
- Each layer has separate `zeroGrad()` and `updateParameters()` methods
- Gradients accumulate across a batch, then weights update via gradient descent
- Gradient clipping (value: 5.0) prevents exploding gradients
- Learning rate: 0.001 for transformer layers, 0.001 for embeddings

**Special Tokens:**
- `[PAD]` (index 0): Padding token
- `[UNK]` (index 1): Unknown words
- `[EOS]` (index 2): End of sequence
- These tokens are filtered during generation to prevent the model from outputting them

**Autoregressive Generation:**
- Model generates one token at a time
- Each new token is appended to the input sequence
- Generation continues until `[EOS]` is produced or max length is reached
- Special tokens are filtered from the probability distribution before sampling

## Training Pipeline Architecture

The pipeline consists of four stages:

### 1. Data Collection (scripts/fetch-text.ts)
- Downloads text from URLs or uses sample data
- Supports multiple sources with `--multiple` flag
- Outputs raw text files to `data/` directory

### 2. Data Preparation (scripts/prepare-training-data.ts)
- Converts long text into training pairs using sliding window technique
- **Window Size**: Number of words in input/target (e.g., 5 words)
- **Stride**: How many words to advance the window (e.g., stride=1 for maximum overlap)
- Creates `{ input: "word1 word2 ...", target: "word2 word3 ..." }` pairs
- Filters short words with `--min-word` parameter
- Can limit samples with `--max` parameter

### 3. Model Training (scripts/train-model.ts)
- Builds vocabulary from training data (includes special tokens)
- Initializes model with configurable hyperparameters
- Trains using gradient descent with cross-entropy loss
- Saves two versions: timestamped and "latest"
- Outputs to `models/` directory in JSON format

### 4. Model Deployment
- Models in `models/` are automatically available to WebUI
- WebUI loads `models/default-latest.json` by default
- Can also load models from IndexedDB (browser storage)

### Pipeline Configuration

**Default Settings:**
- Window size: 5 words
- Stride: 1 word (maximum data efficiency)
- Epochs: 50
- Embedding dimension: 32
- Transformer layers: 2

**Recommended Settings by Data Size:**
- Small (<500 words): window=3-5, stride=1, epochs=50-100, embedding=16-32
- Medium (500-2000 words): window=5-7, stride=2, epochs=50-150, embedding=32-64
- Large (>2000 words): window=7-10, stride=3, epochs=100-200, embedding=64-128

## Model Serialization

Models are saved as JSON with this structure:
```typescript
{
  version: "1.0",
  config: {
    vocabSize: number,
    embeddingDim: number,
    numLayers: number,
    vocab: string[]
  },
  weights: {
    embedding: number[][],
    transformers: [{
      wq, wk, wv: number[][]  // Attention weights
      w1, w2: number[][]       // Feed-forward weights
      layerNorm1: { gamma, beta }
      layerNorm2: { gamma, beta }
    }],
    output: {
      weights: number[][],
      bias: number[]
    }
  }
}
```

**Model Loading:**
- WebUI first tries to load from `models/default-latest.json` in the repository
- Falls back to IndexedDB if repository model is not found
- Use `SimpleLLM.deserialize(data)` to load a saved model

## Web Interface

The web interface (`docs/index.html` + `src/web.ts`) provides:
- Interactive chat interface
- Model training in the browser
- Model save/load to IndexedDB
- Repository model loading from `models/` directory
- Real-time training progress

**Build Process:**
- Entry point: `src/web.ts`
- Webpack config: `webpack.config.js`
- TypeScript config: `tsconfig.web.json`
- Output: `docs/bundle.js`
- Served via webpack-dev-server on port 8080

## GitHub Actions

The repository includes automated training workflows:

**Train Language Model (.github/workflows/train-model.yml):**
- Triggers: Manual dispatch, scheduled (weekly), or PR to `data/` or `scripts/`
- Configurable parameters: data source, model name, window size, epochs, etc.
- Automatically commits trained models back to the repository
- Uploads training artifacts (models, data, reports)

**Quick Pipeline Test (.github/workflows/quick-test.yml):**
- Validates pipeline functionality on PRs
- Runs minimal training (5 epochs) for fast CI/CD
- Verifies model file structure

## Code Patterns

**Adding New Training Data:**
Edit `src/training-data.ts` and add entries to the `trainingData` array. Each entry should have an `input` and `target` field with lowercase text.

**Adjusting Model Architecture:**
To change the number of layers or embedding dimension, modify the constructor parameters in `SimpleLLM`:
```typescript
const llm = new SimpleLLM(vocab, embeddingDim, numLayers);
```

**Custom Training Loop:**
The `train()` method in `SimpleLLM` handles the full training loop. To customize:
1. Modify learning rate in individual layer classes
2. Adjust gradient clipping values
3. Change loss computation in the training loop

**Adding New Layers:**
Each layer should implement:
- `forward(input)`: Compute output and cache values needed for backprop
- `backward(gradient)`: Compute gradients and return gradient for previous layer
- `zeroGrad()`: Reset gradient accumulators
- `updateParameters()`: Apply gradients to weights

## Important Notes

**TypeScript Configurations:**
- `tsconfig.json`: Main TypeScript config for `src/`
- `tsconfig.web.json`: Web bundle build (target: ES2015 for browsers)
- `tsconfig.scripts.json`: Scripts with Node.js types

**Gradient Accumulation:**
- Gradients accumulate during backward pass for all samples in the training data
- Always call `zeroGrad()` at the start of each training step
- Always call `updateParameters()` after backward pass

**Matrix Operations:**
- All matrix operations are implemented from scratch (no external libraries)
- `matmul`, `transpose`, `vecMatmul` are implemented in transformer layer
- Be careful with matrix dimensions when modifying architecture

**Vocabulary Management:**
- Vocabulary is built from training data, not predefined
- Special tokens `[PAD]`, `[UNK]`, `[EOS]` are always at indices 0, 1, 2
- Use `createVocab(data)` from `src/training-data.ts` to build vocabulary

**Model Storage:**
- Repository models: `models/*.json` (committed to Git)
- Browser models: IndexedDB via `ModelStorage` class
- Both storage methods use the same serialization format
