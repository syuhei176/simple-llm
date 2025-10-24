# Simple LLM

A minimal Transformer-based language model built from scratch in TypeScript.

## Features

- **Self-Attention Mechanism**: Implements Query, Key, Value matrices for proper self-attention
- **Positional Encoding**: Adds position information to token embeddings
- **Layer Normalization**: Stabilizes training with normalization after each sub-layer
- **Multiple Transformer Layers**: 3 stacked transformer layers for better representation
- **Embedding Layer**: Trainable token embeddings with Xavier initialization (64 dimensions)
- **Output Layer**: Linear transformation from embedding dimension to vocabulary size
- **Rich Training Data**: 70+ diverse conversation pairs
- **Backpropagation**: Proper gradient computation and weight updates
- **Web Interface**: Interactive demo that runs entirely in the browser

## Architecture

```
Input Tokens
    ↓
Embedding Layer (vocab_size → 64 dims)
    ↓
Positional Encoding (adds position info)
    ↓
Transformer Layer 1:
  ├─ Self-Attention (Q, K, V)
  ├─ Residual Connection + Layer Norm
  ├─ Feed-Forward Network
  └─ Residual Connection + Layer Norm
    ↓
Transformer Layer 2 (same structure)
    ↓
Transformer Layer 3 (same structure)
    ↓
Output Layer (64 dims → vocab_size)
    ↓
Softmax → Predicted Tokens
```

## Installation

```bash
# Install dependencies
pnpm install

# Build for web
npm run build:web

# Run development server
npm run dev
```

## Usage

### Web Demo

1. Build the project: `npm run build:web`
2. Open `docs/index.html` in your browser
3. Click "Train Model" to train the language model
4. Chat with the AI after training is complete

### Command Line

```bash
# Run the CLI version
npx ts-node src/index.ts
```

## Training Data

The model is trained on 70+ diverse conversation pairs covering:

- **Greetings**: "hello" → "hi there", "good morning" → "good morning to you"
- **Questions**: "how are you" → "I am doing well thank you", "what is your name" → "I am a simple AI assistant"
- **Colors**: "colors" → "red blue green yellow orange purple pink"
- **Animals**: "animals" → "cat dog bird fish elephant tiger bear"
- **Emotions**: "are you happy" → "yes I am happy to chat"
- **And many more categories**: food, places, time, hobbies, family, etc.

### Converting Long Texts to Training Data

The project includes a **sliding window** utility to convert long texts into training data automatically. This is highly efficient for autoregressive language models.

#### Basic Usage

```typescript
import { convertTextToTrainingData } from './src/training-data';

const longText = "the cat sat on the mat and the dog sat on the rug";

// Convert with window size 5 and stride 1
const trainingData = convertTextToTrainingData(longText, 5, 1);
// Generates:
// { input: "the cat sat on the", target: "cat sat on the mat" }
// { input: "cat sat on the mat", target: "sat on the mat and" }
// ...
```

#### Parameters

- **windowSize**: Number of words in each window (e.g., 3-10)
- **stride**: How many words to move the window (default: 1)
  - stride=1: Maximum data efficiency, high overlap
  - stride=windowSize/2: Balanced overlap
  - stride=windowSize: No overlap

#### Multiple Texts

```typescript
import { convertMultipleTextsToTrainingData } from './src/training-data';

const texts = [
  "first document with some text",
  "second document with more content",
];

const data = convertMultipleTextsToTrainingData(texts, 5, 2);
```

#### Recommended Settings

- **Small data (100-500 words)**: windowSize=3-5, stride=1
- **Medium data (500-2000 words)**: windowSize=5-7, stride=2
- **Large data (2000+ words)**: windowSize=7-10, stride=3

See `examples/sliding-window-example.ts` for complete examples.

## GitHub Pages Deployment

This project is configured to deploy to GitHub Pages using the `docs` folder.

### Setup

1. Go to your repository Settings
2. Navigate to Pages section
3. Under "Source", select "Deploy from a branch"
4. Select branch: `main` (or your default branch)
5. Select folder: `/docs`
6. Click Save

Your site will be available at: `https://<username>.github.io/<repository-name>/`

### Manual Deployment

```bash
# Build the web bundle
npm run build:web

# Commit and push
git add docs/
git commit -m "Deploy to GitHub Pages"
git push origin main
```

## Project Structure

```
simple-llm/
├── src/
│   ├── llm/                    # Main LLM orchestrator
│   ├── tokenizer/              # Simple word tokenizer
│   ├── embedding/              # Embedding layer
│   ├── transformer/            # Self-attention transformer with layer norm
│   ├── output/                 # Output layer
│   ├── positional-encoding.ts  # Positional encoding utilities
│   ├── layer-norm.ts           # Layer normalization
│   ├── training-data.ts        # Training data & sliding window utilities
│   ├── transpose.ts            # Matrix utility
│   ├── index.ts                # CLI entry point
│   └── web.ts                  # Web entry point
├── examples/
│   └── sliding-window-example.ts  # Sliding window usage examples
├── docs/              # GitHub Pages directory
│   ├── index.html     # Web interface
│   └── bundle.js      # Compiled bundle
├── webpack.config.js  # Webpack configuration
├── tsconfig.json      # TypeScript configuration
└── package.json       # Project configuration
```

## Technical Details

### Model Configuration
- **Embedding Dimension**: 64
- **Transformer Layers**: 3
- **Vocabulary Size**: ~200 words (dynamically generated)
- **Training Data**: 70+ conversation pairs

### Training
- Optimizer: Simple gradient descent
- Learning rate: 0.01
- Loss function: Cross-entropy
- Epochs: 100

### Self-Attention
```
Q = input × Wq
K = input × Wk
V = input × Wv

Attention(Q, K, V) = softmax(Q·K^T / √d) · V
```

### Positional Encoding
```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### Improvements Made

This implementation includes several improvements over basic neural networks:

1. **Proper Attention**: Uses Q, K, V matrices instead of simple element-wise operations
2. **Positional Encoding**: Adds position information to embeddings
3. **Layer Normalization**: Stabilizes training with normalization layers
4. **Multiple Transformer Layers**: 3 stacked layers for deeper representation
5. **Increased Embedding Dimension**: 64-dimensional embeddings (up from 16)
6. **Rich Training Data**: 70+ diverse examples (up from 5)
7. **Output Layer**: Correctly maps from embedding space to vocabulary space
8. **Xavier Initialization**: Better weight initialization for training stability
9. **Residual Connections**: Skip connections in each transformer layer
10. **Gradient Propagation**: Proper backpropagation through all layers

## Limitations

This is an educational implementation with simplifications:

- Only 3 transformer layers (real LLMs have 12-96+ layers)
- Simple tokenizer (word-level, no subwords like BPE or WordPiece)
- No multi-head attention (uses single attention head)
- Small vocabulary size (~200 words)
- Simple gradient descent (no Adam optimizer)
- Simplified gradient computation (not fully accurate backprop)
- Limited training data (70+ pairs vs millions in production)
- No dropout or other regularization techniques

## License

ISC

## Contributing

This is an educational project. Feel free to fork and experiment!
