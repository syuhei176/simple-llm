# Simple LLM

A minimal Transformer-based language model built from scratch in TypeScript.

## Features

- **Self-Attention Mechanism**: Implements Query, Key, Value matrices for proper self-attention
- **Embedding Layer**: Trainable token embeddings with Xavier initialization
- **Output Layer**: Linear transformation from embedding dimension to vocabulary size
- **Backpropagation**: Proper gradient computation and weight updates
- **Web Interface**: Interactive demo that runs entirely in the browser

## Architecture

```
Input Tokens
    ↓
Embedding Layer (vocab_size → embedding_dim)
    ↓
Transformer Layer (Self-Attention + Feed-Forward)
    ↓
Output Layer (embedding_dim → vocab_size)
    ↓
Predicted Tokens
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

The model is trained on simple input-output pairs:

- "hello world" → "I am AI"
- "color" → "red blue green yellow"
- "how are you" → "thank you and you?"
- "hello" → "how are you"
- "animals" → "cat dog bird fish"

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
│   ├── llm/           # Main LLM orchestrator
│   ├── tokenizer/     # Simple word tokenizer
│   ├── embedding/     # Embedding layer
│   ├── transformer/   # Self-attention transformer
│   ├── output/        # Output layer
│   ├── transpose.ts   # Matrix utility
│   ├── index.ts       # CLI entry point
│   └── web.ts         # Web entry point
├── docs/              # GitHub Pages directory
│   ├── index.html     # Web interface
│   └── bundle.js      # Compiled bundle
├── webpack.config.js  # Webpack configuration
├── tsconfig.json      # TypeScript configuration
└── package.json       # Project configuration
```

## Technical Details

### Embedding Dimension
Default: 16 (configurable)

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

### Improvements Made

This implementation includes several improvements over basic neural networks:

1. **Proper Attention**: Uses Q, K, V matrices instead of simple element-wise operations
2. **Output Layer**: Correctly maps from embedding space to vocabulary space
3. **Xavier Initialization**: Better weight initialization for training stability
4. **Residual Connections**: Skip connections in the transformer layer
5. **Gradient Propagation**: Proper backpropagation through all layers

## Limitations

This is an educational implementation with simplifications:

- Single transformer layer (real LLMs have many)
- Small embedding dimension
- Simple tokenizer (word-level, no subwords)
- No positional encoding
- Limited training data
- Simplified gradient computation

## License

ISC

## Contributing

This is an educational project. Feel free to fork and experiment!
