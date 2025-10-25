import { Optimizer } from '../optimizer';

/**
 * Multi-Head Attention implementation
 *
 * This class implements multi-head self-attention mechanism with:
 * - Multiple attention heads (each with independent Q/K/V projections)
 * - Parallel computation across heads
 * - Output projection to combine head outputs
 * - Full backward pass for training
 */
export class MultiHeadAttention {
  embeddingDim: number;
  numHeads: number;
  headDim: number;

  // Weight matrices for each head: [numHeads][embeddingDim][headDim]
  wq: number[][][];
  wk: number[][][];
  wv: number[][][];

  // Output projection: [embeddingDim][embeddingDim]
  wo: number[][];

  // Gradient accumulators
  gradWq: number[][][];
  gradWk: number[][][];
  gradWv: number[][][];
  gradWo: number[][];

  // Cache for backward pass
  lastInput: number[][] = [];
  lastQ: number[][][] = []; // [numHeads][seqLen][headDim]
  lastK: number[][][] = [];
  lastV: number[][][] = [];
  lastScores: number[][][] = []; // [numHeads][seqLen][seqLen]
  lastAttentionWeights: number[][][] = [];
  lastHeadOutputs: number[][][] = []; // [numHeads][seqLen][headDim]
  lastConcatenated: number[][] = []; // [seqLen][embeddingDim]

  learningRate: number = 0.001;

  constructor(embeddingDim: number, numHeads: number = 4) {
    if (embeddingDim % numHeads !== 0) {
      throw new Error(`embeddingDim (${embeddingDim}) must be divisible by numHeads (${numHeads})`);
    }

    this.embeddingDim = embeddingDim;
    this.numHeads = numHeads;
    this.headDim = embeddingDim / numHeads;

    // Xavier initialization for each head
    const scale = Math.sqrt(1.0 / embeddingDim);

    // Initialize Q, K, V weights for each head
    this.wq = this.initHeadWeights(numHeads, embeddingDim, this.headDim, scale);
    this.wk = this.initHeadWeights(numHeads, embeddingDim, this.headDim, scale);
    this.wv = this.initHeadWeights(numHeads, embeddingDim, this.headDim, scale);

    // Initialize output projection
    this.wo = this.initWeights(embeddingDim, embeddingDim, scale);

    // Initialize gradients to zero
    this.gradWq = this.initHeadWeights(numHeads, embeddingDim, this.headDim, 0);
    this.gradWk = this.initHeadWeights(numHeads, embeddingDim, this.headDim, 0);
    this.gradWv = this.initHeadWeights(numHeads, embeddingDim, this.headDim, 0);
    this.gradWo = this.initWeights(embeddingDim, embeddingDim, 0);
  }

  private initWeights(rows: number, cols: number, scale: number): number[][] {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
    );
  }

  private initHeadWeights(numHeads: number, inputDim: number, outputDim: number, scale: number): number[][][] {
    return Array.from({ length: numHeads }, () =>
      this.initWeights(inputDim, outputDim, scale)
    );
  }

  // Vector-matrix multiplication
  private vecMatmul(vec: number[], mat: number[][]): number[] {
    const result = new Array(mat[0].length).fill(0);
    for (let j = 0; j < mat[0].length; j++) {
      for (let i = 0; i < vec.length; i++) {
        result[j] += vec[i] * mat[i][j];
      }
    }
    return result;
  }

  // Softmax function (per row)
  private softmax(matrix: number[][]): number[][] {
    return matrix.map(row => {
      const maxVal = Math.max(...row);
      const expScores = row.map(x => Math.exp(x - maxVal));
      const sumExp = expScores.reduce((a, b) => a + b, 1e-8);
      return expScores.map(x => x / sumExp);
    });
  }

  // Softmax backward
  private softmaxBackward(gradOutput: number[][], softmaxOutput: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < gradOutput.length; i++) {
      const s = softmaxOutput[i];
      const grad = gradOutput[i];

      // Jacobian matrix for softmax
      const sum = grad.reduce((acc, g, j) => acc + g * s[j], 0);
      result[i] = s.map((si, j) => si * (grad[j] - sum));
    }
    return result;
  }

  /**
   * Forward pass for multi-head attention
   * @param input [seqLen][embeddingDim]
   * @returns [seqLen][embeddingDim]
   */
  forward(input: number[][]): number[][] {
    this.lastInput = input.map(row => [...row]);
    const seqLen = input.length;

    // Initialize storage for head outputs
    this.lastQ = [];
    this.lastK = [];
    this.lastV = [];
    this.lastScores = [];
    this.lastAttentionWeights = [];
    this.lastHeadOutputs = [];

    const headOutputs: number[][][] = [];

    // Process each head independently
    for (let h = 0; h < this.numHeads; h++) {
      // Project input to Q, K, V for this head
      const Q = input.map(vec => this.vecMatmul(vec, this.wq[h])); // [seqLen][headDim]
      const K = input.map(vec => this.vecMatmul(vec, this.wk[h]));
      const V = input.map(vec => this.vecMatmul(vec, this.wv[h]));

      this.lastQ[h] = Q;
      this.lastK[h] = K;
      this.lastV[h] = V;

      // Compute attention scores: Q * K^T / sqrt(headDim)
      const scale = Math.sqrt(this.headDim);
      const scores: number[][] = [];

      for (let i = 0; i < seqLen; i++) {
        scores[i] = [];
        for (let j = 0; j < seqLen; j++) {
          let dot = 0;
          for (let k = 0; k < this.headDim; k++) {
            dot += Q[i][k] * K[j][k];
          }
          scores[i][j] = dot / scale;
        }
      }
      this.lastScores[h] = scores;

      // Apply softmax to get attention weights
      const attnWeights = this.softmax(scores);
      this.lastAttentionWeights[h] = attnWeights;

      // Compute attention output: attnWeights * V
      const headOutput: number[][] = [];
      for (let i = 0; i < seqLen; i++) {
        headOutput[i] = new Array(this.headDim).fill(0);
        for (let j = 0; j < seqLen; j++) {
          for (let k = 0; k < this.headDim; k++) {
            headOutput[i][k] += attnWeights[i][j] * V[j][k];
          }
        }
      }
      this.lastHeadOutputs[h] = headOutput;
      headOutputs[h] = headOutput;
    }

    // Concatenate head outputs: [seqLen][numHeads * headDim] = [seqLen][embeddingDim]
    const concatenated: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      concatenated[i] = [];
      for (let h = 0; h < this.numHeads; h++) {
        for (let d = 0; d < this.headDim; d++) {
          concatenated[i].push(headOutputs[h][i][d]);
        }
      }
    }
    this.lastConcatenated = concatenated;

    // Apply output projection
    const output = concatenated.map(vec => this.vecMatmul(vec, this.wo));
    return output;
  }

  /**
   * Backward pass for multi-head attention
   * @param gradOutput [seqLen][embeddingDim]
   * @returns [seqLen][embeddingDim]
   */
  backward(gradOutput: number[][]): number[][] {
    const seqLen = gradOutput.length;

    // Backward through output projection
    const gradConcatenated: number[][] = Array.from({ length: seqLen }, () =>
      new Array(this.embeddingDim).fill(0)
    );

    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const g = gradOutput[i][j];
        for (let k = 0; k < this.embeddingDim; k++) {
          // Gradient w.r.t. Wo
          this.gradWo[k][j] += g * this.lastConcatenated[i][k];
          // Gradient w.r.t. concatenated
          gradConcatenated[i][k] += g * this.wo[k][j];
        }
      }
    }

    // Split gradConcatenated back into head gradients
    const gradHeadOutputs: number[][][] = [];
    for (let h = 0; h < this.numHeads; h++) {
      gradHeadOutputs[h] = [];
      for (let i = 0; i < seqLen; i++) {
        gradHeadOutputs[h][i] = [];
        for (let d = 0; d < this.headDim; d++) {
          const idx = h * this.headDim + d;
          gradHeadOutputs[h][i][d] = gradConcatenated[i][idx];
        }
      }
    }

    // Initialize gradient w.r.t. input
    const gradInput: number[][] = Array.from({ length: seqLen }, () =>
      new Array(this.embeddingDim).fill(0)
    );

    // Backward through each head
    for (let h = 0; h < this.numHeads; h++) {
      const Q = this.lastQ[h];
      const K = this.lastK[h];
      const V = this.lastV[h];
      const attnWeights = this.lastAttentionWeights[h];
      const gradHeadOutput = gradHeadOutputs[h];

      // Backward: headOutput = attnWeights @ V
      const gradAttnWeights: number[][] = Array.from({ length: seqLen }, () =>
        new Array(seqLen).fill(0)
      );
      const gradV: number[][] = Array.from({ length: seqLen }, () =>
        new Array(this.headDim).fill(0)
      );

      for (let i = 0; i < seqLen; i++) {
        for (let k = 0; k < this.headDim; k++) {
          const g = gradHeadOutput[i][k];
          for (let j = 0; j < seqLen; j++) {
            gradAttnWeights[i][j] += g * V[j][k];
            gradV[j][k] += g * attnWeights[i][j];
          }
        }
      }

      // Backward: softmax
      const gradScores = this.softmaxBackward(gradAttnWeights, attnWeights);

      // Backward: scores = Q @ K^T / scale
      const scale = Math.sqrt(this.headDim);
      const gradScoresScaled = gradScores.map(row => row.map(val => val / scale));

      const gradQ: number[][] = Array.from({ length: seqLen }, () =>
        new Array(this.headDim).fill(0)
      );
      const gradK: number[][] = Array.from({ length: seqLen }, () =>
        new Array(this.headDim).fill(0)
      );

      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          const g = gradScoresScaled[i][j];
          for (let k = 0; k < this.headDim; k++) {
            gradQ[i][k] += g * K[j][k];
            gradK[j][k] += g * Q[i][k];
          }
        }
      }

      // Backward: Q, K, V projections
      // Q = input @ Wq
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < this.headDim; j++) {
          const gq = gradQ[i][j];
          const gk = gradK[i][j];
          const gv = gradV[i][j];

          for (let k = 0; k < this.embeddingDim; k++) {
            // Gradient w.r.t. Wq, Wk, Wv
            this.gradWq[h][k][j] += gq * this.lastInput[i][k];
            this.gradWk[h][k][j] += gk * this.lastInput[i][k];
            this.gradWv[h][k][j] += gv * this.lastInput[i][k];

            // Gradient w.r.t. input
            gradInput[i][k] += gq * this.wq[h][k][j];
            gradInput[i][k] += gk * this.wk[h][k][j];
            gradInput[i][k] += gv * this.wv[h][k][j];
          }
        }
      }
    }

    return gradInput;
  }

  /**
   * Reset gradients to zero
   */
  zeroGrad(): void {
    // Zero gradients for Q, K, V
    for (let h = 0; h < this.numHeads; h++) {
      for (let i = 0; i < this.embeddingDim; i++) {
        for (let j = 0; j < this.headDim; j++) {
          this.gradWq[h][i][j] = 0;
          this.gradWk[h][i][j] = 0;
          this.gradWv[h][i][j] = 0;
        }
      }
    }

    // Zero gradients for output projection
    for (let i = 0; i < this.embeddingDim; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        this.gradWo[i][j] = 0;
      }
    }
  }

  /**
   * Update parameters using accumulated gradients
   */
  updateParameters(): void {
    const clipValue = 5.0;

    // Update Q, K, V for each head
    for (let h = 0; h < this.numHeads; h++) {
      for (let i = 0; i < this.embeddingDim; i++) {
        for (let j = 0; j < this.headDim; j++) {
          // Clip and update Wq
          const clippedGradWq = Math.max(-clipValue, Math.min(clipValue, this.gradWq[h][i][j]));
          this.wq[h][i][j] += this.learningRate * clippedGradWq;

          // Clip and update Wk
          const clippedGradWk = Math.max(-clipValue, Math.min(clipValue, this.gradWk[h][i][j]));
          this.wk[h][i][j] += this.learningRate * clippedGradWk;

          // Clip and update Wv
          const clippedGradWv = Math.max(-clipValue, Math.min(clipValue, this.gradWv[h][i][j]));
          this.wv[h][i][j] += this.learningRate * clippedGradWv;
        }
      }
    }

    // Update output projection
    for (let i = 0; i < this.embeddingDim; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        const clippedGradWo = Math.max(-clipValue, Math.min(clipValue, this.gradWo[i][j]));
        this.wo[i][j] += this.learningRate * clippedGradWo;
      }
    }
  }

  /**
   * Update parameters using optimizer
   */
  updateWithOptimizer(optimizer: Optimizer, layerKey: string): void {
    const clipValue = 5.0;

    // Update Q, K, V for each head
    for (let h = 0; h < this.numHeads; h++) {
      // Clip gradients
      const clippedGradWq = this.gradWq[h].map(row =>
        row.map(g => Math.max(-clipValue, Math.min(clipValue, g)))
      );
      const clippedGradWk = this.gradWk[h].map(row =>
        row.map(g => Math.max(-clipValue, Math.min(clipValue, g)))
      );
      const clippedGradWv = this.gradWv[h].map(row =>
        row.map(g => Math.max(-clipValue, Math.min(clipValue, g)))
      );

      // Update using optimizer
      optimizer.update(`${layerKey}_wq_head${h}`, this.wq[h], clippedGradWq);
      optimizer.update(`${layerKey}_wk_head${h}`, this.wk[h], clippedGradWk);
      optimizer.update(`${layerKey}_wv_head${h}`, this.wv[h], clippedGradWv);
    }

    // Clip and update output projection
    const clippedGradWo = this.gradWo.map(row =>
      row.map(g => Math.max(-clipValue, Math.min(clipValue, g)))
    );
    optimizer.update(`${layerKey}_wo`, this.wo, clippedGradWo);
  }
}
