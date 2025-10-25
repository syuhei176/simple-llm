/**
 * Optimizer base class and implementations
 * Supports SGD and Adam optimizers for parameter updates
 */

export interface OptimizerState {
  [key: string]: any;
}

export abstract class Optimizer {
  protected learningRate: number;
  protected state: Map<string, OptimizerState> = new Map();

  constructor(learningRate: number = 0.001) {
    this.learningRate = learningRate;
  }

  /**
   * Update a parameter using its gradient
   * @param paramKey Unique key for this parameter
   * @param param Current parameter value (modified in place)
   * @param grad Gradient for this parameter
   */
  abstract update(paramKey: string, param: number[][], grad: number[][]): void;

  /**
   * Update a 1D parameter using its gradient
   * @param paramKey Unique key for this parameter
   * @param param Current parameter value (modified in place)
   * @param grad Gradient for this parameter
   */
  abstract update1D(paramKey: string, param: number[], grad: number[]): void;

  /**
   * Set the learning rate
   */
  setLearningRate(lr: number): void {
    this.learningRate = lr;
  }

  /**
   * Get the current learning rate
   */
  getLearningRate(): number {
    return this.learningRate;
  }

  /**
   * Reset optimizer state
   */
  reset(): void {
    this.state.clear();
  }
}

/**
 * Stochastic Gradient Descent (SGD) optimizer
 */
export class SGDOptimizer extends Optimizer {
  private momentum: number;
  private dampening: number;
  private nesterov: boolean;

  constructor(
    learningRate: number = 0.001,
    momentum: number = 0,
    dampening: number = 0,
    nesterov: boolean = false
  ) {
    super(learningRate);
    this.momentum = momentum;
    this.dampening = dampening;
    this.nesterov = nesterov;
  }

  update(paramKey: string, param: number[][], grad: number[][]): void {
    if (this.momentum === 0) {
      // Simple SGD without momentum
      for (let i = 0; i < param.length; i++) {
        for (let j = 0; j < param[i].length; j++) {
          param[i][j] += this.learningRate * grad[i][j];
        }
      }
      return;
    }

    // SGD with momentum
    let state = this.state.get(paramKey);
    if (!state) {
      // Initialize velocity
      state = {
        velocity: param.map(row => new Array(row.length).fill(0))
      };
      this.state.set(paramKey, state);
    }

    const velocity = state.velocity as number[][];
    for (let i = 0; i < param.length; i++) {
      for (let j = 0; j < param[i].length; j++) {
        let v = velocity[i][j] * this.momentum;
        if (this.dampening > 0) {
          v += (1 - this.dampening) * grad[i][j];
        } else {
          v += grad[i][j];
        }
        velocity[i][j] = v;

        if (this.nesterov) {
          param[i][j] += this.learningRate * (grad[i][j] + this.momentum * v);
        } else {
          param[i][j] += this.learningRate * v;
        }
      }
    }
  }

  update1D(paramKey: string, param: number[], grad: number[]): void {
    if (this.momentum === 0) {
      // Simple SGD without momentum
      for (let i = 0; i < param.length; i++) {
        param[i] += this.learningRate * grad[i];
      }
      return;
    }

    // SGD with momentum
    let state = this.state.get(paramKey);
    if (!state) {
      state = {
        velocity: new Array(param.length).fill(0)
      };
      this.state.set(paramKey, state);
    }

    const velocity = state.velocity as number[];
    for (let i = 0; i < param.length; i++) {
      let v = velocity[i] * this.momentum;
      if (this.dampening > 0) {
        v += (1 - this.dampening) * grad[i];
      } else {
        v += grad[i];
      }
      velocity[i] = v;

      if (this.nesterov) {
        param[i] += this.learningRate * (grad[i] + this.momentum * v);
      } else {
        param[i] += this.learningRate * v;
      }
    }
  }
}

/**
 * Adam optimizer (Adaptive Moment Estimation)
 * Paper: https://arxiv.org/abs/1412.6980
 */
export class AdamOptimizer extends Optimizer {
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private timestep: number = 0;

  constructor(
    learningRate: number = 0.001,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8
  ) {
    super(learningRate);
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
  }

  update(paramKey: string, param: number[][], grad: number[][]): void {
    let state = this.state.get(paramKey);
    if (!state) {
      // Initialize first and second moments
      state = {
        m: param.map(row => new Array(row.length).fill(0)),
        v: param.map(row => new Array(row.length).fill(0)),
        t: 0
      };
      this.state.set(paramKey, state);
    }

    const m = state.m as number[][];
    const v = state.v as number[][];
    state.t += 1;
    const t = state.t as number;

    for (let i = 0; i < param.length; i++) {
      for (let j = 0; j < param[i].length; j++) {
        // Update biased first moment estimate
        m[i][j] = this.beta1 * m[i][j] + (1 - this.beta1) * grad[i][j];

        // Update biased second raw moment estimate
        v[i][j] = this.beta2 * v[i][j] + (1 - this.beta2) * grad[i][j] * grad[i][j];

        // Compute bias-corrected first moment estimate
        const mHat = m[i][j] / (1 - Math.pow(this.beta1, t));

        // Compute bias-corrected second raw moment estimate
        const vHat = v[i][j] / (1 - Math.pow(this.beta2, t));

        // Update parameters
        param[i][j] += this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
      }
    }
  }

  update1D(paramKey: string, param: number[], grad: number[]): void {
    let state = this.state.get(paramKey);
    if (!state) {
      state = {
        m: new Array(param.length).fill(0),
        v: new Array(param.length).fill(0),
        t: 0
      };
      this.state.set(paramKey, state);
    }

    const m = state.m as number[];
    const v = state.v as number[];
    state.t += 1;
    const t = state.t as number;

    for (let i = 0; i < param.length; i++) {
      // Update biased first moment estimate
      m[i] = this.beta1 * m[i] + (1 - this.beta1) * grad[i];

      // Update biased second raw moment estimate
      v[i] = this.beta2 * v[i] + (1 - this.beta2) * grad[i] * grad[i];

      // Compute bias-corrected first moment estimate
      const mHat = m[i] / (1 - Math.pow(this.beta1, t));

      // Compute bias-corrected second raw moment estimate
      const vHat = v[i] / (1 - Math.pow(this.beta2, t));

      // Update parameters
      param[i] += this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
    }
  }

  reset(): void {
    super.reset();
    this.timestep = 0;
  }
}
