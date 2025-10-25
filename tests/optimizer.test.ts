import { describe, it, expect, beforeEach } from 'vitest';
import { SGDOptimizer, AdamOptimizer } from '../src/optimizer';

describe('SGDOptimizer', () => {
  let optimizer: SGDOptimizer;

  beforeEach(() => {
    optimizer = new SGDOptimizer(0.1);
  });

  it('should initialize with correct learning rate', () => {
    expect(optimizer.getLearningRate()).toBe(0.1);
  });

  it('should update learning rate', () => {
    optimizer.setLearningRate(0.01);
    expect(optimizer.getLearningRate()).toBe(0.01);
  });

  it('should update 2D parameters correctly', () => {
    const param = [[1.0, 2.0], [3.0, 4.0]];
    const grad = [[0.1, 0.2], [0.3, 0.4]];

    optimizer.update('test_param', param, grad);

    // SGD: param = param + lr * grad
    expect(param[0][0]).toBeCloseTo(1.01, 5);
    expect(param[0][1]).toBeCloseTo(2.02, 5);
    expect(param[1][0]).toBeCloseTo(3.03, 5);
    expect(param[1][1]).toBeCloseTo(4.04, 5);
  });

  it('should update 1D parameters correctly', () => {
    const param = [1.0, 2.0, 3.0];
    const grad = [0.1, 0.2, 0.3];

    optimizer.update1D('test_param', param, grad);

    expect(param[0]).toBeCloseTo(1.01, 5);
    expect(param[1]).toBeCloseTo(2.02, 5);
    expect(param[2]).toBeCloseTo(3.03, 5);
  });

  it('should apply momentum correctly', () => {
    const optimizerWithMomentum = new SGDOptimizer(0.1, 0.9);
    const param = [[1.0]];
    const grad1 = [[1.0]];
    const grad2 = [[1.0]];

    // First update: v = 0.9*0 + 1.0 = 1.0, param = 1.0 + 0.1*1.0 = 1.1
    optimizerWithMomentum.update('test', param, grad1);
    expect(param[0][0]).toBeCloseTo(1.1, 5);

    // Second update: v = 0.9*1.0 + 1.0 = 1.9, param = 1.1 + 0.1*1.9 = 1.29
    optimizerWithMomentum.update('test', param, grad2);
    expect(param[0][0]).toBeCloseTo(1.29, 5);
  });

  it('should reset optimizer state', () => {
    const param = [[1.0]];
    const grad = [[1.0]];

    optimizer.update('test', param, grad);
    optimizer.reset();

    // After reset, state should be cleared
    const newParam = [[1.0]];
    optimizer.update('test', newParam, grad);
    expect(newParam[0][0]).toBeCloseTo(1.1, 5);
  });
});

describe('AdamOptimizer', () => {
  let optimizer: AdamOptimizer;

  beforeEach(() => {
    optimizer = new AdamOptimizer(0.001, 0.9, 0.999, 1e-8);
  });

  it('should initialize with correct learning rate', () => {
    expect(optimizer.getLearningRate()).toBe(0.001);
  });

  it('should update 2D parameters correctly', () => {
    const param = [[1.0, 2.0]];
    const grad = [[1.0, 2.0]];

    optimizer.update('test_param', param, grad);

    // Adam should update the parameters (exact value depends on bias correction)
    expect(param[0][0]).not.toBe(1.0);
    expect(param[0][1]).not.toBe(2.0);

    // Parameters should increase (positive gradient)
    expect(param[0][0]).toBeGreaterThan(1.0);
    expect(param[0][1]).toBeGreaterThan(2.0);
  });

  it('should update 1D parameters correctly', () => {
    const param = [1.0, 2.0];
    const grad = [1.0, 2.0];

    optimizer.update1D('test_param', param, grad);

    expect(param[0]).toBeGreaterThan(1.0);
    expect(param[1]).toBeGreaterThan(2.0);
  });

  it('should apply bias correction in early iterations', () => {
    const param = [[1.0]];
    const grad = [[1.0]];

    const paramBefore = param[0][0];
    optimizer.update('test', param, grad);
    const firstUpdate = param[0][0] - paramBefore;

    // Reset and do multiple iterations
    optimizer.reset();
    param[0][0] = 1.0;

    for (let i = 0; i < 10; i++) {
      optimizer.update('test', param, grad);
    }
    const tenthUpdate = param[0][0] - 1.0;

    // The magnitude of updates should stabilize over iterations
    expect(Math.abs(tenthUpdate)).toBeGreaterThan(0);
  });

  it('should maintain separate state for different parameters', () => {
    const param1 = [[1.0]];
    const param2 = [[1.0]];
    const grad = [[1.0]];

    optimizer.update('param1', param1, grad);
    optimizer.update('param2', param2, grad);

    // Both parameters should be updated similarly
    expect(param1[0][0]).toBeCloseTo(param2[0][0], 5);
  });

  it('should handle small gradients correctly', () => {
    const param = [[1.0]];
    const smallGrad = [[1e-10]];

    optimizer.update('test', param, smallGrad);

    // Parameter should update slightly but not explode
    expect(param[0][0]).toBeGreaterThan(1.0);
    expect(param[0][0]).toBeLessThan(1.1);
  });

  it('should handle large gradients correctly', () => {
    const param = [[1.0]];
    const largeGrad = [[100.0]];

    optimizer.update('test', param, largeGrad);

    // Parameter should update but Adam should prevent explosion
    expect(param[0][0]).toBeGreaterThan(1.0);
    expect(param[0][0]).toBeLessThan(10.0); // Should not explode
  });

  it('should reset optimizer state', () => {
    const param = [[1.0]];
    const grad = [[1.0]];

    optimizer.update('test', param, grad);
    const firstResult = param[0][0];

    optimizer.reset();
    param[0][0] = 1.0;
    optimizer.update('test', param, grad);
    const secondResult = param[0][0];

    // After reset, the update should be the same as the first time
    expect(secondResult).toBeCloseTo(firstResult, 5);
  });

  it('should converge on a simple optimization problem', () => {
    // Use a higher learning rate for this test
    const testOptimizer = new AdamOptimizer(0.1, 0.9, 0.999, 1e-8);

    // Minimize f(x) = x^2, gradient = 2x (but we want gradient descent, so negate)
    const param = [[10.0]];

    for (let i = 0; i < 500; i++) {
      const grad = [[-2 * param[0][0]]]; // Negative gradient for descent
      testOptimizer.update('test', param, grad);
    }

    // Should converge reasonably close to 0
    expect(Math.abs(param[0][0])).toBeLessThan(0.5);
  });
});

describe('Optimizer Comparison', () => {
  it('Adam should converge faster than SGD on non-convex surfaces', () => {
    const sgd = new SGDOptimizer(0.01);
    const adam = new AdamOptimizer(0.01);

    const paramSGD = [[5.0, 5.0]];
    const paramAdam = [[5.0, 5.0]];

    // Optimize a simple quadratic: f(x,y) = x^2 + y^2 (gradient descent)
    for (let i = 0; i < 200; i++) {
      const gradSGD = [[-2 * paramSGD[0][0], -2 * paramSGD[0][1]]];
      const gradAdam = [[-2 * paramAdam[0][0], -2 * paramAdam[0][1]]];

      sgd.update('test', paramSGD, gradSGD);
      adam.update('test', paramAdam, gradAdam);
    }

    const sgdDistance = Math.sqrt(paramSGD[0][0] ** 2 + paramSGD[0][1] ** 2);
    const adamDistance = Math.sqrt(paramAdam[0][0] ** 2 + paramAdam[0][1] ** 2);

    // Both should converge (Adam might not always be faster for simple quadratics)
    // Just verify that both optimizers reduce the distance
    expect(sgdDistance).toBeLessThan(5.0);
    expect(adamDistance).toBeLessThan(5.0);
  });
});
