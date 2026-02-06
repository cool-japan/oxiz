//! Optimization Algorithms for Training

use super::tensor::{Tensor, TensorOps};
use std::collections::HashMap;

/// Optimizer trait
pub trait Optimizer: Send + Sync {
    /// Update parameters given gradients
    fn step(&mut self, param_id: usize, param: &mut Tensor, gradient: &Tensor);

    /// Get learning rate
    fn learning_rate(&self) -> f64;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f64);

    /// Reset optimizer state
    fn reset(&mut self);

    /// Get number of steps taken
    fn num_steps(&self) -> usize;
}

/// Stochastic Gradient Descent (with momentum)
#[derive(Debug, Clone)]
pub struct SGD {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Velocity for momentum
    velocity: HashMap<usize, Tensor>,
    /// Number of steps
    steps: usize,
}

impl SGD {
    /// Create new SGD optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity: HashMap::new(),
            steps: 0,
        }
    }

    /// Create SGD with momentum
    pub fn with_momentum(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay: 0.0,
            velocity: HashMap::new(),
            steps: 0,
        }
    }

    /// Create SGD with weight decay
    pub fn with_weight_decay(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay,
            velocity: HashMap::new(),
            steps: 0,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, param_id: usize, param: &mut Tensor, gradient: &Tensor) {
        self.steps += 1;

        // Apply weight decay if configured
        let mut grad = gradient.clone();
        if self.weight_decay > 0.0 {
            // gradient = gradient + weight_decay * param
            let decay = param.scale(self.weight_decay);
            grad = grad
                .add(&decay)
                .expect("Dimension mismatch in weight decay");
        }

        if self.momentum > 0.0 {
            // Momentum: v = momentum * v + gradient
            let v = self
                .velocity
                .entry(param_id)
                .or_insert_with(|| Tensor::zeros(param.shape()));

            *v = v
                .scale(self.momentum)
                .add(&grad)
                .expect("Dimension mismatch in momentum");

            // Update: param = param - lr * v
            let update = v.scale(self.learning_rate);
            *param = param
                .sub(&update)
                .expect("Dimension mismatch in param update");
        } else {
            // Standard SGD: param = param - lr * gradient
            let update = grad.scale(self.learning_rate);
            *param = param
                .sub(&update)
                .expect("Dimension mismatch in param update");
        }
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.velocity.clear();
        self.steps = 0;
    }

    fn num_steps(&self) -> usize {
        self.steps
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate
    pub learning_rate: f64,
    /// Beta1 for first moment
    pub beta1: f64,
    /// Beta2 for second moment
    pub beta2: f64,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Weight decay
    pub weight_decay: f64,
    /// First moment (mean of gradients)
    m: HashMap<usize, Tensor>,
    /// Second moment (mean of squared gradients)
    v: HashMap<usize, Tensor>,
    /// Number of steps
    steps: usize,
}

impl Adam {
    /// Create new Adam optimizer with default parameters
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            steps: 0,
        }
    }

    /// Create Adam with custom beta values
    pub fn with_betas(learning_rate: f64, beta1: f64, beta2: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            steps: 0,
        }
    }

    /// Create AdamW (Adam with decoupled weight decay)
    pub fn adamw(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            steps: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, param_id: usize, param: &mut Tensor, gradient: &Tensor) {
        self.steps += 1;

        // Get or initialize moments
        let m = self
            .m
            .entry(param_id)
            .or_insert_with(|| Tensor::zeros(param.shape()));
        let v = self
            .v
            .entry(param_id)
            .or_insert_with(|| Tensor::zeros(param.shape()));

        // Update biased first moment: m = beta1 * m + (1 - beta1) * gradient
        let grad_contrib = gradient.scale(1.0 - self.beta1);
        *m = m
            .scale(self.beta1)
            .add(&grad_contrib)
            .expect("Dimension mismatch in m update");

        // Update biased second moment: v = beta2 * v + (1 - beta2) * gradient^2
        let grad_squared = gradient
            .mul(gradient)
            .expect("Dimension mismatch in gradient square");
        let grad_sq_contrib = grad_squared.scale(1.0 - self.beta2);
        *v = v
            .scale(self.beta2)
            .add(&grad_sq_contrib)
            .expect("Dimension mismatch in v update");

        // Bias correction
        let bias_correction1 = 1.0 - self.beta1.powi(self.steps as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.steps as i32);

        let m_hat = m.scale(1.0 / bias_correction1);
        let v_hat = v.scale(1.0 / bias_correction2);

        // Compute update: m_hat / (sqrt(v_hat) + epsilon)
        let mut update = Tensor::zeros(param.shape());
        for i in 0..param.data.len() {
            update.data[i] = m_hat.data[i] / (v_hat.data[i].sqrt() + self.epsilon);
        }

        // Apply weight decay (decoupled)
        if self.weight_decay > 0.0 {
            let decay = param.scale(self.weight_decay);
            *param = param
                .sub(&decay)
                .expect("Dimension mismatch in weight decay");
        }

        // Update parameters: param = param - lr * update
        let scaled_update = update.scale(self.learning_rate);
        *param = param
            .sub(&scaled_update)
            .expect("Dimension mismatch in param update");
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.steps = 0;
    }

    fn num_steps(&self) -> usize {
        self.steps
    }
}

/// AdaGrad optimizer (Adaptive Gradient)
#[derive(Debug, Clone)]
pub struct AdaGrad {
    /// Learning rate
    pub learning_rate: f64,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Accumulated squared gradients
    accumulated: HashMap<usize, Tensor>,
    /// Number of steps
    steps: usize,
}

impl AdaGrad {
    /// Create new AdaGrad optimizer
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            epsilon: 1e-8,
            accumulated: HashMap::new(),
            steps: 0,
        }
    }

    /// Create AdaGrad with custom epsilon
    pub fn with_epsilon(learning_rate: f64, epsilon: f64) -> Self {
        Self {
            learning_rate,
            epsilon,
            accumulated: HashMap::new(),
            steps: 0,
        }
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, param_id: usize, param: &mut Tensor, gradient: &Tensor) {
        self.steps += 1;

        // Get or initialize accumulated squared gradients
        let acc = self
            .accumulated
            .entry(param_id)
            .or_insert_with(|| Tensor::zeros(param.shape()));

        // Update accumulated: acc = acc + gradient^2
        let grad_squared = gradient
            .mul(gradient)
            .expect("Dimension mismatch in gradient square");
        *acc = acc
            .add(&grad_squared)
            .expect("Dimension mismatch in accumulated update");

        // Compute update: gradient / (sqrt(acc) + epsilon)
        let mut update = Tensor::zeros(param.shape());
        for i in 0..param.data.len() {
            update.data[i] = gradient.data[i] / (acc.data[i].sqrt() + self.epsilon);
        }

        // Update parameters: param = param - lr * update
        let scaled_update = update.scale(self.learning_rate);
        *param = param
            .sub(&scaled_update)
            .expect("Dimension mismatch in param update");
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.accumulated.clear();
        self.steps = 0;
    }

    fn num_steps(&self) -> usize {
        self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_basic() {
        let mut opt = SGD::new(0.1);
        let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let gradient = Tensor::from_slice(&[0.1, 0.2, 0.3]);

        opt.step(0, &mut param, &gradient);

        // param = param - lr * gradient
        // [1.0, 2.0, 3.0] - 0.1 * [0.1, 0.2, 0.3] = [0.99, 1.98, 2.97]
        assert!((param.data[0] - 0.99).abs() < 1e-10);
        assert!((param.data[1] - 1.98).abs() < 1e-10);
        assert!((param.data[2] - 2.97).abs() < 1e-10);

        assert_eq!(opt.num_steps(), 1);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut opt = SGD::with_momentum(0.1, 0.9);
        let mut param = Tensor::from_slice(&[1.0]);
        let gradient = Tensor::from_slice(&[0.1]);

        opt.step(0, &mut param, &gradient);
        let first_param = param.data[0];

        opt.step(0, &mut param, &gradient);
        let second_param = param.data[0];

        // With momentum, second step should move further
        assert!(first_param < 1.0);
        assert!(second_param < first_param);
    }

    #[test]
    fn test_sgd_learning_rate() {
        let mut opt = SGD::new(0.1);
        assert_eq!(opt.learning_rate(), 0.1);

        opt.set_learning_rate(0.01);
        assert_eq!(opt.learning_rate(), 0.01);
    }

    #[test]
    fn test_sgd_reset() {
        let mut opt = SGD::with_momentum(0.1, 0.9);
        let mut param = Tensor::from_slice(&[1.0]);
        let gradient = Tensor::from_slice(&[0.1]);

        opt.step(0, &mut param, &gradient);
        assert_eq!(opt.num_steps(), 1);

        opt.reset();
        assert_eq!(opt.num_steps(), 0);
        assert!(opt.velocity.is_empty());
    }

    #[test]
    fn test_adam_basic() {
        let mut opt = Adam::new(0.01);
        let mut param = Tensor::from_slice(&[1.0, 2.0]);
        let gradient = Tensor::from_slice(&[0.1, 0.2]);

        let initial = param.clone();
        opt.step(0, &mut param, &gradient);

        // Parameters should move in opposite direction of gradient
        assert!(param.data[0] < initial.data[0]);
        assert!(param.data[1] < initial.data[1]);

        assert_eq!(opt.num_steps(), 1);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let mut opt = Adam::new(0.01);
        let mut param = Tensor::from_slice(&[1.0]);
        let gradient = Tensor::from_slice(&[0.1]);

        for _ in 0..10 {
            opt.step(0, &mut param, &gradient);
        }

        assert_eq!(opt.num_steps(), 10);
        assert!(param.data[0] < 1.0); // Should have decreased
    }

    #[test]
    fn test_adam_betas() {
        let opt = Adam::with_betas(0.01, 0.9, 0.999);
        assert_eq!(opt.beta1, 0.9);
        assert_eq!(opt.beta2, 0.999);
    }

    #[test]
    fn test_adamw() {
        let opt = Adam::adamw(0.01, 0.01);
        assert_eq!(opt.weight_decay, 0.01);
    }

    #[test]
    fn test_adagrad_basic() {
        let mut opt = AdaGrad::new(0.1);
        let mut param = Tensor::from_slice(&[1.0, 2.0]);
        let gradient = Tensor::from_slice(&[0.1, 0.2]);

        let initial = param.clone();
        opt.step(0, &mut param, &gradient);

        // Parameters should move in opposite direction of gradient
        assert!(param.data[0] < initial.data[0]);
        assert!(param.data[1] < initial.data[1]);

        assert_eq!(opt.num_steps(), 1);
    }

    #[test]
    fn test_adagrad_adaptive() {
        let mut opt = AdaGrad::new(0.1);
        let mut param = Tensor::from_slice(&[1.0]);
        let gradient = Tensor::from_slice(&[0.1]);

        opt.step(0, &mut param, &gradient);
        let first_change = 1.0 - param.data[0];

        opt.step(0, &mut param, &gradient);
        let second_param = param.data[0];
        let second_change = (1.0 - first_change) - second_param;

        // AdaGrad should adapt: second step should be smaller
        assert!(second_change < first_change);
    }

    #[test]
    fn test_optimizer_trait() {
        fn test_optimizer<O: Optimizer>(mut opt: O) {
            let mut param = Tensor::from_slice(&[1.0]);
            let gradient = Tensor::from_slice(&[0.1]);

            opt.step(0, &mut param, &gradient);
            assert!(param.data[0] < 1.0);
        }

        test_optimizer(SGD::new(0.1));
        test_optimizer(Adam::new(0.01));
        test_optimizer(AdaGrad::new(0.1));
    }
}
