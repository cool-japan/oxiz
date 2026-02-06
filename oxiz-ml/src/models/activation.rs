//! Activation Functions for Neural Networks

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    /// Rectified Linear Unit
    ReLU,
    /// Leaky ReLU
    LeakyReLU,
    /// Sigmoid
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Linear (identity)
    Linear,
    /// Exponential Linear Unit
    ELU,
    /// Softplus
    Softplus,
}

impl Activation {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
            Activation::ELU => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Activation::Softplus => (1.0 + x.exp()).ln(),
        }
    }

    /// Compute derivative of activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Linear => 1.0,
            Activation::ELU => {
                if x > 0.0 {
                    1.0
                } else {
                    x.exp()
                }
            }
            Activation::Softplus => 1.0 / (1.0 + (-x).exp()),
        }
    }

    /// Apply activation to vector in-place
    pub fn apply_vec_inplace(&self, v: &mut [f64]) {
        for x in v {
            *x = self.apply(*x);
        }
    }

    /// Compute derivative for vector
    pub fn derivative_vec(&self, v: &[f64]) -> Vec<f64> {
        v.iter().map(|&x| self.derivative(x)).collect()
    }
}

/// Trait for activation functions
pub trait ActivationFn: Send + Sync {
    /// Apply activation
    fn apply(&self, x: f64) -> f64;

    /// Compute derivative
    fn derivative(&self, x: f64) -> f64;

    /// Apply to vector
    fn apply_vec(&self, v: &[f64]) -> Vec<f64> {
        v.iter().map(|&x| self.apply(x)).collect()
    }

    /// Apply to vector in-place
    fn apply_vec_inplace(&self, v: &mut [f64]) {
        for x in v {
            *x = self.apply(*x);
        }
    }
}

impl ActivationFn for Activation {
    fn apply(&self, x: f64) -> f64 {
        Activation::apply(self, x)
    }

    fn derivative(&self, x: f64) -> f64 {
        Activation::derivative(self, x)
    }

    fn apply_vec_inplace(&self, v: &mut [f64]) {
        Activation::apply_vec_inplace(self, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let act = Activation::ReLU;
        assert_eq!(act.apply(5.0), 5.0);
        assert_eq!(act.apply(-3.0), 0.0);
        assert_eq!(act.apply(0.0), 0.0);

        assert_eq!(act.derivative(5.0), 1.0);
        assert_eq!(act.derivative(-3.0), 0.0);
    }

    #[test]
    fn test_leaky_relu() {
        let act = Activation::LeakyReLU;
        assert_eq!(act.apply(5.0), 5.0);
        assert_eq!(act.apply(-2.0), -0.02);

        assert_eq!(act.derivative(5.0), 1.0);
        assert_eq!(act.derivative(-2.0), 0.01);
    }

    #[test]
    fn test_sigmoid() {
        let act = Activation::Sigmoid;
        let result = act.apply(0.0);
        assert!((result - 0.5).abs() < 1e-10);

        let result_pos = act.apply(10.0);
        assert!(result_pos > 0.99);

        let result_neg = act.apply(-10.0);
        assert!(result_neg < 0.01);
    }

    #[test]
    fn test_tanh() {
        let act = Activation::Tanh;
        let result = act.apply(0.0);
        assert!((result - 0.0).abs() < 1e-10);

        let result_pos = act.apply(10.0);
        assert!(result_pos > 0.99);

        let result_neg = act.apply(-10.0);
        assert!(result_neg < -0.99);
    }

    #[test]
    fn test_linear() {
        let act = Activation::Linear;
        assert_eq!(act.apply(5.0), 5.0);
        assert_eq!(act.apply(-3.0), -3.0);
        assert_eq!(act.derivative(100.0), 1.0);
    }

    #[test]
    fn test_elu() {
        let act = Activation::ELU;
        assert_eq!(act.apply(5.0), 5.0);
        assert!(act.apply(-1.0) < 0.0);
        assert!(act.apply(-1.0) > -1.0);
    }

    #[test]
    fn test_softplus() {
        let act = Activation::Softplus;
        let result = act.apply(0.0);
        assert!((result - (2.0_f64).ln()).abs() < 1e-10);

        let large = act.apply(10.0);
        assert!((large - 10.0).abs() < 0.01); // softplus(x) ≈ x for large x
    }

    #[test]
    fn test_sigmoid_derivative() {
        let act = Activation::Sigmoid;
        let deriv = act.derivative(0.0);
        assert!((deriv - 0.25).abs() < 1e-10); // sigmoid'(0) = 0.25
    }

    #[test]
    fn test_tanh_derivative() {
        let act = Activation::Tanh;
        let deriv = act.derivative(0.0);
        assert!((deriv - 1.0).abs() < 1e-10); // tanh'(0) = 1
    }

    #[test]
    fn test_apply_vec_inplace() {
        let act = Activation::ReLU;
        let mut v = vec![-1.0, 0.0, 1.0, 2.0];
        act.apply_vec_inplace(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_derivative_vec() {
        let act = Activation::ReLU;
        let v = vec![-1.0, 0.0, 1.0, 2.0];
        let derivs = act.derivative_vec(&v);
        assert_eq!(derivs, vec![0.0, 0.0, 1.0, 1.0]);
    }
}
