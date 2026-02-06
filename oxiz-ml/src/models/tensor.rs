//! Lightweight Tensor Operations
#![allow(clippy::needless_range_loop)] // Tensor indexing
//!
//! Provides basic tensor operations needed for neural networks.
//! Optimized for small tensors (typical feature sizes: 10-50 dimensions).

use serde::{Deserialize, Serialize};

use super::{ModelError, ModelResult};
use smallvec::SmallVec;

/// A simple tensor type backed by a flat vector
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    /// Flat data storage
    pub data: Vec<f64>,
    /// Shape of the tensor
    pub shape: SmallVec<[usize; 4]>,
}

impl Tensor {
    /// Create a new tensor with given shape, initialized to zero
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: SmallVec::from_slice(shape),
        }
    }

    /// Create a new tensor with given shape, initialized to ones
    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape: SmallVec::from_slice(shape),
        }
    }

    /// Create a tensor from data and shape
    pub fn from_vec(data: Vec<f64>, shape: &[usize]) -> ModelResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(ModelError::DimensionMismatch {
                expected: expected_size,
                got: data.len(),
            });
        }

        Ok(Self {
            data,
            shape: SmallVec::from_slice(shape),
        })
    }

    /// Create a 1D tensor from a slice
    pub fn from_slice(data: &[f64]) -> Self {
        Self {
            data: data.to_vec(),
            shape: SmallVec::from_slice(&[data.len()]),
        }
    }

    /// Create a random tensor with uniform distribution [min, max]
    pub fn random_uniform(shape: &[usize], min: f64, max: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let size = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| rng.random_range(min..max)).collect();

        Self {
            data,
            shape: SmallVec::from_slice(shape),
        }
    }

    /// Create a random tensor with normal distribution (mean=0, std=1)
    pub fn random_normal(shape: &[usize], mean: f64, std: f64) -> Self {
        use rand::Rng;

        let mut rng = rand::rng();
        let size = shape.iter().product();

        // Box-Muller transform for normal distribution
        let data: Vec<f64> = (0..size)
            .map(|_| {
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std * z
            })
            .collect();

        Self {
            data,
            shape: SmallVec::from_slice(shape),
        }
    }

    /// He initialization (for ReLU networks)
    pub fn he_init(shape: &[usize]) -> Self {
        if shape.len() < 2 {
            return Self::random_normal(shape, 0.0, 0.01);
        }

        let fan_in = shape[0];
        let std = (2.0 / fan_in as f64).sqrt();
        Self::random_normal(shape, 0.0, std)
    }

    /// Xavier/Glorot initialization (for tanh/sigmoid networks)
    pub fn xavier_init(shape: &[usize]) -> Self {
        if shape.len() < 2 {
            return Self::random_uniform(shape, -0.1, 0.1);
        }

        let fan_in = shape[0];
        let fan_out = shape[1];
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        Self::random_uniform(shape, -limit, limit)
    }

    /// Get total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Reshape tensor (must preserve total size)
    pub fn reshape(&mut self, new_shape: &[usize]) -> ModelResult<()> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(ModelError::DimensionMismatch {
                expected: self.data.len(),
                got: new_size,
            });
        }
        self.shape = SmallVec::from_slice(new_shape);
        Ok(())
    }

    /// Get element at index (flat indexing)
    pub fn get(&self, idx: usize) -> Option<f64> {
        self.data.get(idx).copied()
    }

    /// Set element at index (flat indexing)
    pub fn set(&mut self, idx: usize, value: f64) -> ModelResult<()> {
        if idx >= self.data.len() {
            return Err(ModelError::DimensionMismatch {
                expected: self.data.len(),
                got: idx,
            });
        }
        self.data[idx] = value;
        Ok(())
    }

    /// Fill tensor with a constant value
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Apply function element-wise
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Apply function element-wise in-place
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(f64) -> f64,
    {
        for x in &mut self.data {
            *x = f(*x);
        }
    }

    /// Check for NaN or Inf values
    pub fn has_nan_or_inf(&self) -> bool {
        self.data.iter().any(|&x| x.is_nan() || x.is_infinite())
    }

    /// Clip values to range [min, max]
    pub fn clip(&mut self, min: f64, max: f64) {
        for x in &mut self.data {
            *x = x.clamp(min, max);
        }
    }
}

/// Tensor operations trait
pub trait TensorOps {
    /// Element-wise addition
    fn add(&self, other: &Self) -> ModelResult<Self>
    where
        Self: Sized;

    /// Element-wise subtraction
    fn sub(&self, other: &Self) -> ModelResult<Self>
    where
        Self: Sized;

    /// Element-wise multiplication
    fn mul(&self, other: &Self) -> ModelResult<Self>
    where
        Self: Sized;

    /// Scalar multiplication
    fn scale(&self, scalar: f64) -> Self
    where
        Self: Sized;

    /// Dot product (1D tensors)
    fn dot(&self, other: &Self) -> ModelResult<f64>;

    /// Matrix-vector multiplication
    fn matmul_vec(&self, vec: &Self) -> ModelResult<Self>
    where
        Self: Sized;

    /// Sum all elements
    fn sum(&self) -> f64;

    /// Mean of all elements
    fn mean(&self) -> f64;

    /// Standard deviation
    fn std(&self) -> f64;

    /// L2 norm
    fn norm(&self) -> f64;
}

impl TensorOps for Tensor {
    fn add(&self, other: &Self) -> ModelResult<Self> {
        if self.data.len() != other.data.len() {
            return Err(ModelError::DimensionMismatch {
                expected: self.data.len(),
                got: other.data.len(),
            });
        }

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(Self {
            data,
            shape: self.shape.clone(),
        })
    }

    fn sub(&self, other: &Self) -> ModelResult<Self> {
        if self.data.len() != other.data.len() {
            return Err(ModelError::DimensionMismatch {
                expected: self.data.len(),
                got: other.data.len(),
            });
        }

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(Self {
            data,
            shape: self.shape.clone(),
        })
    }

    fn mul(&self, other: &Self) -> ModelResult<Self> {
        if self.data.len() != other.data.len() {
            return Err(ModelError::DimensionMismatch {
                expected: self.data.len(),
                got: other.data.len(),
            });
        }

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a * b)
            .collect();

        Ok(Self {
            data,
            shape: self.shape.clone(),
        })
    }

    fn scale(&self, scalar: f64) -> Self {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Self {
            data,
            shape: self.shape.clone(),
        }
    }

    fn dot(&self, other: &Self) -> ModelResult<f64> {
        if self.data.len() != other.data.len() {
            return Err(ModelError::DimensionMismatch {
                expected: self.data.len(),
                got: other.data.len(),
            });
        }

        Ok(self
            .data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| a * b)
            .sum())
    }

    fn matmul_vec(&self, vec: &Self) -> ModelResult<Self> {
        // Assume self is [m, n] and vec is [n]
        if self.shape.len() != 2 {
            return Err(ModelError::InvalidConfig(
                "Matrix must be 2D for matmul_vec".to_string(),
            ));
        }

        if vec.shape.len() != 1 {
            return Err(ModelError::InvalidConfig(
                "Vector must be 1D for matmul_vec".to_string(),
            ));
        }

        let m = self.shape[0];
        let n = self.shape[1];

        if n != vec.data.len() {
            return Err(ModelError::DimensionMismatch {
                expected: n,
                got: vec.data.len(),
            });
        }

        let mut result = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                result[i] += self.data[i * n + j] * vec.data[j];
            }
        }

        Ok(Tensor::from_slice(&result))
    }

    fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    fn mean(&self) -> f64 {
        if self.data.is_empty() {
            0.0
        } else {
            self.sum() / self.data.len() as f64
        }
    }

    fn std(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }

        let mean = self.mean();
        let variance: f64 =
            self.data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / self.data.len() as f64;

        variance.sqrt()
    }

    fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.size(), 6);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::ones(&[3, 2]);
        assert_eq!(t.size(), 6);
        assert!(t.data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_tensor_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::from_vec(data.clone(), &[2, 2]).unwrap();
        assert_eq!(t.data, data);
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_tensor_from_vec_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_vec(data, &[2, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_from_slice() {
        let data = &[1.0, 2.0, 3.0];
        let t = Tensor::from_slice(data);
        assert_eq!(t.data, data);
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_tensor_random_uniform() {
        let t = Tensor::random_uniform(&[100], 0.0, 1.0);
        assert_eq!(t.size(), 100);
        assert!(t.data.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_tensor_random_normal() {
        let t = Tensor::random_normal(&[1000], 0.0, 1.0);
        let mean = t.mean();
        let std = t.std();

        // Check if mean and std are roughly correct (within reasonable tolerance)
        assert!((mean - 0.0).abs() < 0.1);
        assert!((std - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_tensor_reshape() {
        let mut t = Tensor::zeros(&[2, 3]);
        assert!(t.reshape(&[3, 2]).is_ok());
        assert_eq!(t.shape(), &[3, 2]);

        assert!(t.reshape(&[2, 2]).is_err());
    }

    #[test]
    fn test_tensor_get_set() {
        let mut t = Tensor::zeros(&[3]);
        t.set(1, 5.0).unwrap();
        assert_eq!(t.get(1), Some(5.0));
    }

    #[test]
    fn test_tensor_fill() {
        let mut t = Tensor::zeros(&[3, 2]);
        t.fill(7.0);
        assert!(t.data.iter().all(|&x| x == 7.0));
    }

    #[test]
    fn test_tensor_map() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let t2 = t.map(|x| x * 2.0);
        assert_eq!(t2.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_tensor_map_inplace() {
        let mut t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        t.map_inplace(|x| x * 2.0);
        assert_eq!(t.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_tensor_has_nan_or_inf() {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        assert!(!t1.has_nan_or_inf());

        let t2 = Tensor::from_slice(&[1.0, f64::NAN, 3.0]);
        assert!(t2.has_nan_or_inf());

        let t3 = Tensor::from_slice(&[1.0, f64::INFINITY, 3.0]);
        assert!(t3.has_nan_or_inf());
    }

    #[test]
    fn test_tensor_clip() {
        let mut t = Tensor::from_slice(&[-2.0, 0.0, 5.0, 10.0]);
        t.clip(0.0, 5.0);
        assert_eq!(t.data, vec![0.0, 0.0, 5.0, 5.0]);
    }

    #[test]
    fn test_tensor_add() {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let t2 = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let result = t1.add(&t2).unwrap();
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_sub() {
        let t1 = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let t2 = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let result = t1.sub(&t2).unwrap();
        assert_eq!(result.data, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_tensor_mul() {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let t2 = Tensor::from_slice(&[2.0, 3.0, 4.0]);
        let result = t1.mul(&t2).unwrap();
        assert_eq!(result.data, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_tensor_scale() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let result = t.scale(2.0);
        assert_eq!(result.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_tensor_dot() {
        let t1 = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let t2 = Tensor::from_slice(&[4.0, 5.0, 6.0]);
        let result = t1.dot(&t2).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_tensor_matmul_vec() {
        // Matrix: [[1, 2], [3, 4]]
        let mat = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let vec = Tensor::from_slice(&[5.0, 6.0]);

        let result = mat.matmul_vec(&vec).unwrap();
        // [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
        assert_eq!(result.data, vec![17.0, 39.0]);
    }

    #[test]
    fn test_tensor_sum() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.sum(), 10.0);
    }

    #[test]
    fn test_tensor_mean() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.mean(), 2.5);
    }

    #[test]
    fn test_tensor_std() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let std = t.std();
        // Approximate standard deviation
        assert!((std - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_tensor_norm() {
        let t = Tensor::from_slice(&[3.0, 4.0]);
        assert_eq!(t.norm(), 5.0); // sqrt(9 + 16) = 5
    }
}
