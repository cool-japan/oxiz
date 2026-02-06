//! Neural Network Implementation
//!
//! Lightweight feedforward neural network with backpropagation.
//! Designed for fast inference (<100μs per prediction).

use super::activation::Activation;
use super::loss::Loss;
use super::optimizer::Optimizer;
use super::tensor::{Tensor, TensorOps};
use super::{Model, ModelError, ModelResult};
use serde::{Deserialize, Serialize};

/// A single layer in the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Weight matrix \[input_dim, output_dim\]
    pub weights: Tensor,
    /// Bias vector \[output_dim\]
    pub bias: Tensor,
    /// Activation function
    pub activation: Activation,
    /// Cached input (for backpropagation)
    #[serde(skip)]
    cached_input: Option<Tensor>,
    /// Cached pre-activation (for backpropagation)
    #[serde(skip)]
    cached_z: Option<Tensor>,
    /// Cached output (for backpropagation)
    #[serde(skip)]
    cached_output: Option<Tensor>,
}

impl Layer {
    /// Create a new layer
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        // Use He initialization for ReLU, Xavier for others
        // Weight matrix is [output_dim, input_dim] for matmul: W * x = y
        // where x is [input_dim] and y is [output_dim]
        let weights = if matches!(activation, Activation::ReLU | Activation::LeakyReLU) {
            Tensor::he_init(&[output_dim, input_dim])
        } else {
            Tensor::xavier_init(&[output_dim, input_dim])
        };

        let bias = Tensor::zeros(&[output_dim]);

        Self {
            weights,
            bias,
            activation,
            cached_input: None,
            cached_z: None,
            cached_output: None,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Tensor, training: bool) -> ModelResult<Tensor> {
        if input.shape().len() != 1 {
            return Err(ModelError::InvalidConfig(
                "Input must be 1D vector".to_string(),
            ));
        }

        // Weights are [output_dim, input_dim], so input should match shape()[1]
        if input.data.len() != self.weights.shape()[1] {
            return Err(ModelError::DimensionMismatch {
                expected: self.weights.shape()[1],
                got: input.data.len(),
            });
        }

        // Cache input for backprop
        if training {
            self.cached_input = Some(input.clone());
        }

        // z = W * x + b
        let z = self.weights.matmul_vec(input)?;
        let z = z.add(&self.bias)?;

        if training {
            self.cached_z = Some(z.clone());
        }

        // Apply activation
        let output = z.map(|x| self.activation.apply(x));

        if training {
            self.cached_output = Some(output.clone());
        }

        Ok(output)
    }

    /// Backward pass
    pub fn backward(&self, grad_output: &Tensor) -> ModelResult<(Tensor, Tensor, Tensor)> {
        let cached_input = self
            .cached_input
            .as_ref()
            .ok_or_else(|| ModelError::TrainingError("No cached input for backprop".to_string()))?;
        let cached_z = self
            .cached_z
            .as_ref()
            .ok_or_else(|| ModelError::TrainingError("No cached z for backprop".to_string()))?;

        // Compute gradient through activation: grad_z = grad_output * activation'(z)
        let activation_grad = cached_z.map(|x| self.activation.derivative(x));
        let grad_z = grad_output.mul(&activation_grad)?;

        // Weights are [output_dim, input_dim]
        let output_dim = self.weights.shape()[0];
        let input_dim = self.weights.shape()[1];

        // Gradient w.r.t weights: grad_W[i,j] = grad_z[i] * input[j]
        let mut grad_weights = Tensor::zeros(&[output_dim, input_dim]);
        for i in 0..output_dim {
            for j in 0..input_dim {
                grad_weights.data[i * input_dim + j] = grad_z.data[i] * cached_input.data[j];
            }
        }

        // Gradient w.r.t bias: grad_b = grad_z
        let grad_bias = grad_z.clone();

        // Gradient w.r.t input: grad_input[j] = sum_i(W[i,j] * grad_z[i])
        let mut grad_input = Tensor::zeros(&[input_dim]);
        for j in 0..input_dim {
            for i in 0..output_dim {
                grad_input.data[j] += self.weights.data[i * input_dim + j] * grad_z.data[i];
            }
        }

        Ok((grad_input, grad_weights, grad_bias))
    }

    /// Clear cached values
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
        self.cached_z = None;
        self.cached_output = None;
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.size() + self.bias.size()
    }
}

/// Neural network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Layer sizes (including input and output)
    pub layer_sizes: Vec<usize>,
    /// Activation functions for each hidden layer
    pub activations: Vec<Activation>,
    /// Loss function
    pub loss: Loss,
    /// Learning rate
    pub learning_rate: f64,
    /// Use batch normalization
    pub batch_norm: bool,
    /// Dropout rate (0.0 = no dropout)
    pub dropout_rate: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![10, 8, 1],
            activations: vec![Activation::ReLU, Activation::Linear],
            loss: Loss::MSE,
            learning_rate: 0.01,
            batch_norm: false,
            dropout_rate: 0.0,
        }
    }
}

/// Feedforward neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    /// Network layers
    layers: Vec<Layer>,
    /// Configuration
    config: NetworkConfig,
    /// Number of training steps
    #[serde(skip)]
    training_steps: usize,
}

impl NeuralNetwork {
    /// Create a new neural network from configuration
    pub fn new(config: NetworkConfig) -> ModelResult<Self> {
        if config.layer_sizes.len() < 2 {
            return Err(ModelError::InvalidConfig(
                "Network must have at least 2 layers (input and output)".to_string(),
            ));
        }

        if config.activations.len() != config.layer_sizes.len() - 1 {
            return Err(ModelError::InvalidConfig(
                "Number of activations must equal number of layers minus 1".to_string(),
            ));
        }

        let mut layers = Vec::new();
        for i in 0..config.layer_sizes.len() - 1 {
            let input_dim = config.layer_sizes[i];
            let output_dim = config.layer_sizes[i + 1];
            let activation = config.activations[i];

            layers.push(Layer::new(input_dim, output_dim, activation));
        }

        Ok(Self {
            layers,
            config,
            training_steps: 0,
        })
    }

    /// Create a simple network with given layer sizes and ReLU activations
    pub fn simple(layer_sizes: Vec<usize>) -> ModelResult<Self> {
        if layer_sizes.len() < 2 {
            return Err(ModelError::InvalidConfig(
                "Network must have at least 2 layers".to_string(),
            ));
        }

        let mut activations = vec![Activation::ReLU; layer_sizes.len() - 2];
        activations.push(Activation::Linear); // Linear output layer

        let config = NetworkConfig {
            layer_sizes,
            activations,
            loss: Loss::MSE,
            learning_rate: 0.01,
            batch_norm: false,
            dropout_rate: 0.0,
        };

        Self::new(config)
    }

    /// Forward pass
    pub fn forward(&mut self, input: &[f64], training: bool) -> ModelResult<Vec<f64>> {
        if input.len() != self.config.layer_sizes[0] {
            return Err(ModelError::DimensionMismatch {
                expected: self.config.layer_sizes[0],
                got: input.len(),
            });
        }

        let mut current = Tensor::from_slice(input);

        for layer in &mut self.layers {
            current = layer.forward(&current, training)?;
        }

        Ok(current.data)
    }

    /// Forward pass returning tensor
    fn forward_tensor(&mut self, input: &Tensor, training: bool) -> ModelResult<Tensor> {
        let mut current = input.clone();

        for layer in &mut self.layers {
            current = layer.forward(&current, training)?;
        }

        Ok(current)
    }

    /// Backward pass
    fn backward(&self, loss_grad: &Tensor) -> ModelResult<Vec<(Tensor, Tensor)>> {
        let mut grad_output = loss_grad.clone();
        let mut gradients = Vec::new();

        // Backpropagate through layers in reverse order
        for layer in self.layers.iter().rev() {
            let (grad_input, grad_weights, grad_bias) = layer.backward(&grad_output)?;

            gradients.push((grad_weights, grad_bias));
            grad_output = grad_input;
        }

        // Reverse to match layer order
        gradients.reverse();

        Ok(gradients)
    }

    /// Training step with optimizer
    pub fn train_step<O: Optimizer>(
        &mut self,
        input: &[f64],
        target: &[f64],
        optimizer: &mut O,
    ) -> ModelResult<f64> {
        if target.len() != self.config.layer_sizes[self.config.layer_sizes.len() - 1] {
            return Err(ModelError::DimensionMismatch {
                expected: self.config.layer_sizes[self.config.layer_sizes.len() - 1],
                got: target.len(),
            });
        }

        // Forward pass
        let input_tensor = Tensor::from_slice(input);
        let output = self.forward_tensor(&input_tensor, true)?;

        // Compute loss
        let target_tensor = Tensor::from_slice(target);
        let loss_value = self
            .config
            .loss
            .compute_vec(&output.data, &target_tensor.data);

        // Compute loss gradient
        let loss_grad = self
            .config
            .loss
            .gradient_vec(&output.data, &target_tensor.data);
        let loss_grad_tensor = Tensor::from_slice(&loss_grad);

        // Backward pass
        let gradients = self.backward(&loss_grad_tensor)?;

        // Update weights using optimizer
        for (i, (grad_weights, grad_bias)) in gradients.into_iter().enumerate() {
            let param_id_weights = i * 2;
            let param_id_bias = i * 2 + 1;

            optimizer.step(param_id_weights, &mut self.layers[i].weights, &grad_weights);
            optimizer.step(param_id_bias, &mut self.layers[i].bias, &grad_bias);
        }

        // Clear caches
        for layer in &mut self.layers {
            layer.clear_cache();
        }

        self.training_steps += 1;

        Ok(loss_value)
    }

    /// Get configuration
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }

    /// Get number of training steps
    pub fn num_training_steps(&self) -> usize {
        self.training_steps
    }

    /// Reset training statistics
    pub fn reset_stats(&mut self) {
        self.training_steps = 0;
    }
}

impl Model for NeuralNetwork {
    fn input_dim(&self) -> usize {
        self.config.layer_sizes[0]
    }

    fn output_dim(&self) -> usize {
        self.config.layer_sizes[self.config.layer_sizes.len() - 1]
    }

    fn predict(&self, input: &[f64]) -> Vec<f64> {
        // Inference mode (no caching)
        if input.len() != self.input_dim() {
            return vec![0.0; self.output_dim()];
        }

        let mut current = Tensor::from_slice(input);

        for layer in &self.layers {
            // Manual forward without caching
            let z = layer
                .weights
                .matmul_vec(&current)
                .expect("Dimension mismatch in predict");
            let z = z.add(&layer.bias).expect("Dimension mismatch in predict");
            current = z.map(|x| layer.activation.apply(x));
        }

        current.data
    }

    fn train(&mut self, input: &[f64], target: &[f64]) -> ModelResult<f64> {
        // Use simple SGD for default training
        let mut optimizer = super::optimizer::SGD::new(self.config.learning_rate);
        self.train_step(input, target, &mut optimizer)
    }

    fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.num_parameters()).sum()
    }

    fn save(&self) -> ModelResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| ModelError::SerializationError(e.to_string()))
    }

    fn load(&mut self, data: &[u8]) -> ModelResult<()> {
        let loaded: NeuralNetwork = serde_json::from_slice(data)
            .map_err(|e| ModelError::SerializationError(e.to_string()))?;

        self.layers = loaded.layers;
        self.config = loaded.config;
        self.training_steps = 0;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::new(10, 5, Activation::ReLU);
        // Weights are [output_dim, input_dim] = [5, 10]
        assert_eq!(layer.weights.shape(), &[5, 10]);
        assert_eq!(layer.bias.shape(), &[5]);
        assert_eq!(layer.num_parameters(), 55); // 10*5 + 5
    }

    #[test]
    fn test_layer_forward() {
        let mut layer = Layer::new(3, 2, Activation::ReLU);
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0]);

        let output = layer.forward(&input, false).unwrap();
        assert_eq!(output.shape(), &[2]);
    }

    #[test]
    fn test_network_creation() {
        let config = NetworkConfig {
            layer_sizes: vec![10, 8, 4, 1],
            activations: vec![Activation::ReLU, Activation::ReLU, Activation::Linear],
            loss: Loss::MSE,
            learning_rate: 0.01,
            batch_norm: false,
            dropout_rate: 0.0,
        };

        let network = NeuralNetwork::new(config).unwrap();
        assert_eq!(network.input_dim(), 10);
        assert_eq!(network.output_dim(), 1);
        assert_eq!(network.layers.len(), 3);
    }

    #[test]
    fn test_network_simple() {
        let network = NeuralNetwork::simple(vec![5, 3, 1]).unwrap();
        assert_eq!(network.input_dim(), 5);
        assert_eq!(network.output_dim(), 1);
    }

    #[test]
    fn test_network_invalid_config() {
        let config = NetworkConfig {
            layer_sizes: vec![10],
            activations: vec![],
            loss: Loss::MSE,
            learning_rate: 0.01,
            batch_norm: false,
            dropout_rate: 0.0,
        };

        let result = NeuralNetwork::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_network_forward() {
        let network = NeuralNetwork::simple(vec![3, 2, 1]).unwrap();
        let input = vec![1.0, 2.0, 3.0];

        let output = network.predict(&input);
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_network_training() {
        let mut network = NeuralNetwork::simple(vec![2, 3, 1]).unwrap();
        let input = vec![1.0, 2.0];
        let target = vec![3.0];

        let loss = network.train(&input, &target).unwrap();
        assert!(loss >= 0.0);
        assert_eq!(network.num_training_steps(), 1);
    }

    #[test]
    fn test_network_training_convergence() {
        let mut network = NeuralNetwork::simple(vec![2, 4, 1]).unwrap();
        let mut optimizer = crate::models::optimizer::Adam::new(0.01);

        // Train on simple pattern: output = 2*x1 + 3*x2
        let data = vec![
            (vec![1.0, 0.0], vec![2.0]),
            (vec![0.0, 1.0], vec![3.0]),
            (vec![1.0, 1.0], vec![5.0]),
            (vec![2.0, 1.0], vec![7.0]),
        ];

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for epoch in 0..100 {
            for (input, target) in &data {
                let loss = network.train_step(input, target, &mut optimizer).unwrap();

                if epoch == 0 {
                    initial_loss = loss;
                }
                if epoch == 99 {
                    final_loss = loss;
                }
            }
        }

        // Loss should decrease
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn test_network_num_parameters() {
        let network = NeuralNetwork::simple(vec![10, 5, 1]).unwrap();
        // (10*5 + 5) + (5*1 + 1) = 55 + 6 = 61
        assert_eq!(network.num_parameters(), 61);
    }

    #[test]
    fn test_network_save_load() {
        let mut network = NeuralNetwork::simple(vec![3, 2, 1]).unwrap();

        // Train a bit to have some non-random weights
        let input = vec![1.0, 2.0, 3.0];
        let target = vec![5.0];
        network.train(&input, &target).unwrap();

        // Save
        let saved = network.save().unwrap();
        assert!(!saved.is_empty());

        // Create new network and load
        let mut network2 = NeuralNetwork::simple(vec![3, 2, 1]).unwrap();
        network2.load(&saved).unwrap();

        // Predictions should be the same
        let pred1 = network.predict(&input);
        let pred2 = network2.predict(&input);

        for (p1, p2) in pred1.iter().zip(pred2.iter()) {
            assert!((p1 - p2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_network_dimension_mismatch() {
        let mut network = NeuralNetwork::simple(vec![3, 2, 1]).unwrap();
        let wrong_input = vec![1.0, 2.0]; // Should be 3

        let result = network.forward(&wrong_input, false);
        assert!(result.is_err());
    }
}
