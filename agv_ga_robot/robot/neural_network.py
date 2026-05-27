"""Neural Network - Simple feedforward network for robot control."""

import numpy as np
import math


class NeuralNetwork:
    """
    Simple feedforward neural network with tanh activation.
    Genome = flattened weights and biases.
    """
    
    def __init__(self, layer_sizes: list = None):
        """
        Initialize network.
        
        Args:
            layer_sizes: List of layer sizes, e.g., [16, 32, 16, 2]
        """
        self.layer_sizes = layer_sizes or [16, 32, 16, 2]
        
        # Initialize weights and biases
        self.weights = []  # List of weight matrices
        self.biases = []   # List of bias vectors
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier initialization."""
        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            
            # Xavier initialization: limit = sqrt(6 / (in + out))
            limit = math.sqrt(6.0 / (in_size + out_size))
            
            # Weight matrix [in_size, out_size]
            w = np.random.uniform(-limit, limit, size=(in_size, out_size))
            self.weights.append(w)
            
            # Bias vector [out_size]
            b = np.zeros(out_size)
            self.biases.append(b)
    
    def get_genome_size(self) -> int:
        """
        Get total number of parameters (for GA genome).
        
        Returns:
            Number of weights + biases
        """
        total = 0
        for w in self.weights:
            total += w.size
        for b in self.biases:
            total += b.size
        return total
    
    def set_genome(self, genome: np.ndarray) -> None:
        """
        Set network weights from genome (flat array).
        
        Args:
            genome: Flat array of parameters (from GA)
        """
        idx = 0
        
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            self.weights[i] = genome[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
        
        for i in range(len(self.biases)):
            b_size = self.biases[i].size
            self.biases[i] = genome[idx:idx + b_size].copy()
            idx += b_size
    
    def get_genome(self) -> np.ndarray:
        """
        Get network weights as flat genome array.
        
        Returns:
            Flat array of all parameters
        """
        parts = []
        
        for w in self.weights:
            parts.append(w.flatten())
        
        for b in self.biases:
            parts.append(b.flatten())
        
        return np.concatenate(parts)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through network.
        
        Args:
            inputs: Input vector [input_size]
        
        Returns:
            Output vector [output_size]
        """
        x = inputs.flatten().astype(np.float32)
        
        # Forward through hidden layers with tanh activation
        for i in range(len(self.weights) - 1):
            x = np.tanh(np.dot(x, self.weights[i]) + self.biases[i])
        
        # Output layer with tanh activation (for motor outputs in [-1, 1])
        output = np.tanh(np.dot(x, self.weights[-1]) + self.biases[-1])
        
        return output
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict output (alias for forward for clarity).
        
        Args:
            inputs: Input vector
        
        Returns:
            Output vector (motor commands [-1, 1])
        """
        return self.forward(inputs)
    
    def copy(self) -> 'NeuralNetwork':
        """Create a copy of this network."""
        net_copy = NeuralNetwork(self.layer_sizes)
        net_copy.set_genome(self.get_genome().copy())
        return net_copy
