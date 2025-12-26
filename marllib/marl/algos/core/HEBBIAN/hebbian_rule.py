# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Hebbian plasticity rule implementation.
Based on per-weight ABCD plasticity parameters.
"""

import numpy as np


class HebbianRulePerWeight:
    """
    Implements per-weight Hebbian plasticity rule with ABCD parameters.
    
    Weight update rule:
        Δw = μ(t) * (A*pre*post + B*pre + C*post + D)
    
    where:
        - pre: pre-synaptic activation
        - post: post-synaptic activation
        - A, B, C, D: per-weight plasticity parameters
        - μ(t): learning rate with exponential decay
    """
    
    # Default initialization scale for plasticity parameters
    DEFAULT_INIT_SCALE = 0.1
    
    def __init__(self, weights_shape, lr=0.01, decay_rate=0.001, clamp=2.0, init_scale=None):
        """
        Initialize Hebbian rule.
        
        Args:
            weights_shape: Shape of weight matrix (tuple)
            lr: Initial learning rate (float)
            decay_rate: Exponential decay rate for learning rate (float)
            clamp: Maximum absolute value for weight clamping (float)
            init_scale: Scale for random initialization of plasticity parameters (float)
        """
        self.weights_shape = weights_shape
        self.lr = lr
        self.initial_lr = lr
        self.decay_rate = decay_rate
        self.clamp = clamp
        self.timestep = 0
        
        if init_scale is None:
            init_scale = self.DEFAULT_INIT_SCALE
        
        # Initialize ABCD plasticity parameters per weight
        self.A = np.random.randn(*weights_shape) * init_scale
        self.B = np.random.randn(*weights_shape) * init_scale
        self.C = np.random.randn(*weights_shape) * init_scale
        self.D = np.random.randn(*weights_shape) * init_scale
    
    def update(self, weights, pre_activation, post_activation):
        """
        Apply Hebbian update to weights.
        
        Args:
            weights: Current weight matrix (numpy array)
            pre_activation: Pre-synaptic activations (numpy array)
            post_activation: Post-synaptic activations (numpy array)
        
        Returns:
            Updated weights (numpy array)
        """
        # Update learning rate with exponential decay
        self.lr = self.initial_lr * np.exp(-self.decay_rate * self.timestep)
        self.timestep += 1
        
        # Compute Hebbian update
        # Δw = μ(t) * (A*pre*post + B*pre + C*post + D)
        pre = pre_activation.reshape(-1, 1)
        post = post_activation.reshape(1, -1)
        
        delta = self.lr * (
            self.A * pre * post +
            self.B * pre +
            self.C * post +
            self.D
        )
        
        # Apply update
        weights = weights + delta
        
        # Clamp weights
        if self.clamp is not None:
            weights = np.clip(weights, -self.clamp, self.clamp)
        
        return weights
    
    def reset_timestep(self):
        """Reset timestep counter for learning rate decay."""
        self.timestep = 0
        self.lr = self.initial_lr
    
    def get_parameters(self):
        """
        Get all plasticity parameters as a flat array.
        
        Returns:
            Flattened array of all parameters (A, B, C, D)
        """
        return np.concatenate([
            self.A.flatten(),
            self.B.flatten(),
            self.C.flatten(),
            self.D.flatten()
        ])
    
    def set_parameters(self, params):
        """
        Set plasticity parameters from a flat array.
        
        Args:
            params: Flattened array of all parameters (A, B, C, D)
        """
        param_size = np.prod(self.weights_shape)
        self.A = params[:param_size].reshape(self.weights_shape)
        self.B = params[param_size:2*param_size].reshape(self.weights_shape)
        self.C = params[2*param_size:3*param_size].reshape(self.weights_shape)
        self.D = params[3*param_size:4*param_size].reshape(self.weights_shape)
