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
Hebbian Agent with online plasticity learning.
3-layer MLP with per-weight ABCD plasticity parameters.
"""

import numpy as np
import gym
from .hebbian_rule import HebbianRulePerWeight


class HebbianAgent:
    """
    Agent with 3-layer neural network and Hebbian plasticity.
    
    Architecture:
        Input -> Hidden1 (tanh) -> Hidden2 (tanh) -> Output
    
    Each layer has associated Hebbian plasticity rules that update
    weights online during the forward pass.
    """
    
    # Default initialization and exploration parameters
    DEFAULT_WEIGHT_INIT_SCALE = 0.1
    DEFAULT_EXPLORATION_NOISE = 0.1
    
    def __init__(self, obs_space, act_space, config):
        """
        Initialize Hebbian agent.
        
        Args:
            obs_space: Observation space (gym.Space)
            act_space: Action space (gym.Space)
            config: Configuration dict with hyperparameters
        """
        # Get observation and action dimensions
        if isinstance(obs_space, gym.spaces.Box):
            self.obs_dim = obs_space.shape[0]
        elif isinstance(obs_space, gym.spaces.Dict):
            self.obs_dim = obs_space['obs'].shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {type(obs_space)}")
        
        # Handle both Discrete and Box action spaces
        self.is_discrete = isinstance(act_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.act_dim = act_space.n
        elif isinstance(act_space, gym.spaces.Box):
            self.act_dim = act_space.shape[0]
        else:
            raise ValueError(f"Unsupported action space: {type(act_space)}")
        
        # Get hyperparameters from config
        self.hidden_dim1 = config.get('hidden_dim1', 8)
        self.hidden_dim2 = config.get('hidden_dim2', 8)
        self.lr = config.get('lr', 0.01)
        self.decay_rate = config.get('decay_rate', 0.001)
        self.clamp = config.get('clamp', 2.0)
        self.action_gain = config.get('action_gain', 2.0)
        self.weight_init_scale = config.get('weight_init_scale', self.DEFAULT_WEIGHT_INIT_SCALE)
        self.exploration_noise = config.get('exploration_noise', self.DEFAULT_EXPLORATION_NOISE)
        
        # Initialize weights
        self.W1 = np.random.randn(self.obs_dim, self.hidden_dim1) * self.weight_init_scale
        self.W2 = np.random.randn(self.hidden_dim1, self.hidden_dim2) * self.weight_init_scale
        self.W3 = np.random.randn(self.hidden_dim2, self.act_dim) * self.weight_init_scale
        
        # Initialize Hebbian plasticity rules for each layer
        self.hebbian1 = HebbianRulePerWeight(
            (self.obs_dim, self.hidden_dim1),
            lr=self.lr,
            decay_rate=self.decay_rate,
            clamp=self.clamp
        )
        self.hebbian2 = HebbianRulePerWeight(
            (self.hidden_dim1, self.hidden_dim2),
            lr=self.lr,
            decay_rate=self.decay_rate,
            clamp=self.clamp
        )
        self.hebbian3 = HebbianRulePerWeight(
            (self.hidden_dim2, self.act_dim),
            lr=self.lr,
            decay_rate=self.decay_rate,
            clamp=self.clamp
        )
    
    def act(self, obs, deterministic=False):
        """
        Select action based on observation.
        
        Args:
            obs: Observation (numpy array or dict)
            deterministic: Whether to act deterministically (bool)
        
        Returns:
            Action (int for discrete, numpy array for continuous)
        """
        # Extract observation if dict
        if isinstance(obs, dict):
            obs = obs['obs']
        
        # Ensure observation is a numpy array
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # Flatten if needed
        if len(obs.shape) > 1:
            obs = obs.flatten()
        
        # Forward pass with online Hebbian updates
        # Layer 1
        h1 = np.tanh(np.dot(obs, self.W1))
        self.W1 = self.hebbian1.update(self.W1, obs, h1)
        
        # Layer 2
        h2 = np.tanh(np.dot(h1, self.W2))
        self.W2 = self.hebbian2.update(self.W2, h1, h2)
        
        # Layer 3 (output)
        output = np.dot(h2, self.W3)
        self.W3 = self.hebbian3.update(self.W3, h2, output)
        
        # Convert to action
        if self.is_discrete:
            # Apply softmax and sample
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / np.sum(exp_output)
            if deterministic:
                action = np.argmax(probs)
            else:
                action = np.random.choice(self.act_dim, p=probs)
            return int(action)
        else:
            # Continuous actions with tanh activation and gain
            action = np.tanh(output) * self.action_gain
            if not deterministic:
                # Add exploration noise
                action = action + np.random.randn(*action.shape) * self.exploration_noise
            return action
    
    def reset(self):
        """Reset Hebbian learning timesteps (for new episode)."""
        self.hebbian1.reset_timestep()
        self.hebbian2.reset_timestep()
        self.hebbian3.reset_timestep()
    
    def get_all_parameters(self):
        """
        Get all agent parameters (weights + plasticity params) as flat array.
        
        Returns:
            Flattened array of all parameters
        """
        return np.concatenate([
            self.W1.flatten(),
            self.W2.flatten(),
            self.W3.flatten(),
            self.hebbian1.get_parameters(),
            self.hebbian2.get_parameters(),
            self.hebbian3.get_parameters()
        ])
    
    def set_all_parameters(self, params):
        """
        Set all agent parameters from flat array.
        
        Args:
            params: Flattened array of all parameters
        """
        idx = 0
        
        # Set weights
        w1_size = self.obs_dim * self.hidden_dim1
        self.W1 = params[idx:idx+w1_size].reshape(self.obs_dim, self.hidden_dim1)
        idx += w1_size
        
        w2_size = self.hidden_dim1 * self.hidden_dim2
        self.W2 = params[idx:idx+w2_size].reshape(self.hidden_dim1, self.hidden_dim2)
        idx += w2_size
        
        w3_size = self.hidden_dim2 * self.act_dim
        self.W3 = params[idx:idx+w3_size].reshape(self.hidden_dim2, self.act_dim)
        idx += w3_size
        
        # Set plasticity parameters
        hebbian1_size = 4 * w1_size
        self.hebbian1.set_parameters(params[idx:idx+hebbian1_size])
        idx += hebbian1_size
        
        hebbian2_size = 4 * w2_size
        self.hebbian2.set_parameters(params[idx:idx+hebbian2_size])
        idx += hebbian2_size
        
        hebbian3_size = 4 * w3_size
        self.hebbian3.set_parameters(params[idx:idx+hebbian3_size])
    
    def get_parameter_count(self):
        """
        Get total number of parameters.
        
        Returns:
            Total parameter count (int)
        """
        # Weights: W1 + W2 + W3
        weights_count = (
            self.obs_dim * self.hidden_dim1 +
            self.hidden_dim1 * self.hidden_dim2 +
            self.hidden_dim2 * self.act_dim
        )
        
        # Plasticity params: 4 parameters (A,B,C,D) per weight
        plasticity_count = 4 * weights_count
        
        return weights_count + plasticity_count
