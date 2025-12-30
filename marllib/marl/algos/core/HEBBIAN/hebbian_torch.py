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
PyTorch implementation of Hebbian Agent with online plasticity learning.
3-layer MLP with per-weight ABCD plasticity parameters.
"""

import torch
import torch.nn as nn
import gym
import numpy as np


class HebbianRulePerWeightTorch(nn.Module):
    """
    PyTorch implementation of per-weight Hebbian plasticity rule with ABCD parameters.
    
    Weight update rule:
        Δw = μ(t) * (A*pre*post + B*pre + C*post + D)
    
    where:
        - pre: pre-synaptic activation
        - post: post-synaptic activation
        - A, B, C, D: per-weight plasticity parameters (learnable)
        - μ(t): learning rate with exponential decay
    """
    
    DEFAULT_INIT_SCALE = 0.1
    
    def __init__(self, in_features, out_features, lr=0.01, decay_rate=0.001, clamp=2.0, init_scale=None):
        """
        Initialize Hebbian rule.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            lr: Initial learning rate (float)
            decay_rate: Exponential decay rate for learning rate (float)
            clamp: Maximum absolute value for weight clamping (float)
            init_scale: Scale for random initialization of plasticity parameters (float)
        """
        super(HebbianRulePerWeightTorch, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.initial_lr = lr
        self.decay_rate = decay_rate
        self.clamp_value = clamp
        self.timestep = 0
        
        if init_scale is None:
            init_scale = self.DEFAULT_INIT_SCALE
        
        # Initialize ABCD plasticity parameters as learnable parameters
        self.A = nn.Parameter(torch.randn(in_features, out_features) * init_scale)
        self.B = nn.Parameter(torch.randn(in_features, out_features) * init_scale)
        self.C = nn.Parameter(torch.randn(in_features, out_features) * init_scale)
        self.D = nn.Parameter(torch.randn(in_features, out_features) * init_scale)
        
        # Current learning rate (not a parameter, just state)
        self.register_buffer('lr', torch.tensor(lr))
    
    def update_weights(self, weights, pre_activation, post_activation):
        """
        Apply Hebbian update to weights.
        
        Args:
            weights: Current weight matrix (torch.Tensor)
            pre_activation: Pre-synaptic activations (torch.Tensor)
            post_activation: Post-synaptic activations (torch.Tensor)
        
        Returns:
            Updated weights (torch.Tensor)
        """
        # Update learning rate with exponential decay
        self.lr = self.initial_lr * torch.exp(-self.decay_rate * self.timestep)
        self.timestep += 1
        
        # Ensure inputs are 2D
        if pre_activation.dim() == 1:
            pre_activation = pre_activation.unsqueeze(0)
        if post_activation.dim() == 1:
            post_activation = post_activation.unsqueeze(0)
        
        # Compute Hebbian update: Δw = μ(t) * (A*pre*post + B*pre + C*post + D)
        # pre: [batch, in_features], post: [batch, out_features]
        # We want: [in_features, out_features]
        pre = pre_activation.t()  # [in_features, batch]
        post = post_activation    # [batch, out_features]
        
        outer_product = torch.mm(pre, post)  # [in_features, out_features]
        
        delta = self.lr * (
            self.A * outer_product +
            self.B * pre.mean(dim=1, keepdim=True).expand_as(weights) +
            self.C * post.mean(dim=0, keepdim=True).expand_as(weights) +
            self.D
        )
        
        # Apply update
        weights = weights + delta
        
        # Clamp weights
        if self.clamp_value is not None:
            weights = torch.clamp(weights, -self.clamp_value, self.clamp_value)
        
        return weights
    
    def reset_timestep(self):
        """Reset timestep counter for learning rate decay."""
        self.timestep = 0
        self.lr = torch.tensor(self.initial_lr)


class HebbianAgentTorch(nn.Module):
    """
    PyTorch implementation of agent with 3-layer neural network and Hebbian plasticity.
    
    Architecture:
        Input -> Hidden1 (tanh) -> Hidden2 (tanh) -> Output
    
    Each layer has associated Hebbian plasticity rules that update
    weights online during the forward pass.
    """
    
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
        super(HebbianAgentTorch, self).__init__()
        
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
            self.act_space = act_space
        elif isinstance(act_space, gym.spaces.Box):
            self.act_dim = act_space.shape[0]
            self.act_space = act_space
            # Store action bounds for clipping
            self.action_low = torch.from_numpy(act_space.low).float()
            self.action_high = torch.from_numpy(act_space.high).float()
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
        
        # Initialize weight parameters
        self.W1 = nn.Parameter(torch.randn(self.obs_dim, self.hidden_dim1) * self.weight_init_scale)
        self.W2 = nn.Parameter(torch.randn(self.hidden_dim1, self.hidden_dim2) * self.weight_init_scale)
        self.W3 = nn.Parameter(torch.randn(self.hidden_dim2, self.act_dim) * self.weight_init_scale)
        
        # Initialize Hebbian plasticity rules for each layer
        self.hebbian1 = HebbianRulePerWeightTorch(
            self.obs_dim, self.hidden_dim1,
            lr=self.lr, decay_rate=self.decay_rate, clamp=self.clamp
        )
        self.hebbian2 = HebbianRulePerWeightTorch(
            self.hidden_dim1, self.hidden_dim2,
            lr=self.lr, decay_rate=self.decay_rate, clamp=self.clamp
        )
        self.hebbian3 = HebbianRulePerWeightTorch(
            self.hidden_dim2, self.act_dim,
            lr=self.lr, decay_rate=self.decay_rate, clamp=self.clamp
        )
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass with Hebbian weight updates.
        
        Args:
            obs: Observation (numpy array or dict or torch.Tensor)
            deterministic: Whether to act deterministically (bool)
        
        Returns:
            Action (int for discrete, torch.Tensor for continuous)
        """
        # Convert observation to tensor
        if isinstance(obs, dict):
            obs = obs['obs']
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        elif not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        # Flatten if needed
        if obs.dim() > 1 and obs.size(0) == 1:
            obs = obs.squeeze(0)
        elif obs.dim() > 1:
            obs = obs.flatten()
        
        # Forward pass with online Hebbian updates
        with torch.no_grad():
            # Layer 1
            h1 = torch.tanh(torch.matmul(obs, self.W1))
            self.W1.data = self.hebbian1.update_weights(self.W1.data, obs, h1)
            
            # Layer 2
            h2 = torch.tanh(torch.matmul(h1, self.W2))
            self.W2.data = self.hebbian2.update_weights(self.W2.data, h1, h2)
            
            # Layer 3 (output)
            output = torch.matmul(h2, self.W3)
            self.W3.data = self.hebbian3.update_weights(self.W3.data, h2, output)
        
        # Convert to action
        if self.is_discrete:
            # Apply softmax and sample
            logits = output
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = torch.multinomial(probs, 1).item()
            return action
        else:
            # Continuous actions with tanh activation and gain
            action = torch.tanh(output) * self.action_gain
            if not deterministic:
                # Add exploration noise
                noise = torch.randn_like(action) * self.exploration_noise
                action = action + noise
            # Clip action to valid bounds
            action = torch.clamp(action, self.action_low, self.action_high)
            return action.detach().cpu().numpy()
    
    def act(self, obs, deterministic=False):
        """Alias for forward for compatibility."""
        return self.forward(obs, deterministic)
    
    def reset(self):
        """Reset Hebbian learning timesteps (for new episode)."""
        self.hebbian1.reset_timestep()
        self.hebbian2.reset_timestep()
        self.hebbian3.reset_timestep()
    
    def get_all_parameters(self):
        """
        Get all agent parameters (weights + plasticity params) as flat array.
        
        Returns:
            Flattened numpy array of all parameters
        """
        params = []
        for param in self.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def set_all_parameters(self, params):
        """
        Set all agent parameters from flat array.
        
        Args:
            params: Flattened numpy array of all parameters
        """
        params_tensor = torch.from_numpy(params).float()
        idx = 0
        
        for param in self.parameters():
            param_size = param.numel()
            param.data = params_tensor[idx:idx+param_size].reshape(param.shape)
            idx += param_size
    
    def get_parameter_count(self):
        """
        Get total number of parameters.
        
        Returns:
            Total parameter count (int)
        """
        return sum(p.numel() for p in self.parameters())
