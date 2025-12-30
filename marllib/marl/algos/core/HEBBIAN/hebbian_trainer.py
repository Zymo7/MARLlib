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
Hebbian trainer that bridges algorithm with MARLlib infrastructure.
Supports both NumPy and PyTorch backends.
"""

import numpy as np
from .hebbian_agent import HebbianAgent
from .cmaes_optimizer import CMAESOptimizer

# Try to import PyTorch version
try:
    from .hebbian_torch import HebbianAgentTorch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HebbianTrainer:
    """
    Trainer for Hebbian agents using CMA-ES evolution.
    
    This class handles:
    - Creating agents for each policy
    - Evaluating agent populations in environments
    - Updating the CMA-ES optimizer
    """
    
    def __init__(self, env, env_info, config):
        """
        Initialize Hebbian trainer.
        
        Args:
            env: Environment instance
            env_info: Environment information dict
            config: Configuration dict with training parameters
        """
        self.env = env
        self.env_info = env_info
        self.config = config
        
        # Get environment spaces
        self.obs_space = env_info['space_obs']
        self.act_space = env_info['space_act']
        self.num_agents = env_info['num_agents']
        
        # Determine backend (pytorch or numpy)
        self.use_torch = config.get('use_torch', TORCH_AVAILABLE)
        if self.use_torch and not TORCH_AVAILABLE:
            print("Warning: PyTorch backend requested but not available. Falling back to NumPy.")
            self.use_torch = False
        
        # Training hyperparameters
        self.generations = config.get('generations', 100)
        self.pop_size = config.get('pop_size', 30)
        self.sigma0 = config.get('sigma0', 0.5)
        self.eval_episodes = config.get('eval_episodes', 3)
        self.max_steps = config.get('max_steps', 600)
        self.verbose = config.get('verbose', False)
        self.seed = config.get('seed', None)
        
        # Algorithm hyperparameters (passed to agents)
        self.algo_config = {
            'hidden_dim1': config.get('hidden_dim1', 8),
            'hidden_dim2': config.get('hidden_dim2', 8),
            'lr': config.get('lr', 0.01),
            'decay_rate': config.get('decay_rate', 0.001),
            'clamp': config.get('clamp', 2.0),
            'action_gain': config.get('action_gain', 2.0),
            'weight_init_scale': config.get('weight_init_scale', 0.1),
            'exploration_noise': config.get('exploration_noise', 0.1)
        }
        
        # Select agent class based on backend
        if self.use_torch:
            self.agent_class = HebbianAgentTorch
            print(f"Using PyTorch backend for Hebbian agents")
        else:
            self.agent_class = HebbianAgent
            print(f"Using NumPy backend for Hebbian agents")
        
        # Create a template agent to get parameter dimensions
        template_agent = self.agent_class(self.obs_space, self.act_space, self.algo_config)
        self.param_dim = template_agent.get_parameter_count()
        
        # Initialize CMA-ES optimizer
        self.optimizer = CMAESOptimizer(
            self.param_dim,
            pop_size=self.pop_size,
            sigma0=self.sigma0,
            verbose=self.verbose,
            seed=self.seed
        )
        
        # Best parameters tracking
        self.best_params = None
        self.best_fitness = float('-inf')
    
    def evaluate_agent(self, params, num_episodes=None):
        """
        Evaluate an agent with given parameters.
        
        Args:
            params: Parameter vector for the agent
            num_episodes: Number of episodes to evaluate (uses self.eval_episodes if None)
        
        Returns:
            Average cumulative reward across episodes
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes
        
        total_reward = 0.0
        
        for _ in range(num_episodes):
            # Create agents for all policies with same parameters
            agents = {}
            for agent_id in self.env.agents:
                agent = self.agent_class(self.obs_space, self.act_space, self.algo_config)
                agent.set_all_parameters(params)
                agents[agent_id] = agent
            
            # Reset environment and agents
            obs_dict = self.env.reset()
            for agent in agents.values():
                agent.reset()
            
            episode_reward = 0.0
            done = {agent_id: False for agent_id in self.env.agents}
            
            for step in range(self.max_steps):
                # Get actions from all agents
                actions = {}
                for agent_id, agent in agents.items():
                    if not done[agent_id]:
                        obs = obs_dict[agent_id]
                        actions[agent_id] = agent.act(obs)
                
                # Step environment
                obs_dict, rewards, done, info = self.env.step(actions)
                
                # Accumulate rewards
                for agent_id, reward in rewards.items():
                    episode_reward += reward
                
                # Check if all agents are done
                if all(done.values()):
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def train_generation(self):
        """
        Train one generation using CMA-ES.
        
        Returns:
            Dictionary with generation statistics
        """
        # Sample population
        population = self.optimizer.ask()
        
        # Evaluate each individual
        fitness_values = []
        for params in population:
            fitness = self.evaluate_agent(params)
            # CMA-ES minimizes, so negate for maximization
            fitness_values.append(-fitness)
        
        # Update optimizer
        self.optimizer.tell(population, fitness_values)
        
        # Track best solution
        best_params, best_fitness_neg = self.optimizer.best_solution()
        best_fitness = -best_fitness_neg
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_params = best_params
        
        # Compute statistics
        actual_fitness = [-f for f in fitness_values]
        stats = {
            'mean_fitness': np.mean(actual_fitness),
            'max_fitness': np.max(actual_fitness),
            'min_fitness': np.min(actual_fitness),
            'best_fitness_overall': self.best_fitness
        }
        
        return stats
    
    def save_checkpoint(self, path):
        """
        Save best parameters to file.
        
        Args:
            path: File path to save parameters
        """
        if self.best_params is not None:
            np.save(path, self.best_params)
    
    def load_checkpoint(self, path):
        """
        Load parameters from file.
        
        Args:
            path: File path to load parameters from
        """
        self.best_params = np.load(path)
