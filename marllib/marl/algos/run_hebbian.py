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
Run script for Hebbian MARL algorithm.
"""

import os
from typing import Any, Dict
from marllib.marl.algos.core.HEBBIAN import HebbianTrainer


def run_hebbian(exp_info: Dict, env: Any, model: Any, stop: Dict = None) -> Dict:
    """
    Run Hebbian MARL algorithm using CMA-ES evolution.
    
    This is the main entry point for training Hebbian agents in MARLlib.
    
    Args:
        exp_info: Experiment configuration dictionary containing:
            - env_args: Environment configuration
            - algo_args: Algorithm hyperparameters
            - training hyperparameters (generations, pop_size, etc.)
        env: Environment instance
        model: Model class (not used for Hebbian, but kept for API consistency)
        stop: Stop conditions dictionary (optional)
    
    Returns:
        Dictionary containing training results
    """
    
    ########################
    ### environment info ###
    ########################
    
    env_info = env.get_env_info()
    map_name = exp_info['env_args']['map_name']
    agent_name_ls = env.agents
    env_info["agent_name_ls"] = agent_name_ls
    
    ######################
    ### Configuration  ###
    ######################
    
    # Merge algorithm args and training args from exp_info
    config = {}
    
    # Get algorithm hyperparameters
    if 'algo_args' in exp_info:
        config.update(exp_info['algo_args'])
    
    # Get training hyperparameters (generations, pop_size, etc.)
    training_keys = ['generations', 'pop_size', 'sigma0', 'eval_episodes', 'max_steps']
    for key in training_keys:
        if key in exp_info:
            config[key] = exp_info[key]
    
    # Apply stop conditions if provided
    if stop is not None:
        if 'training_iteration' in stop:
            config['generations'] = stop['training_iteration']
    
    ######################
    ### Create Trainer ###
    ######################
    
    print(f"\n{'='*60}")
    print(f"Starting Hebbian MARL Training")
    print(f"{'='*60}")
    print(f"Environment: {exp_info['env']} - {map_name}")
    print(f"Number of agents: {env_info['num_agents']}")
    print(f"Observation space: {env_info['space_obs']}")
    print(f"Action space: {env_info['space_act']}")
    print(f"\nAlgorithm Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    trainer = HebbianTrainer(env, env_info, config)
    
    print(f"Total parameters per agent: {trainer.param_dim}")
    print(f"Population size: {trainer.pop_size}")
    print(f"Generations: {trainer.generations}")
    print()
    
    ##################
    ### Training   ###
    ##################
    
    results = {
        'generations': [],
        'mean_fitness': [],
        'max_fitness': [],
        'best_fitness_overall': []
    }
    
    for generation in range(trainer.generations):
        # Train one generation
        stats = trainer.train_generation()
        
        # Store results
        results['generations'].append(generation)
        results['mean_fitness'].append(stats['mean_fitness'])
        results['max_fitness'].append(stats['max_fitness'])
        results['best_fitness_overall'].append(stats['best_fitness_overall'])
        
        # Print progress
        print(f"Generation {generation+1}/{trainer.generations} | "
              f"Mean: {stats['mean_fitness']:.2f} | "
              f"Max: {stats['max_fitness']:.2f} | "
              f"Best Overall: {stats['best_fitness_overall']:.2f}")
        
        # Check for early stopping based on stop conditions
        if stop is not None:
            if 'episode_reward_mean' in stop:
                if stats['mean_fitness'] >= stop['episode_reward_mean']:
                    print(f"\nReached target reward: {stop['episode_reward_mean']}")
                    break
    
    ######################
    ### Save Results   ###
    ######################
    
    # Save best parameters
    checkpoint_dir = exp_info.get('local_dir', './results')
    if checkpoint_dir == "":
        checkpoint_dir = './results'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"hebbian_{exp_info['env']}_{map_name}_best.npy"
    )
    trainer.save_checkpoint(checkpoint_path)
    print(f"\nBest parameters saved to: {checkpoint_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best fitness achieved: {trainer.best_fitness:.2f}")
    print(f"Parameters saved to: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Close environment
    env.close()
    
    # Return results in a format similar to Ray Tune results
    results['best_fitness'] = trainer.best_fitness
    results['best_params'] = trainer.best_params
    results['checkpoint_path'] = checkpoint_path
    
    return results
