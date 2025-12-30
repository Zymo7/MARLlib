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
Standalone example for running Hebbian MARL on PettingZoo MPE environments.

This version does NOT require Ray/RLlib installation - it works directly with
PettingZoo environments and the Hebbian algorithm components.

Requirements:
- pettingzoo[mpe]
- numpy
- cma
- gym
"""

import sys
import os

# Add parent directory to path to import marllib modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pettingzoo.mpe import simple_tag_v3
from marllib.marl.algos.core.HEBBIAN import HebbianAgent, HebbianTrainer


def create_mpe_simple_tag():
    """Create and configure MPE simple_tag environment."""
    # Create environment
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=25,
        continuous_actions=False
    )
    env.reset()
    
    # Get environment info
    agent_ids = env.agents
    obs_space = env.observation_space(agent_ids[0])
    act_space = env.action_space(agent_ids[0])
    
    env_info = {
        'space_obs': obs_space,
        'space_act': act_space,
        'num_agents': len(agent_ids),
        'agent_ids': agent_ids
    }
    
    return env, env_info


def main():
    """Main training function."""
    
    print("=" * 70)
    print("Hebbian MARL Standalone Example - MPE simple_tag")
    print("=" * 70)
    
    # Step 1: Create environment
    print("\n[1/4] Creating PettingZoo MPE simple_tag environment...")
    try:
        env, env_info = create_mpe_simple_tag()
        print(f"✓ Environment created successfully")
        print(f"  - Number of agents: {env_info['num_agents']}")
        print(f"  - Observation space: {env_info['space_obs']}")
        print(f"  - Action space: {env_info['space_act']}")
    except Exception as e:
        print(f"✗ Error creating environment: {e}")
        print("\nMake sure PettingZoo is installed:")
        print("  pip install pettingzoo[mpe]")
        return
    
    # Step 2: Configure Hebbian algorithm
    print("\n[2/4] Configuring Hebbian algorithm...")
    
    config = {
        # Network architecture
        'hidden_dim1': 8,
        'hidden_dim2': 8,
        
        # Hebbian plasticity parameters
        'lr': 0.01,
        'decay_rate': 0.001,
        'clamp': 2.0,
        'action_gain': 2.0,
        
        # Training parameters
        'generations': 20,        # Small number for quick test
        'pop_size': 15,           # Small population for quick test
        'sigma0': 0.5,
        'eval_episodes': 2,       # Episodes per evaluation
        'max_steps': 100,         # Steps per episode (reduced for quick test)
        'verbose': False,
        'seed': 42                # For reproducibility
    }
    
    print(f"✓ Configuration set")
    print(f"  - Generations: {config['generations']}")
    print(f"  - Population size: {config['pop_size']}")
    print(f"  - Network: {env_info['space_obs'].shape[0]} -> {config['hidden_dim1']} -> {config['hidden_dim2']} -> {env_info['space_act'].n}")
    
    # Step 3: Create trainer
    print("\n[3/4] Initializing Hebbian trainer...")
    try:
        trainer = HebbianTrainer(env, env_info, config)
        print(f"✓ Trainer initialized")
        print(f"  - Total parameters per agent: {trainer.param_dim}")
    except Exception as e:
        print(f"✗ Error initializing trainer: {e}")
        print("\nMake sure CMA-ES is installed:")
        print("  pip install cma")
        return
    
    # Step 4: Run training
    print("\n[4/4] Running training...")
    print("-" * 70)
    
    try:
        for generation in range(config['generations']):
            stats = trainer.train_generation()
            
            # Print progress every generation
            print(f"Gen {generation+1:3d}/{config['generations']:3d} | "
                  f"Mean: {stats['mean_fitness']:7.2f} | "
                  f"Max: {stats['max_fitness']:7.2f} | "
                  f"Best: {stats['best_fitness_overall']:7.2f}")
        
        print("-" * 70)
        
        # Save results
        os.makedirs('./results/hebbian_standalone', exist_ok=True)
        checkpoint_path = './results/hebbian_standalone/best_params.npy'
        trainer.save_checkpoint(checkpoint_path)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Best fitness achieved: {trainer.best_fitness:.2f}")
        print(f"Parameters saved to: {checkpoint_path}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == '__main__':
    main()
