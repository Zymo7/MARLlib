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
Example script for running Hebbian MARL on MPE environments.

This demonstrates how to:
1. Create an MPE environment
2. Configure the Hebbian algorithm
3. Run training with CMA-ES evolution
4. Save and load trained parameters
"""

from marllib import marl
from marllib.marl.algos import run_hebbian


def main():
    """Main training function."""
    
    # Step 1: Create environment
    # MPE simple_tag: predators try to catch prey
    print("Creating MPE simple_tag environment...")
    env = marl.make_env(
        environment_name="mpe",
        map_name="simple_tag",
        force_coop=False
    )
    
    # Get environment configuration
    env_instance, env_config = env
    
    # Step 2: Configure Hebbian algorithm
    print("\nConfiguring Hebbian algorithm...")
    
    # Algorithm hyperparameters
    algo_config = {
        'hidden_dim1': 8,
        'hidden_dim2': 8,
        'lr': 0.01,
        'decay_rate': 0.001,
        'clamp': 2.0,
        'action_gain': 2.0
    }
    
    # Training hyperparameters
    training_config = {
        'generations': 50,      # Reduced for faster testing
        'pop_size': 20,         # Reduced for faster testing
        'sigma0': 0.5,
        'eval_episodes': 2,     # Reduced for faster testing
        'max_steps': 600
    }
    
    # Combine configurations
    exp_config = env_config.copy()
    exp_config['algo_args'] = algo_config
    exp_config.update(training_config)
    exp_config['local_dir'] = './results/hebbian_mpe'
    
    # Step 3: Run training
    print("\nStarting Hebbian MARL training...")
    print("=" * 60)
    
    results = run_hebbian(
        exp_info=exp_config,
        env=env_instance,
        model=None,  # Hebbian doesn't use MARLlib's model system
        stop={'episode_reward_mean': 100}  # Optional early stopping
    )
    
    # Step 4: Display results
    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"Best fitness: {results['best_fitness']:.2f}")
    print(f"Checkpoint saved to: {results['checkpoint_path']}")
    
    # Plot training progress (optional)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(results['generations'], results['mean_fitness'], 
                label='Mean Fitness', alpha=0.7)
        plt.plot(results['generations'], results['max_fitness'], 
                label='Max Fitness', alpha=0.7)
        plt.plot(results['generations'], results['best_fitness_overall'], 
                label='Best Overall', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Hebbian MARL Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('./results/hebbian_mpe/training_progress.png')
        print(f"Training plot saved to: ./results/hebbian_mpe/training_progress.png")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
