# Hebbian MARL Algorithm Integration

This document describes the integration of Hebbian plasticity-based multi-agent reinforcement learning into MARLlib.

## Overview

The Hebbian MARL algorithm uses online Hebbian plasticity learning with:
- Per-weight ABCD plasticity parameters
- 3-layer neural network with tanh activation
- CMA-ES evolutionary optimization
- Support for both discrete and continuous action spaces

## File Structure

```
marllib/marl/algos/core/HEBBIAN/
├── __init__.py                 # Module exports
├── hebbian_rule.py            # Per-weight ABCD Hebbian plasticity rule
├── hebbian_agent.py           # 3-layer MLP agent with online plasticity
├── cmaes_optimizer.py         # CMA-ES evolution strategy wrapper
└── hebbian_trainer.py         # Training interface for MARLlib

marllib/marl/algos/
├── run_hebbian.py             # Main run script for Hebbian algorithm
└── hyperparams/common/
    └── hebbian.yaml           # Default hyperparameters

examples/
└── run_hebbian_mpe.py         # Example usage on MPE environments
```

## Key Features

### 1. Hebbian Plasticity (hebbian_rule.py)
- Per-weight ABCD parameters for flexible learning rules
- Exponential learning rate decay: μ(t) = μ₀ * exp(-decay_rate * t)
- Weight clamping to prevent unbounded growth
- Configurable initialization scale

### 2. Agent Architecture (hebbian_agent.py)
- 3-layer MLP: Input → Hidden1 (tanh) → Hidden2 (tanh) → Output
- Online weight updates during forward pass
- Support for both Box (continuous) and Discrete action spaces
- Action clipping for continuous spaces
- Configurable exploration noise

### 3. CMA-ES Optimizer (cmaes_optimizer.py)
- Wrapper around the `cma` Python library
- Population-based evolutionary optimization
- Configurable verbosity for debugging
- Optional seed for reproducibility

### 4. Trainer (hebbian_trainer.py)
- Bridges Hebbian algorithm with MARLlib's multi-agent environments
- Evaluates agent populations in parallel
- Tracks best parameters across generations
- Checkpoint saving and loading

## Usage Example

```python
from marllib import marl
from marllib.marl.algos import run_hebbian

# Create environment
env_instance, env_config = marl.make_env(
    environment_name="mpe",
    map_name="simple_tag",
    force_coop=False
)

# Configure algorithm
config = env_config.copy()
config['algo_args'] = {
    'hidden_dim1': 8,
    'hidden_dim2': 8,
    'lr': 0.01,
    'decay_rate': 0.001,
    'clamp': 2.0,
    'action_gain': 2.0
}
config.update({
    'generations': 100,
    'pop_size': 30,
    'sigma0': 0.5,
    'eval_episodes': 3,
    'max_steps': 600
})

# Run training
results = run_hebbian(
    exp_info=config,
    env=env_instance,
    model=None,
    stop={'episode_reward_mean': 100}
)
```

## Configuration Parameters

### Algorithm Parameters (algo_args)
- `hidden_dim1`: First hidden layer size (default: 8)
- `hidden_dim2`: Second hidden layer size (default: 8)
- `lr`: Initial learning rate for Hebbian updates (default: 0.01)
- `decay_rate`: Exponential decay rate for learning rate (default: 0.001)
- `clamp`: Weight clamping value (default: 2.0)
- `action_gain`: Gain for continuous actions (default: 2.0)
- `weight_init_scale`: Scale for weight initialization (default: 0.1)
- `exploration_noise`: Exploration noise for continuous actions (default: 0.1)

### Training Parameters
- `generations`: Number of generations to train (default: 100)
- `pop_size`: CMA-ES population size (default: 30)
- `sigma0`: Initial standard deviation for CMA-ES (default: 0.5)
- `eval_episodes`: Episodes per fitness evaluation (default: 3)
- `max_steps`: Maximum steps per episode (default: 600)
- `verbose`: Enable verbose CMA-ES output (default: false)
- `seed`: Random seed for reproducibility (optional)

## Supported Environments

The Hebbian algorithm works with all MARLlib-supported environments:
- Multi-Agent Particle Environment (MPE)
- StarCraft Multi-Agent Challenge (SMAC)
- Google Research Football (GRF)
- Multi-Agent MuJoCo (MAMuJoCo)
- And others...

## Dependencies

The integration requires the `cma` package:
```
cma==3.2.2
```

This has been added to `requirements.txt`.

## Design Principles

1. **Minimal changes to core algorithm**: The Hebbian plasticity logic is preserved exactly as designed
2. **Adapter pattern**: Wrapper classes bridge Hebbian and MARLlib interfaces
3. **Environment agnostic**: Works with all MARLlib environments through standardized interface
4. **Standard interfaces**: Follows MARLlib's existing patterns for algorithm integration

## Future Enhancements

- EvoRainbow quality diversity enhancements
- Parallel evaluation optimization
- Advanced fitness functions (novelty, behavioral diversity)
- Multi-objective optimization support

## References

Based on:
- Source repository: `Zymo7/Hebbian_MARL_v14`
- Hebbian learning theory
- CMA-ES evolutionary optimization
