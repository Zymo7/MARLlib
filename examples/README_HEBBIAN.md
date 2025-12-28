# Hebbian MARL Examples

This directory contains example scripts for using the Hebbian MARL algorithm.

## Available Examples

### 1. `run_hebbian_standalone.py` ⭐ **Recommended for Quick Start**

**Use this if you want to test Hebbian MARL quickly without full MARLlib setup.**

A standalone example that works directly with PettingZoo environments without requiring Ray/RLlib.

**Requirements:**
```bash
pip install numpy cma gym==0.20.0 pettingzoo[mpe]
```

**Run:**
```bash
python examples/run_hebbian_standalone.py
```

**Features:**
- ✅ No Ray/RLlib required
- ✅ Direct PettingZoo environment integration
- ✅ Quick setup and testing
- ✅ Easy to understand and modify

---

### 2. `run_hebbian_mpe.py` (Full MARLlib Integration)

**Use this if you have MARLlib fully installed and want to use it with MARLlib's environment wrappers.**

Example that uses the full MARLlib API with environment registration and configuration system.

**Requirements:**
```bash
# Full MARLlib installation with all dependencies
pip install -r requirements.txt
```

**Run:**
```bash
python examples/run_hebbian_mpe.py
```

**Features:**
- ✅ Uses MARLlib's environment wrappers
- ✅ Consistent with other MARLlib examples
- ✅ Access to MARLlib's environment registry
- ❗ Requires Ray/RLlib and full MARLlib setup

---

## Common Error: Ray/RLlib Import Issues

If you see errors like:
```
from ray.rllib.models.catalog import ModelCatalog
ImportError: ...
```

**Solution:** Use the standalone version (`run_hebbian_standalone.py`) which doesn't require Ray.

---

## Training Parameters

Both scripts support the same configuration parameters:

### Network Architecture
- `hidden_dim1`: First hidden layer size (default: 8)
- `hidden_dim2`: Second hidden layer size (default: 8)

### Hebbian Plasticity
- `lr`: Initial learning rate (default: 0.01)
- `decay_rate`: Learning rate decay (default: 0.001)
- `clamp`: Weight clamping value (default: 2.0)
- `action_gain`: Action scaling for continuous spaces (default: 2.0)

### Training
- `generations`: Number of evolutionary generations (default: 20-100)
- `pop_size`: CMA-ES population size (default: 15-30)
- `eval_episodes`: Episodes per fitness evaluation (default: 2-3)
- `max_steps`: Maximum steps per episode (default: 100-600)
- `seed`: Random seed for reproducibility (optional)

---

## Output

Both scripts will:
1. Train Hebbian agents using CMA-ES evolution
2. Display training progress (fitness scores per generation)
3. Save the best parameters to `./results/hebbian_*/best_params.npy`

---

## Next Steps

After training, you can:
- Modify the configuration parameters to experiment
- Try different PettingZoo environments
- Load saved parameters for evaluation
- Extend the code for your own use case

For more details, see `docs/HEBBIAN_INTEGRATION.md`
