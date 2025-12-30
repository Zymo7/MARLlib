# Hebbian MARL Examples

This directory contains example scripts for using the Hebbian MARL algorithm.

## Available Examples

### 1. `run_hebbian_pytorch.py` ⭐ **Recommended - PyTorch Backend**

**Use this if you want PyTorch-based neural networks (matches original Hebbian MARL architecture).**

Uses PyTorch for neural network implementation, providing better GPU support and compatibility with the original Hebbian MARL v14 implementation.

**Requirements:**
```bash
pip install torch numpy cma gym==0.20.0 pettingzoo[mpe]
```

**Run:**
```bash
python examples/run_hebbian_pytorch.py
```

**Features:**
- ✅ PyTorch neural networks (matches original architecture)
- ✅ GPU acceleration support
- ✅ Automatic differentiation capabilities
- ✅ Compatible with PyTorch ecosystem
- ✅ No Ray/RLlib required

---

### 2. `run_hebbian_standalone.py` (NumPy Backend)

**Use this if you want minimal dependencies without PyTorch.**

A standalone example that works directly with PettingZoo environments using NumPy arrays.

**Requirements:**
```bash
pip install numpy cma gym==0.20.0 pettingzoo[mpe]
```

**Run:**
```bash
python examples/run_hebbian_standalone.py
```

**Features:**
- ✅ No PyTorch/Ray/RLlib required
- ✅ Minimal dependencies
- ✅ Quick setup and testing
- ✅ Easy to understand and modify
- ❗ NumPy arrays instead of PyTorch tensors

---

### 3. `run_hebbian_mpe.py` (Full MARLlib Integration)

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
- ✅ PyTorch backend by default
- ❗ Requires Ray/RLlib and full MARLlib setup

---

## Choosing the Right Example

| Feature | PyTorch | NumPy | MARLlib |
|---------|---------|-------|---------|
| **Dependencies** | Moderate | Minimal | Full |
| **Backend** | PyTorch | NumPy | PyTorch |
| **GPU Support** | ✅ Yes | ❌ No | ✅ Yes |
| **Original Architecture** | ✅ Yes | ❌ No | ✅ Yes |
| **Ray/RLlib Required** | ❌ No | ❌ No | ✅ Yes |
| **Setup Complexity** | Low | Lowest | High |
| **Best For** | General use | Testing | Full integration |

**Recommendation**: Use `run_hebbian_pytorch.py` for most cases as it provides the best balance of features and ease of use while maintaining compatibility with the original PyTorch-based architecture.

---

## Backend Selection

All examples support both backends through configuration:

```python
config = {
    # ... other config ...
    'use_torch': True   # Use PyTorch (default: True if available)
    # 'use_torch': False  # Use NumPy
}
```

The trainer will automatically fall back to NumPy if PyTorch is not available.

---

## Common Error: Ray/RLlib Import Issues

If you see errors like:
```
from ray.rllib.models.catalog import ModelCatalog
ImportError: ...
```

**Solution 1 (Recommended):** Use the PyTorch standalone version (`run_hebbian_pytorch.py`) which doesn't require Ray.

**Solution 2:** Use the NumPy standalone version (`run_hebbian_standalone.py`) which has minimal dependencies.

---

## Training Parameters

All scripts support the same configuration parameters:

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
- `use_torch`: Use PyTorch backend (default: True if available)

---

## Output

All scripts will:
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
