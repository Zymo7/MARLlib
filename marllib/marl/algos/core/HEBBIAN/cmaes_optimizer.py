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
CMA-ES optimizer wrapper for Hebbian agent evolution.
"""

import cma
import numpy as np


class CMAESOptimizer:
    """
    Wrapper around the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
    
    This is used to evolve the Hebbian agent parameters.
    """
    
    def __init__(self, param_dim, pop_size=30, sigma0=0.5, verbose=False, seed=None):
        """
        Initialize CMA-ES optimizer.
        
        Args:
            param_dim: Number of parameters to optimize (int)
            pop_size: Population size (int)
            sigma0: Initial standard deviation (float)
            verbose: Enable verbose output for debugging (bool)
            seed: Random seed for reproducibility (int, optional)
        """
        self.param_dim = param_dim
        self.pop_size = pop_size
        self.sigma0 = sigma0
        
        # Initialize CMA-ES with zero mean
        x0 = np.zeros(param_dim)
        
        # Configure verbosity
        verb_disp = 1 if verbose else 0
        verbose_level = 1 if verbose else -9
        
        # Build CMA-ES options
        options = {
            'popsize': pop_size,
            'verb_disp': verb_disp,
            'verbose': verbose_level
        }
        
        # Add seed if provided for reproducibility
        if seed is not None:
            options['seed'] = seed
        
        # Create optimizer
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    
    def ask(self):
        """
        Sample a population of parameter vectors.
        
        Returns:
            List of parameter vectors (list of numpy arrays)
        """
        return self.es.ask()
    
    def tell(self, solutions, fitness_values):
        """
        Update optimizer with fitness evaluations.
        
        Args:
            solutions: List of parameter vectors (list of numpy arrays)
            fitness_values: List of corresponding fitness values (list of floats)
                           Note: CMA-ES minimizes, so negate if maximizing
        """
        self.es.tell(solutions, fitness_values)
    
    def best_solution(self):
        """
        Get the best solution found so far.
        
        Returns:
            Tuple of (best_params, best_fitness)
        """
        best_params = self.es.result.xbest
        best_fitness = self.es.result.fbest
        return best_params, best_fitness
    
    def stop(self):
        """
        Check if optimization should stop.
        
        Returns:
            Dictionary of stop conditions (empty if should continue)
        """
        return self.es.stop()
    
    @property
    def result(self):
        """Get optimization result."""
        return self.es.result
