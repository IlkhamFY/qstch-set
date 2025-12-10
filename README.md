# STCH-BoTorch

A production-grade Python library for **Smooth Tchebycheff (STCH)** scalarization methods for multi-objective Bayesian optimization with BoTorch.

## Overview

Standard Tchebycheff (TCH) scalarization uses a `max` operator, which creates non-differentiable "kinks" in the optimization landscape. This hampers gradient-based optimizers (like L-BFGS-B) used by BoTorch.

**STCH-BoTorch** implements smooth, differentiable alternatives that enable efficient gradient-based optimization while maintaining the theoretical properties of Tchebycheff scalarization.

## Installation

```bash
pip install stch-botorch
```

For development:

```bash
git clone https://github.com/IlkhamFY/stch-botorch.git
cd stch-botorch
pip install -e ".[dev]"
```

## Known Limitations (v0.1.0)

- `optimize_acqf` with `raw_samples > 1` requires `sequential=True` 
  (BoTorch shape validation quirk, documented in [BOTORCH_COMPATIBILITY.md](BOTORCH_COMPATIBILITY.md))
- Direct `model.posterior().sample()` → objective evaluation works perfectly
- See [notebooks/stch_demo.ipynb](notebooks/stch_demo.ipynb) for working examples

These are BoTorch ecosystem issues, not STCH-BoTorch issues.

## Methods Comparison

| Method | Description | Differentiable | Use Case |
|--------|-------------|----------------|----------|
| **TCH** | Standard Tchebycheff (uses `max`) |  No | Baseline comparison |
| **STCH** | Smooth Tchebycheff (Lin et al., ICML 2024) |  Yes | Standard multi-objective optimization |
| **STCH-Set** | Smooth Tchebycheff Set (Lin et al., ICLR 2025) |  Yes | Batch optimization for diverse solutions |

## Theory

### Standard Tchebycheff (TCH)

The standard Tchebycheff scalarization for minimization is:

$$U_{TCH}(y) = \max_i w_i (y_i - z^*_i)$$

where $w_i$ are weights, $y_i$ are objective values, and $z^*_i$ is a reference point. The `max` operator creates non-differentiable points where objectives are equal.

### Smooth Tchebycheff (STCH)

STCH approximates the `max` operator using LogSumExp for differentiability:

$$U_{STCH}(y) = -\mu \log\left(\sum_i \exp\left(\frac{w_i (z^*_i - y_i)}{\mu}\right)\right)$$

where $\mu > 0$ is a smoothing parameter. As $\mu \to 0$, STCH converges to TCH. BoTorch maximizes this utility, so lower $y$ values are preferred (minimization semantics).

### Smooth Tchebycheff Set (STCH-Set)

STCH-Set optimizes a batch of $q$ candidates to collectively cover all objectives. It uses nested smoothing:

1. **Inner aggregation** (smooth min over batch): For each objective $i$ and candidate $k$:
   $$R_{ik} = w_i (z^*_i - y_{ik})$$
   $$R_i^{min} = -\mu \log\left(\sum_k \exp\left(\frac{-R_{ik}}{\mu}\right)\right)$$

2. **Outer aggregation** (smooth max over objectives):
   $$S = \mu \log\left(\sum_i \exp\left(\frac{R_i^{min}}{\mu}\right)\right)$$

3. **Utility**: $U_{STCH-Set} = -S$

This aggregates over the $q$ dimension, returning a single scalar per batch.

## Usage Examples

### Basic Usage with qLogNParEGO

```python
import torch
from stch_botorch import SmoothChebyshevObjective
from botorch.acquisition import qLogNParEGO

# Define your objective
objective = SmoothChebyshevObjective(
    weights=torch.tensor([0.5, 0.5]),  # Equal weights for 2 objectives
    ref_point=torch.tensor([0.0, 0.0]),  # Reference point
    mu=0.1  # Smoothing parameter
)

# Use with BoTorch acquisition function
acq_function = qLogNParEGO(
    model=model,
    ref_point=torch.tensor([0.0, 0.0]),
    objective=objective
)

# Optimize acquisition function
candidates, _ = optimize_acqf(
    acq_function=acq_function,
    bounds=bounds,
    q=4,  # Number of candidates
    num_restarts=20,
    raw_samples=100,
)
```

### Using STCH-Set with qSimpleRegret

```python
from stch_botorch import SmoothChebyshevSetObjective
from botorch.acquisition import qSimpleRegret

# STCH-Set objective aggregates over q dimension
objective = SmoothChebyshevSetObjective(
    weights=torch.tensor([0.5, 0.5]),
    ref_point=torch.tensor([0.0, 0.0]),
    mu=0.1
)

# Use with qSimpleRegret (not standard qEI)
acq_function = qSimpleRegret(
    model=model,
    objective=objective
)
```

### Custom Smoothing Parameter

```python
# Smaller mu = tighter approximation to TCH (less smooth)
objective_tight = SmoothChebyshevObjective(
    weights=torch.tensor([0.5, 0.5]),
    ref_point=torch.tensor([0.0, 0.0]),
    mu=0.01  # Very tight approximation
)

# Larger mu = smoother (more differentiable)
objective_smooth = SmoothChebyshevObjective(
    weights=torch.tensor([0.5, 0.5]),
    ref_point=torch.tensor([0.0, 0.0]),
    mu=1.0  # Very smooth
)
```

### Automatic Ideal Point

```python
# If ref_point is None, uses ideal point (minimum of each objective)
objective = SmoothChebyshevObjective(
    weights=torch.tensor([0.5, 0.5]),
    ref_point=None,  # Will compute ideal point from data
    mu=0.1
)
```

### Using Functional API

```python
from stch_botorch import smooth_chebyshev, smooth_chebyshev_set

# Direct function calls
Y = torch.tensor([[1.0, 2.0], [2.0, 1.0]])  # (n, m)
weights = torch.tensor([0.5, 0.5])
ref_point = torch.tensor([0.0, 0.0])

utility = smooth_chebyshev(Y, weights, ref_point, mu=0.1)
# Returns: (n,) - scalarized utility for each point

# STCH-Set for batch
Y_batch = torch.tensor([[[1.0, 2.0], [2.0, 1.0]]])  # (..., q, m)
utility_set = smooth_chebyshev_set(Y_batch, weights, ref_point, mu=0.1)
# Returns: (...) - aggregated utility (q dimension removed)
```

## API Reference

### `SmoothChebyshevObjective`

BoTorch objective that maps `(sample_shape x batch_shape x q x m)` → `(sample_shape x batch_shape x q)`.

**Parameters:**
- `weights` (torch.Tensor): Weight vector of shape `(m,)`. Automatically normalized.
- `ref_point` (torch.Tensor, optional): Reference point of shape `(m,)`. If `None`, uses ideal point.
- `mu` (float): Smoothing parameter. Default is `0.1`.

**Use with:** `qLogNParEGO`, `qParEGO`, and other standard BoTorch acquisition functions.

### `SmoothChebyshevSetObjective`

BoTorch objective that maps `(sample_shape x batch_shape x q x m)` → `(sample_shape x batch_shape)`.

**Parameters:**
- `weights` (torch.Tensor): Weight vector of shape `(m,)`. Automatically normalized.
- `ref_point` (torch.Tensor, optional): Reference point of shape `(m,)`. If `None`, uses ideal point.
- `mu` (float): Smoothing parameter. Default is `0.1`.

**Warning:** Advanced use only. Returns `(sample_shape x batch_shape)`, not standard `(... x q)` shape. Use with `qSimpleRegret` or `FixedFeatureAcquisition`, not standard `qEI`.

### `smooth_chebyshev(Y, weights, ref_point, mu=0.1)`

Functional implementation of Smooth Tchebycheff scalarization.

**Parameters:**
- `Y` (torch.Tensor): Objective values of shape `(..., m)`.
- `weights` (torch.Tensor): Weight vector of shape `(m,)`.
- `ref_point` (torch.Tensor, optional): Reference point of shape `(m,)`.
- `mu` (float): Smoothing parameter. Default is `0.1`.

**Returns:**
- `torch.Tensor`: Scalarized utility of shape `(...)`.

### `smooth_chebyshev_set(Y, weights, ref_point, mu=0.1)`

Functional implementation of Smooth Tchebycheff Set scalarization.

**Parameters:**
- `Y` (torch.Tensor): Objective values of shape `(..., q, m)`.
- `weights` (torch.Tensor): Weight vector of shape `(m,)`.
- `ref_point` (torch.Tensor, optional): Reference point of shape `(m,)`.
- `mu` (float): Smoothing parameter. Default is `0.1`.

**Returns:**
- `torch.Tensor`: Scalarized utility of shape `(...)`. The `q` dimension is removed.

## Key Features

- **Fully Differentiable**: Enables gradient-based optimization with BoTorch
- **Numerically Stable**: Uses `torch.logsumexp` for robust computation
- **Automatic Normalization**: Weights are automatically normalized to sum to 1
- **Flexible Reference Points**: Optional ideal point computation
- **Production Ready**: Comprehensive tests and type hints

## Testing

Run tests with:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=stch_botorch --cov-report=html
```

## Citation

If you use STCH-BoTorch in your research, please cite:

```bibtex
@article{lin2024smooth,
  title={Smooth Tchebycheff Scalarization for Multi-Objective Optimization},
  author={Lin, X. and others},
  journal={ICML},
  year={2024}
}

@article{lin2025smoothset,
  title={Smooth Tchebycheff Set: Few for Many},
  author={Lin, X. and others},
  journal={ICLR},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
