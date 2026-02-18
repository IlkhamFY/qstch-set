# STCH-BoTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/IlkhamFY/stch-botorch/actions/workflows/tests.yml/badge.svg)](https://github.com/IlkhamFY/stch-botorch/actions/workflows/tests.yml)

**Smooth Tchebycheff scalarization for multi-objective Bayesian optimization with BoTorch.**

STCH-BoTorch brings the Smooth Tchebycheff (STCH) and STCH-Set scalarizations from [Lin et al. (ICML 2024)](https://arxiv.org/abs/2402.19078) and [Lin et al. (ICLR 2025)](https://arxiv.org/abs/2406.02418) into the BoTorch ecosystem for sample-efficient optimization of expensive black-box multi-objective problems.

## Key Features

- **`qSTCHSet`** — jointly optimizes a batch of candidates for collective Pareto coverage (O(qm), no hypervolume computation)
- **`qSTCHSetTS`** — Thompson Sampling variant for cheaper per-evaluation cost
- **Fully differentiable** — smooth log-sum-exp enables L-BFGS-B via `optimize_acqf`
- **Scales to many objectives** — O(qm) vs exponential for hypervolume-based methods
- **Drop-in replacement** for `qNParEGO` / `qEHVI` in BoTorch workflows

## Installation

```bash
git clone https://github.com/IlkhamFY/stch-botorch.git
cd stch-botorch
pip install -e .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from stch_botorch.acquisition import qSTCHSet

# Fit a multi-output GP
train_X = torch.rand(20, 3)
train_Y = torch.stack([train_X.sum(dim=-1), (train_X ** 2).sum(dim=-1)], dim=-1)
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# Build acquisition function
acqf = qSTCHSet(
    model=model,
    ref_point=train_Y.min(dim=0).values - 0.1,  # slightly below best observed
    mu=0.1,
)

# Jointly optimize q=4 candidates
bounds = torch.stack([torch.zeros(3), torch.ones(3)])
candidates, value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=4,
    num_restarts=10,
    raw_samples=256,
)
```

## API Overview

### Acquisition Functions

| Class | Description |
|-------|-------------|
| `qSTCHSet` | MC acquisition: averages STCH-Set over posterior samples |
| `qSTCHSetTS` | Thompson Sampling variant (single posterior sample) |

### Scalarization Functions

| Function | Description |
|----------|-------------|
| `smooth_chebyshev(Y, weights, ref_point, mu)` | Single-point STCH scalarization |
| `smooth_chebyshev_set(Y, weights, ref_point, mu)` | Set-based STCH scalarization (aggregates over batch) |

### Objective Wrappers

| Class | Description |
|-------|-------------|
| `SmoothChebyshevObjective` | For use with `qLogNParEGO` etc. — maps `(..., q, m) → (..., q)` |
| `SmoothChebyshevSetObjective` | For use with `qSimpleRegret` — maps `(..., q, m) → (...)` |

## Examples

See [`examples/`](examples/) for complete scripts:

- **`basic_qstchset.py`** — minimal qSTCHSet usage
- **`dtlz2_optimization.py`** — full MOBO loop on DTLZ2

## Reproducing Benchmarks

```bash
# ZDT suite (2 objectives)
python benchmarks/zdt_benchmark.py

# DTLZ2 with 5 objectives
python benchmarks/dtlz_benchmark.py --problem DTLZ2 --m 5 --seeds 5 --iters 30

# DTLZ2 with 10 objectives (many-objective)
python benchmarks/dtlz_benchmark.py --problem DTLZ2 --m 10 --seeds 3 --iters 20
```

Results are saved to `benchmarks/results/`.

## Citation

If you use STCH-BoTorch in your research, please cite:

```bibtex
@inproceedings{lin2024smooth,
  title={Smooth Tchebycheff Scalarization for Multi-Objective Optimization},
  author={Lin, Xi and Zhang, Zhiyuan and Zhong, Xiaoyuan and Deb, Kalyanmoy and Zhang, Qingfu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}

@inproceedings{lin2025few,
  title={Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization},
  author={Lin, Xi and Zhang, Yilu and Zhang, Zhiyuan and Deb, Kalyanmoy and Zhang, Qingfu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## License

[MIT](LICENSE)
