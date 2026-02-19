# qSTCH-Set: Smooth Tchebycheff Set Scalarization for Many-Objective Bayesian Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/IlkhamFY/stch-botorch/actions/workflows/tests.yml/badge.svg)](https://github.com/IlkhamFY/stch-botorch/actions/workflows/tests.yml)

**qSTCH-Set** is the first Monte Carlo acquisition function for multi-objective Bayesian optimization based on Smooth Tchebycheff Set scalarization. It jointly optimizes a batch of candidates for collective Pareto coverage in **O(Km)** time — no hypervolume computation required — enabling scalable optimization with 5, 8, or 10+ objectives.

Built on top of [BoTorch](https://botorch.org/), it serves as a drop-in replacement for `qNParEGO` and `qEHVI` in many-objective settings where hypervolume-based methods become intractable.

> **Paper:** *qSTCH-Set: Smooth Tchebycheff Set Scalarization for Many-Objective Bayesian Optimization* (NeurIPS 2026 submission). PDF link forthcoming.

## Key Features

- **`qSTCHSet`** — MC acquisition: averages STCH-Set scalarization over posterior samples
- **`qSTCHSetTS`** — Thompson Sampling variant for cheaper per-evaluation cost
- **K=m design rule** — using K=m weight vectors matches or beats baselines as objectives scale
- **Fully differentiable** — smooth log-sum-exp enables L-BFGS-B optimization via `optimize_acqf`
- **Scales to many objectives** — O(Km) vs exponential for hypervolume-based methods

## Benchmark Results (DTLZ2)

All results from 5-seed (m=5, m=8) or 3-seed (m=10) experiments on Alliance Canada H100 GPUs.

| Method | m=5 (K=5) | m=8 (K=8) | m=10 (K=10) |
|--------|-----------|-----------|-------------|
| **qSTCH-Set** | 6.22 ± 0.50 | **20.22 ± 1.90** | **46.95 ± 1.31** |
| qNParEGO | **6.44 ± 0.16** | 20.89 ± 0.83† | 44.10 ± 0.99 |
| STCH-NParEGO | — | 16.63 ± 0.23 | 38.39 ± 0.72 |
| Random | 5.37 ± 0.14 | 17.63 ± 0.29 | 40.44 ± 0.33 |

*Hypervolume (↑ better). † qNParEGO at m=8 uses default K=5 from full 5-method comparison.*

**Key finding:** qSTCH-Set with K=m **wins by 6.5%** at m=10 (46.95 vs 44.10), demonstrating increasing advantage as objectives scale.

## Installation

```bash
git clone https://github.com/IlkhamFY/stch-botorch.git
cd stch-botorch
pip install -e .
```

With development/benchmarking dependencies:

```bash
pip install -e ".[dev,bench]"
```

## Quick Start

```python
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from stch_botorch import qSTCHSet
import torch

# Fit a multi-output GP on initial data
train_X = torch.rand(20, 3)
train_Y = torch.stack([train_X.sum(-1), (train_X**2).sum(-1)], -1)
model = SingleTaskGP(train_X, train_Y)
fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

# qSTCH-Set: jointly optimize q=4 candidates
acqf = qSTCHSet(model=model, ref_point=train_Y.min(0).values - 0.1, mu=0.1)
candidates, value = optimize_acqf(acqf, torch.stack([torch.zeros(3), torch.ones(3)]), q=4, num_restarts=10, raw_samples=256)
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

## Reproducing Paper Experiments

```bash
# DTLZ2 with 5 objectives (K=m=5)
python benchmarks/dtlz_benchmark.py --problem DTLZ2 --m 5 --K 5 --seeds 5 --iters 20

# DTLZ2 with 10 objectives (K=m=10, many-objective)
python benchmarks/dtlz_benchmark.py --problem DTLZ2 --m 10 --K 10 --seeds 3 --iters 15

# K-ablation study (m=5, varying K)
python benchmarks/dtlz_benchmark.py --problem DTLZ2 --m 5 --K 3 --seeds 3 --iters 20
python benchmarks/dtlz_benchmark.py --problem DTLZ2 --m 5 --K 10 --seeds 3 --iters 20
```

Results are saved to `benchmarks/results/` and `results/`.

## Examples

See [`examples/`](examples/) for complete scripts:

- **`basic_qstchset.py`** — minimal qSTCHSet usage
- **`dtlz2_optimization.py`** — full MOBO loop on DTLZ2

## Citation

If you use qSTCH-Set in your research, please cite:

```bibtex
@inproceedings{yourname2026qstchset,
  title={qSTCH-Set: Smooth Tchebycheff Set Scalarization for Many-Objective Bayesian Optimization},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
  note={Under review}
}
```

The underlying STCH and STCH-Set scalarizations are from:

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
