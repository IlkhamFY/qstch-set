# Experiment Trajectories

Structured decision trace logging for STCH-Set-BO research.
Each experiment gets a JSON file capturing not just results but the reasoning.

## Schema
```json
{
  "id": "exp-001",
  "timestamp": "2026-02-18T10:15:00",
  "hypothesis": "STCH-Set with q=K=m should outperform qNParEGO at m>=5",
  "config": {
    "problem": "DTLZ2", "m": 5, "K": 5, "mu": 0.1,
    "n_init": 20, "n_iters": 30, "seeds": [0,1,2,3,4]
  },
  "decision_trace": [
    "Chose K=m based on Lin ICLR25 Theorem 2",
    "mu=0.1 default from Lin paper, ablation pending",
    "5 seeds for statistical significance at NeurIPS standard"
  ],
  "result": {"stch_set_hv": "6.646±0.066", "qnparego_hv": "TBD"},
  "conclusion": "Set advantage confirmed at m=5",
  "next_action": "Run m=8,10 to find crossover point",
  "lessons": ["q=1 for set method is meaningless, always use q=K"]
}
```

This is our mini context graph for the research.
