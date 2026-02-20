"""
Run benchmark experiment from YAML configuration.

Usage:
    python benchmarks/run_from_config.py --config configs/dtlz2_m5.yaml
"""

import argparse
import os
import sys
import time
import torch
import yaml
from botorch.test_functions.multi_objective import DTLZ2
from stch_botorch import run_mobo

def load_config(path):
    if not os.path.exists(path):
        print(f"Error: Config file {path} not found.")
        sys.exit(1)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_problem(name, n_obj, dim):
    if name == "DTLZ2":
        return DTLZ2(dim=dim, num_objectives=n_obj, negate=True)
    else:
        raise ValueError(f"Unknown problem: {name}")

def main():
    parser = argparse.ArgumentParser(description="Run MOBO benchmark from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    p_conf = config['problem']
    m_conf = config['method']
    b_conf = config['bo']
    o_conf = config['output']
    
    out_dir = o_conf.get('dir', 'results')
    os.makedirs(out_dir, exist_ok=True)
    
    n_seeds = b_conf.get('n_seeds', 1)
    
    print(f"Running experiment: {p_conf['name']} (m={p_conf['n_obj']}, d={p_conf['dim']})")
    print(f"Method: {m_conf['name']}, Batch: {m_conf['batch_size']}")
    print(f"Seeds: {n_seeds}")
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        
        # Setup problem
        problem = get_problem(p_conf['name'], p_conf['n_obj'], p_conf['dim'])
        bounds = torch.stack([torch.zeros(p_conf['dim']), torch.ones(p_conf['dim'])])
        
        # Calculate budget
        n_init = b_conf.get('n_init', 2 * (p_conf['dim'] + 1))
        n_iter = b_conf.get('n_iter', 20)
        batch_size = m_conf.get('batch_size', p_conf['n_obj'])
        budget = n_init + n_iter * batch_size
        
        # Run
        try:
            result = run_mobo(
                f=problem,
                bounds=bounds,
                n_obj=p_conf['n_obj'],
                budget=budget,
                batch_size=batch_size,
                n_init=n_init,
                mu=m_conf.get('mu', 0.1),
                seed=seed,
                verbose=True
            )
            
            # Save result
            # We save simple metrics for now
            res_file = os.path.join(out_dir, f"seed_{seed}_hv.txt")
            with open(res_file, 'w') as f:
                for hv in result.hypervolume_history:
                    f.write(f"{hv}\n")
            
            print(f"Saved HV history to {res_file}")
            print(f"Final HV: {result.hypervolume_history[-1]:.4f}")
            
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
