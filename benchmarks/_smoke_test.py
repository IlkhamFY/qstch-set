"""Quick smoke test: 1 seed, 5 sequential evals per method on ZDT1."""
import sys, time, torch, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

# Monkeypatch config for fast test
import zdt_benchmark as Z
Z.N_SEQUENTIAL = 5
Z.N_TOTAL = Z.N_INITIAL + Z.N_SEQUENTIAL
Z.SEEDS = [42]
Z.MC_SAMPLES = 32

prob = Z.ZDTProblem("zdt1")

for name, fn in Z.METHODS.items():
    t0 = time.time()
    try:
        hist, Y = fn(prob, seed=42)
        print(f"OK  {name:<20} {time.time()-t0:6.1f}s  HV={hist[-1]['hv']:.4f}  "
              f"IGD={hist[-1]['igd']:.4f}  #PF={hist[-1]['n_pareto']}")
    except Exception as e:
        print(f"ERR {name:<20} {time.time()-t0:6.1f}s  {e}")
        import traceback; traceback.print_exc()

print("\nSmoke test complete.")
