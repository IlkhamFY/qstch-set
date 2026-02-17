import time, sys
sys.path.insert(0, '.')
from zdt_benchmark import ZDTProblem, run_stch_qnparego, run_stch_set

problem = ZDTProblem('zdt1')

t0 = time.time()
metrics, Y = run_stch_qnparego(problem, seed=0)
print(f"STCH-qNParEGO: {time.time()-t0:.1f}s, final HV={metrics[-1]['hv']:.4f}")

t0 = time.time()
metrics, Y = run_stch_set(problem, seed=0)
print(f"STCH-Set: {time.time()-t0:.1f}s, final HV={metrics[-1]['hv']:.4f}")

print("Done!")
