import json

# m=5 results
with open('benchmarks/results/dtlz2_m5_results.json') as f:
    m5 = json.load(f)

print('=== m=5 Timing (total time / n_iters) ===')
for method, data in m5['methods'].items():
    times = [r['time'] for r in data['runs']]
    n_iters = 30
    per_iter = [t/n_iters for t in times]
    mean_per_iter = sum(per_iter)/len(per_iter)
    print(f'{method}: mean_total={sum(times)/len(times):.1f}s, mean_per_iter={mean_per_iter:.1f}s')

# m=8 results
with open('benchmarks/results/dtlz2_m8_results.json') as f:
    m8 = json.load(f)

print()
print('=== m=8 Timing (total time / n_iters) ===')
for method, data in m8['methods'].items():
    times = [r['time'] for r in data['runs']]
    n_iters = 25
    per_iter = [t/n_iters for t in times]
    mean_per_iter = sum(per_iter)/len(per_iter)
    print(f'{method}: mean_total={sum(times)/len(times):.1f}s, mean_per_iter={mean_per_iter:.1f}s')
