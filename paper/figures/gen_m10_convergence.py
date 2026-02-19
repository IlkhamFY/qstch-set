"""Generate convergence plot for m=10 K=10 from Nibi results."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from m10_K10 JSON (manually extracted from Nibi)
# stch_set trajectory (mean over 3 seeds, 16 iterations including init)
stch_set_mean = [36.7838, 38.9262, 41.1868, 42.7955, 44.1323, 44.6908, 44.9866, 45.0170, 45.0599, 46.4384, 46.5941, 46.8777, 46.9202, 46.9385, 46.9398, 46.9539]
stch_set_std = [0.8808, 0.2263, 0.9616, 1.6960, 0.8110, 0.4054, 0.4686, 0.4814, 0.4973, 1.4326, 1.3999, 1.3203, 1.3131, 1.3122, 1.3130, 1.3120]

# qnparego trajectory (mean over 3 seeds, 16 iterations)
qnparego_mean = [36.7838, 37.0181, 37.2699, 37.9101, 37.9554, 38.7711, 39.1048, 39.7142, 40.6812, 40.8346, 41.7938, 42.6572, 43.3564, 43.5267, 43.9298, 44.1046]
qnparego_std = [0.8808, 0.9817, 0.9340, 0.6216, 0.6245, 1.1003, 0.9873, 1.0968, 0.4027, 0.3187, 0.9164, 0.5008, 0.5995, 0.6875, 0.9953, 0.9929]

iters = np.arange(len(stch_set_mean))

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

# Plot with confidence bands
ax.plot(iters, stch_set_mean, 'b-o', label='qSTCH-Set ($K{=}10$)', markersize=4, linewidth=2)
ax.fill_between(iters, 
                np.array(stch_set_mean) - np.array(stch_set_std),
                np.array(stch_set_mean) + np.array(stch_set_std),
                alpha=0.2, color='blue')

ax.plot(iters, qnparego_mean, 'r-s', label='qNParEGO', markersize=4, linewidth=2)
ax.fill_between(iters,
                np.array(qnparego_mean) - np.array(qnparego_std),
                np.array(qnparego_mean) + np.array(qnparego_std),
                alpha=0.2, color='red')

ax.set_xlabel('BO Iteration', fontsize=12)
ax.set_ylabel('Hypervolume', fontsize=12)
ax.set_title('DTLZ2, $m{=}10$, $K{=}10$ (3 seeds)', fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, len(stch_set_mean)-1)

plt.tight_layout()
plt.savefig('convergence_m10.pdf', dpi=300, bbox_inches='tight')
plt.savefig('convergence_m10.png', dpi=150, bbox_inches='tight')
print("Saved convergence_m10.pdf and .png")
