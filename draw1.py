"""
Fig. 3 — Overall optimization performance on TSPLIB
(a) Path length / reference route length vs instances (ordered by size)
(b) Baseline / Proposed path length (=1), mean ± std

Print-safe encoding (Scientific Reports / grayscale):
  color is NOT the only cue — distinct marker + linestyle per method;
  Proposed Framework uses a thicker solid line.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

FRAMEWORK_KEY = 'PDE_CQE'
FRAMEWORK_NAME = 'Proposed Framework'

# TSPLIB averages (10 runs); Framework values match paper narrative (e.g., St-70 = 764.05)
data = {
    'Instance': [
        'ulysses-16', 'ulysses-22', 'bayg-29', 'bays-29', 'dantzig-42',
        'att-48', 'eil-51', 'berlin-52', 'st-70', 'eil-76',
    ],
    'N': [16, 22, 29, 29, 42, 48, 51, 52, 70, 76],
    'Short': [
        'U16', 'U22', 'Bayg29', 'Bays29', 'Dant42',
        'Att48', 'Eil51', 'Ber52', 'St70', 'Eil76',
    ],
    'Optimum': [74.11, 75.67, 9074.15, 9291.35, 699, 33523.71, 429.98, 7544.37, 678.60, 545.39],
    'ACO': [74.62, 77.48, 11334.54, 11173.94, 791.91, 51225.25, 504.43, 9657.42, 831.39, 676.85],
    'GA': [75.27, 81.19, 10728.40, 11792.14, 928.13, 49025.50, 610.18, 10693.30, 1306.07, 941.49],
    'Greedy': [88.40, 92.40, 11207.54, 11163.61, 893.55, 41789.20, 574.35, 9721.31, 834.72, 681.81],
    'PDE_CQE': [78.57, 79.05, 10029.16, 10367.53, 779.56, 37939.95, 504.03, 8623.66, 764.05, 620.02],
    'QACO': [82.34, 85.84, 10587.33, 10662.26, 832.07, 40296.72, 525.32, 9039.85, 786.13, 636.62],
}

df = pd.DataFrame(data)
df = df.sort_values(['N', 'Instance'], kind='mergesort').reset_index(drop=True)

baselines = ['ACO', 'QACO', 'GA', 'Greedy']
all_algs = [FRAMEWORK_KEY] + baselines

# Panel colors (match prior Fig.3); (a) also uses distinct marker + linestyle
colors = {
    FRAMEWORK_KEY: '#1F4E79',  # deep blue — proposed framework
    'QACO': '#7B68A6',         # muted purple
    'ACO': '#8FAADC',          # light blue
    'GA': '#A0A0A0',           # grey
    'Greedy': '#C4C4C4',       # light grey
}
markers = {
    FRAMEWORK_KEY: 'D',   # diamond
    'QACO': 's',          # square
    'ACO': 'o',           # circle
    'GA': '^',            # triangle up
    'Greedy': 'X',        # filled X (distinct from GA in grayscale)
}
linestyles = {
    FRAMEWORK_KEY: '-',                    # solid
    'QACO': '--',                          # dashed
    'ACO': '-.',                           # dash-dot
    'GA': ':',                             # dotted
    'Greedy': (0, (3, 1.2, 1, 1.2)),       # dash-dot-dot
}
linewidths = {
    FRAMEWORK_KEY: 3.0,   # thicker proposed curve
    'QACO': 1.7,
    'ACO': 1.5,
    'GA': 1.5,
    'Greedy': 1.5,
}
markersizes = {
    FRAMEWORK_KEY: 8.0,
    'QACO': 6.5,
    'ACO': 6.5,
    'GA': 6.5,
    'Greedy': 6.5,
}
zorders = {
    FRAMEWORK_KEY: 5,
    'QACO': 4,
    'ACO': 3,
    'GA': 2,
    'Greedy': 1,
}

# (a) Path length / reference route length
ratio_opt = {alg: df[alg] / df['Optimum'] for alg in all_algs}

# (b) Baseline / Proposed path length (=1.0); mean ± std across instances
rel_to_fw = {alg: (df[alg] / df[FRAMEWORK_KEY]).values for alg in baselines}
rel_mean = {alg: float(np.mean(rel_to_fw[alg])) for alg in baselines}
rel_std = {alg: float(np.std(rel_to_fw[alg], ddof=1)) for alg in baselines}

x = np.arange(len(df))
short_labels = df['Short'].tolist()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))

# ---------------------------------------------------------------------------
# (a) Normalized performance vs TSPLIB instances (ordered by size)
# ---------------------------------------------------------------------------
ax1.axhline(
    1.0, color='#8B0000', linestyle=(0, (4, 2)), linewidth=1.3, alpha=0.9,
    zorder=1, label='Reference (=1)',
)

plot_order = baselines + [FRAMEWORK_KEY]  # framework last (on top)
for alg in plot_order:
    name = FRAMEWORK_NAME if alg == FRAMEWORK_KEY else alg
    ax1.plot(
        x,
        ratio_opt[alg],
        color=colors[alg],
        marker=markers[alg],
        markersize=markersizes[alg],
        linewidth=linewidths[alg],
        linestyle=linestyles[alg],
        label=name,
        zorder=zorders[alg],
        markerfacecolor='white' if alg != FRAMEWORK_KEY else colors[alg],
        markeredgecolor=colors[alg],
        markeredgewidth=1.5 if alg != FRAMEWORK_KEY else 0.9,
    )

ax1.set_xticks(x)
ax1.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=11)
ax1.set_xlabel('TSPLIB instance (ordered by problem size)', fontsize=13, fontweight='bold')
ax1.set_ylabel(r'Path length / reference route length', fontsize=13, fontweight='bold')
ax1.set_title('(a) Normalized optimization performance', fontsize=14, fontweight='bold', pad=8)
ax1.set_xlim(-0.4, len(df) - 0.6)
ymin = min(ratio_opt[alg].min() for alg in all_algs)
ymax = max(ratio_opt[alg].max() for alg in all_algs)
ax1.set_ylim(max(0.95, ymin - 0.05), ymax + 0.08)
ax1.grid(True, linestyle=':', alpha=0.35)
ax1.legend(loc='upper left', framealpha=0.95, edgecolor='0.5', fontsize=10)

# ---------------------------------------------------------------------------
# (b) Relative performance & stability (Framework = 1)
# ---------------------------------------------------------------------------
order_b = ['ACO', 'GA', 'Greedy', 'QACO']
x_b = np.arange(len(order_b))
means_b = [rel_mean[a] for a in order_b]
stds_b = [rel_std[a] for a in order_b]

ax2.bar(
    x_b,
    means_b,
    yerr=stds_b,
    capsize=5,
    color=[colors[a] for a in order_b],
    alpha=0.88,
    edgecolor='black',
    linewidth=0.9,
    error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': '#333333'},
    zorder=2,
)

ax2.axhline(
    1.0, color='#B22222', linestyle='--', linewidth=1.6, alpha=0.9,
    zorder=3, label=f'{FRAMEWORK_NAME} (=1)',
)

# Per-instance points (meaning explained in caption only)
np.random.seed(42)
for i, alg in enumerate(order_b):
    vals = rel_to_fw[alg]
    jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
    ax2.scatter(
        np.full(len(vals), x_b[i]) + jitter,
        vals,
        s=28,
        color=colors[alg],
        alpha=0.55,
        zorder=4,
        edgecolors='black',
        linewidths=0.35,
    )

ax2.set_xticks(x_b)
ax2.set_xticklabels(order_b, fontsize=11, fontweight='bold')
ax2.set_xlabel('Baseline algorithm', fontsize=13, fontweight='bold', labelpad=10)
ax2.set_ylabel(r'Baseline / Proposed path length', fontsize=13, fontweight='bold')
ax2.set_title('(b) Relative performance and stability', fontsize=14, fontweight='bold', pad=8)
ax2.set_ylim(0.9, max(means_b[i] + stds_b[i] for i in range(len(order_b))) + 0.12)
ax2.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
ax2.legend(loc='upper left', framealpha=0.95, edgecolor='0.5', fontsize=10)

plt.tight_layout()
fig.align_xlabels([ax1, ax2])

out_png = 'figure3_overall_optimization_performance.png'
out_pdf = 'figure3_overall_optimization_performance.pdf'
plt.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.08)
plt.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.08)
print(f'OK Figure 3 saved: {out_png} / {out_pdf} (600 dpi)')
print(
    '\nSuggested caption:\n'
    'Fig. 3. Optimization performance on TSPLIB benchmarks. '
    '(a) Path length relative to the reference route length. '
    'Instances are ordered by city count for visualization; the x-axis is categorical. '
    '(b) Baseline / Proposed path length (reference = 1). '
    'Baselines are ordered ACO, GA, Greedy, QACO. '
    'Each point represents one benchmark instance; bars show the mean ± standard '
    'deviation across instances.'
)
plt.close()
