"""
Generate paper-ready figure from stage_timing_real.csv.
Outputs: figs/stage_breakdown.png (300dpi) + .pdf (vector)

Usage:
    python3 make_stage_figure.py
    # or with custom CSV:
    python3 make_stage_figure.py path/to/stage_timing_real.csv
"""
import sys
import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Paper style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'STIX'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,        # TrueType, embeddable
    'ps.fonttype': 42,
})

# ─── Color palette (teal stack + neutral gray) ──────────────────────
COLOR_BUILD     = '#9FE1CB'  # light teal
COLOR_H2D       = '#5DCAA5'  # mid teal
COLOR_CROP      = '#1D9E75'  # strong teal
COLOR_NORMALIZE = '#0F6E56'  # dark teal
COLOR_INDIVIDUAL = '#888780'  # neutral gray

# ─── Read CSV ───────────────────────────────────────────────────────
csv_path = sys.argv[1] if len(sys.argv) > 1 else 'stage_timing_real.csv'
data = defaultdict(dict)  # data[n_obj][f"{method}/{stage}"] = ms

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        n = int(row['n_obj'])
        key = f"{row['method']}/{row['stage']}"
        data[n][key] = float(row['mean_ms'])

n_objs = sorted(data.keys())
labels = [f"{n}" for n in n_objs]
x = np.arange(len(n_objs))
width = 0.34

# ─── Pull out each stage series ─────────────────────────────────────
oct_build  = [data[n]['octopus/1_build_meta_host']    for n in n_objs]
oct_h2d    = [data[n]['octopus/2_h2d_metadata']       for n in n_objs]
oct_crop   = [data[n]['octopus/3_kernel_crop_resize'] for n in n_objs]
oct_norm   = [data[n]['octopus/4_kernel_normalize']   for n in n_objs]
oct_total  = [data[n]['octopus/5_e2e_total']          for n in n_objs]
ind_total  = [data[n]['individual/3_kernel_launches_total'] for n in n_objs]

# ─── Figure ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 3.6))

# Octopus — stacked bars (left of each pair)
xo = x - width/2 - 0.01
bars_b = ax.bar(xo, oct_build, width, color=COLOR_BUILD, edgecolor='white', linewidth=0.4, label='1. Build metadata (host)')
bars_h = ax.bar(xo, oct_h2d, width, bottom=oct_build, color=COLOR_H2D, edgecolor='white', linewidth=0.4, label='2. H2D metadata transfer')
bottom_so_far = [b + h for b, h in zip(oct_build, oct_h2d)]
bars_c = ax.bar(xo, oct_crop, width, bottom=bottom_so_far, color=COLOR_CROP, edgecolor='white', linewidth=0.4, label='3. Kernel: crop + resize')
bottom_so_far = [b + c for b, c in zip(bottom_so_far, oct_crop)]
bars_n = ax.bar(xo, oct_norm, width, bottom=bottom_so_far, color=COLOR_NORMALIZE, edgecolor='white', linewidth=0.4, label='4. Kernel: normalize')

# Individual — single bar (right of each pair)
xi = x + width/2 + 0.01
bars_i = ax.bar(xi, ind_total, width, color=COLOR_INDIVIDUAL, edgecolor='white', linewidth=0.4, label='Individual: N kernel launches')

# ─── Speedup annotations ────────────────────────────────────────────
for i, n in enumerate(n_objs):
    speedup = ind_total[i] / oct_total[i]
    saving = ind_total[i] - oct_total[i]
    y_top = max(ind_total[i], oct_total[i])
    ax.annotate(
        f"{speedup:.2f}×\n−{saving:.0f} ms",
        xy=(x[i], y_top),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center', va='bottom',
        fontsize=8.5, color='#2C2C2A',
        weight='normal',
    )

# ─── Method labels under each pair ──────────────────────────────────
for i in range(len(n_objs)):
    ax.text(xo[i], -max(ind_total)*0.025, 'Oct', ha='center', va='top', fontsize=7.5, color='#1D9E75')
    ax.text(xi[i], -max(ind_total)*0.025, 'Ind', ha='center', va='top', fontsize=7.5, color='#5F5E5A')

# ─── Axes ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels([f"N = {n}" for n in n_objs])
ax.tick_params(axis='x', pad=14)  # leave room for Oct/Ind sub-labels
ax.set_ylabel('Time per frame (ms)')
ax.set_xlabel('')
ax.set_title('Per-stage timing: Octopus vs Individual kernel launches\n'
             'Real VisDrone data, Jetson Orin Nano',
             pad=12, loc='left')
ax.set_ylim(0, max(ind_total) * 1.20)
ax.grid(axis='y', alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)

# ─── Legend (below chart) ──────────────────────────────────────────
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=3, frameon=False, handlelength=1.5, columnspacing=1.2)

plt.tight_layout()

# ─── Save ───────────────────────────────────────────────────────────
out_dir = 'figs'
os.makedirs(out_dir, exist_ok=True)
png_path = os.path.join(out_dir, 'stage_breakdown.png')
pdf_path = os.path.join(out_dir, 'stage_breakdown.pdf')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"  ✓ {png_path}")
print(f"  ✓ {pdf_path}")
print(f"\n  Figure size: 6.5 × 3.6 inches (single-column wide)")
print(f"  PNG: 300 dpi")
print(f"  PDF: vector (embeddable fonts)")