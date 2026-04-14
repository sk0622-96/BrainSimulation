"""
Version 2 Dashboard
====================
Reads results from version2_history.json.
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

os.environ['TK_SILENCE_DEPRECATION'] = '1'

HISTORY_FILE = "/Users/tommysupey/Desktop/Brain Simulation/version2_history.json"
BLUE   = '#2E86AB'
GREEN  = '#4CAF82'
RED    = '#E76F51'
DARK   = '#222222'


def load_history():
    if not os.path.exists(HISTORY_FILE):
        print(f"\n  Could not find {HISTORY_FILE}")
        sys.exit(1)
    with open(HISTORY_FILE) as f:
        data = json.load(f)
    runs = data.get('runs', [])
    if not runs:
        print(f"\n  {HISTORY_FILE} has no runs yet.")
        sys.exit(1)
    print(f"  Loaded {len(runs)} run(s) from {HISTORY_FILE}")
    return runs


def main():
    runs = load_history()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Version 2 Dashboard — Q-Learning with Working Memory',
                 fontsize=13, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.91, bottom=0.08,
                           left=0.08, right=0.97,
                           hspace=0.52, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    all_pg = [r.get('per_generation', []) for r in runs if r.get('per_generation')]
    max_gen = max((len(pg) for pg in all_pg), default=1)

    # --- Chart 1: shaded band + historical average ---
    all_eps_per_gen = []
    for g in range(1, max_gen + 1):
        vals = [pg[g-1]['episodes'] for pg in all_pg if len(pg) >= g]
        all_eps_per_gen.append(vals)

    avg_line = [np.mean(v) for v in all_eps_per_gen]
    min_line = [np.min(v)  for v in all_eps_per_gen]
    max_line = [np.max(v)  for v in all_eps_per_gen]
    gens = list(range(1, max_gen + 1))

    ax1.fill_between(gens, min_line, max_line,
                     color=BLUE, alpha=0.15, label='Min/max range')
    ax1.plot(gens, avg_line, color=BLUE, linewidth=2.5, marker='o',
             markersize=7, markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=BLUE,
             label=f'Historical avg ({len(runs)} runs)')
    for x, y in zip(gens, avg_line):
        ax1.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=9,
                     fontweight='bold', color=BLUE)

    ax1.set_title('Episodes to goal per generation — average across all runs',
                  fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylabel('Episodes', fontsize=10)
    ax1.set_xlabel('Generation', fontsize=10)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    ax1.set_xticks(gens)
    ax1.legend(fontsize=9, loc='upper right', framealpha=0.7)

    # --- Chart 2: avg steps per generation across all runs ---
    gen_avg_steps = []
    gen_labels = []
    for g in range(1, max_gen + 1):
        vals = [pg[g-1]['steps'] for pg in all_pg if len(pg) >= g]
        if vals:
            gen_avg_steps.append(sum(vals) / len(vals))
            gen_labels.append(f'Gen {g}')

    bars2 = ax2.bar(gen_labels, gen_avg_steps, color=BLUE, edgecolor='white',
                    linewidth=1.2, alpha=0.88, width=0.5)
    for bar, val in zip(bars2, gen_avg_steps):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(gen_avg_steps) * 0.015,
                 f'{val:.0f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=DARK)
    ax2.axhline(np.mean(gen_avg_steps), color=RED, linestyle='--',
                linewidth=1.3, alpha=0.7,
                label=f'Overall avg: {np.mean(gen_avg_steps):.0f}')
    ax2.set_title('Avg steps per generation (across all runs)',
                  fontsize=11, fontweight='bold', pad=10)
    ax2.set_ylabel('Avg steps', fontsize=10)
    ax2.set_xlabel('Generation', fontsize=10)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='y', alpha=0.2, linestyle='--')
    ax2.legend(fontsize=9)
    ax2.tick_params(axis='x', labelsize=9)

    # --- Chart 3: avg energy remaining per generation across all runs ---
    gen_avg_energy = []
    for g in range(1, max_gen + 1):
        vals = [pg[g-1]['energy_remaining'] for pg in all_pg if len(pg) >= g]
        if vals:
            gen_avg_energy.append(sum(vals) / len(vals))

    bars3 = ax3.bar(gen_labels, gen_avg_energy, color=GREEN, edgecolor='white',
                    linewidth=1.2, alpha=0.88, width=0.5)
    for bar, val in zip(bars3, gen_avg_energy):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(gen_avg_energy) * 0.015,
                 f'{val:.0f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=DARK)
    ax3.axhline(np.mean(gen_avg_energy), color=RED, linestyle='--',
                linewidth=1.3, alpha=0.7,
                label=f'Overall avg: {np.mean(gen_avg_energy):.0f}')
    ax3.legend(fontsize=9)
    ax3.set_title('Avg energy remaining per generation (across all runs)',
                  fontsize=11, fontweight='bold', pad=10)
    ax3.set_ylabel('Avg energy remaining', fontsize=10)
    ax3.set_xlabel('Generation', fontsize=10)
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.grid(axis='y', alpha=0.2, linestyle='--')
    ax3.tick_params(axis='x', labelsize=9)

    out = '/Users/tommysupey/Desktop/Brain Simulation/version2_dashboard.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved to {out}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  VERSION 2 DASHBOARD")
    print("="*50)
    main()