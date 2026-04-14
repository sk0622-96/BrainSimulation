"""
Version 4 Dashboard
====================
Reads results from version4_history.json.
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

HISTORY_FILE = "/Users/tommysupey/Desktop/Brain Simulation/version4_history.json"
BLUE   = '#2E86AB'
GREEN  = '#4CAF82'
ORANGE = '#F4A261'
PURPLE = '#7C4DFF'
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


def bar_chart(ax, labels, values, colour, title, ylabel, avg_line=True, fmt='.0f'):
    bars = ax.bar(labels, values, color=colour, edgecolor='white',
                  linewidth=1.2, alpha=0.88, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values) * 0.015,
                f'{val:{fmt}}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=DARK)
    if avg_line:
        ax.axhline(np.mean(values), color=RED, linestyle='--', linewidth=1.3,
                   alpha=0.7, label=f'Overall avg: {np.mean(values):{fmt}}')
        ax.legend(fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.tick_params(axis='x', labelsize=8)


def main():
    runs = load_history()
    full_runs = [r for r in runs if r.get('per_success')]

    fig = plt.figure(figsize=(15, 13))
    fig.suptitle('Version 4 Dashboard — HuggingFace Language Brain',
                 fontsize=13, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.93, bottom=0.06,
                           left=0.08, right=0.97,
                           hspace=0.52, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # --- Chart 1: shaded band + historical average ---
    all_org = [r.get('organism_numbers', []) for r in runs if r.get('organism_numbers')]
    max_s = max(len(o) for o in all_org)
    success_labels = [f"S{j+1}" for j in range(max_s)]

    all_eps_per_s = []
    for s in range(max_s):
        vals = [o[s] for o in all_org if len(o) > s]
        all_eps_per_s.append(vals)

    avg_line = [np.mean(v) for v in all_eps_per_s]
    min_line = [np.min(v)  for v in all_eps_per_s]
    max_line = [np.max(v)  for v in all_eps_per_s]

    ax1.fill_between(range(max_s), min_line, max_line,
                     color=BLUE, alpha=0.15, label='Min/max range')
    ax1.plot(range(max_s), avg_line, color=BLUE, linewidth=2.5, marker='o',
             markersize=7, markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=BLUE,
             label=f'Historical avg ({len(runs)} runs)')
    for x, y in enumerate(avg_line):
        ax1.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=9,
                     fontweight='bold', color=BLUE)

    ax1.set_xticks(range(max_s))
    ax1.set_xticklabels(success_labels)
    ax1.set_title('Organism number at each success — average across all runs',
                  fontsize=11, fontweight='bold', pad=10)
    ax1.set_ylabel('Organism #', fontsize=10)
    ax1.set_xlabel('Success #', fontsize=10)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.7)

    if full_runs:
        max_s = max(len(r['per_success']) for r in full_runs)
        success_labels = [f"S{j+1}" for j in range(max_s)]

        def avg_field(field):
            result = []
            for s in range(max_s):
                vals = [r['per_success'][s][field] for r in full_runs if len(r['per_success']) > s]
                result.append(sum(vals) / len(vals) if vals else 0)
            return result

        avg_steps    = avg_field('steps')
        avg_food     = avg_field('food_found')
        avg_energy   = avg_field('energy_remaining')
        avg_attempts = avg_field('attempts')

        bar_chart(ax2, success_labels, avg_steps, BLUE,
                  'Avg steps per success (across all runs)', 'Avg steps')

        bar_chart(ax3, success_labels, avg_food, GREEN,
                  'Avg food found per success (across all runs)', 'Avg food found', fmt='.2f')

        bar_chart(ax4, success_labels, avg_energy, ORANGE,
                  'Avg energy remaining per success (across all runs)', 'Avg energy remaining')

        bar_chart(ax5, success_labels, avg_attempts, PURPLE,
                  'Avg attempts per success (across all runs)', 'Avg attempts')

    out = '/Users/tommysupey/Desktop/Brain Simulation/version4_dashboard.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Saved to {out}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  VERSION 4 DASHBOARD")
    print("="*50)
    main()