"""
Version 2 Dashboard — Split Graphs
=====================================
Saves each chart as an individual PNG.
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.environ['TK_SILENCE_DEPRECATION'] = '1'

HISTORY_FILE = "/Users/tommysupey/Desktop/Brain Simulation/version2_history.json"
OUT_DIR      = "/Users/tommysupey/Desktop/Brain Simulation/charts"
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


def save_chart(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


def chart_episodes(runs, all_pg, max_gen, gens):
    all_eps_per_gen = []
    for g in range(1, max_gen + 1):
        vals = [pg[g-1]['episodes'] for pg in all_pg if len(pg) >= g]
        all_eps_per_gen.append(vals)

    avg_line = [np.mean(v) for v in all_eps_per_gen]
    min_line = [np.min(v)  for v in all_eps_per_gen]
    max_line = [np.max(v)  for v in all_eps_per_gen]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(gens, min_line, max_line,
                    color=BLUE, alpha=0.15, label='Min/max range')
    ax.plot(gens, avg_line, color=BLUE, linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=BLUE,
            label=f'Historical avg ({len(runs)} runs)')
    for x, y in zip(gens, avg_line):
        ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=9,
                    fontweight='bold', color=BLUE)
    ax.set_title('Episodes to goal per generation — average across all runs',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Episodes', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.set_xticks(gens)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.7)
    fig.tight_layout()
    save_chart(fig, 'v2_episodes_per_generation.png')


def chart_steps(all_pg, max_gen):
    gen_avg_steps = []
    gen_labels = []
    for g in range(1, max_gen + 1):
        vals = [pg[g-1]['steps'] for pg in all_pg if len(pg) >= g]
        if vals:
            gen_avg_steps.append(sum(vals) / len(vals))
            gen_labels.append(f'Gen {g}')

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(gen_labels, gen_avg_steps, color=BLUE, edgecolor='white',
                  linewidth=1.2, alpha=0.88, width=0.5)
    for bar, val in zip(bars, gen_avg_steps):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(gen_avg_steps) * 0.015,
                f'{val:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=DARK)
    ax.axhline(np.mean(gen_avg_steps), color=RED, linestyle='--',
               linewidth=1.3, alpha=0.7,
               label=f'Overall avg: {np.mean(gen_avg_steps):.0f}')
    ax.set_title('Avg steps per generation (across all runs)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Avg steps', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    save_chart(fig, 'v2_steps_per_generation.png')


def chart_energy(all_pg, max_gen):
    gen_avg_energy = []
    gen_labels = []
    for g in range(1, max_gen + 1):
        vals = [pg[g-1]['energy_remaining'] for pg in all_pg if len(pg) >= g]
        if vals:
            gen_avg_energy.append(sum(vals) / len(vals))
            gen_labels.append(f'Gen {g}')

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(gen_labels, gen_avg_energy, color=GREEN, edgecolor='white',
                  linewidth=1.2, alpha=0.88, width=0.5)
    for bar, val in zip(bars, gen_avg_energy):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(gen_avg_energy) * 0.015,
                f'{val:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=DARK)
    ax.axhline(np.mean(gen_avg_energy), color=RED, linestyle='--',
               linewidth=1.3, alpha=0.7,
               label=f'Overall avg: {np.mean(gen_avg_energy):.0f}')
    ax.set_title('Avg energy remaining per generation (across all runs)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Avg energy remaining', fontsize=10)
    ax.set_xlabel('Generation', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', labelsize=9)
    fig.tight_layout()
    save_chart(fig, 'v2_energy_per_generation.png')


def main():
    runs = load_history()
    all_pg  = [r.get('per_generation', []) for r in runs if r.get('per_generation')]
    max_gen = max((len(pg) for pg in all_pg), default=1)
    gens    = list(range(1, max_gen + 1))

    chart_episodes(runs, all_pg, max_gen, gens)
    chart_steps(all_pg, max_gen)
    chart_energy(all_pg, max_gen)

    print(f"\n  All charts saved to {OUT_DIR}")


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  VERSION 2 DASHBOARD — SPLIT GRAPHS")
    print("="*50)
    main()