"""
Version 4 Dashboard — Split Graphs
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

HISTORY_FILE = "/Users/tommysupey/Desktop/Brain Simulation/version4_history.json"
OUT_DIR      = "/Users/tommysupey/Desktop/Brain Simulation/charts"
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


def save_chart(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


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


def chart_organism_numbers(runs):
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

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(range(max_s), min_line, max_line,
                    color=BLUE, alpha=0.15, label='Min/max range')
    ax.plot(range(max_s), avg_line, color=BLUE, linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=BLUE,
            label=f'Historical avg ({len(runs)} runs)')
    for x, y in enumerate(avg_line):
        ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=9,
                    fontweight='bold', color=BLUE)
    ax.set_xticks(range(max_s))
    ax.set_xticklabels(success_labels)
    ax.set_title('Organism number at each success — average across all runs',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Organism #', fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.7)
    fig.tight_layout()
    save_chart(fig, 'v4_organism_numbers.png')


def chart_steps(success_labels, avg_steps):
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_chart(ax, success_labels, avg_steps, BLUE,
              'Avg steps per success (across all runs)', 'Avg steps')
    fig.tight_layout()
    save_chart(fig, 'v4_steps_per_success.png')


def chart_food(success_labels, avg_food):
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_chart(ax, success_labels, avg_food, GREEN,
              'Avg food found per success (across all runs)', 'Avg food found', fmt='.2f')
    fig.tight_layout()
    save_chart(fig, 'v4_food_per_success.png')


def chart_energy(success_labels, avg_energy):
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_chart(ax, success_labels, avg_energy, ORANGE,
              'Avg energy remaining per success (across all runs)', 'Avg energy remaining')
    fig.tight_layout()
    save_chart(fig, 'v4_energy_per_success.png')


def chart_attempts(success_labels, avg_attempts):
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_chart(ax, success_labels, avg_attempts, PURPLE,
              'Avg attempts per success (across all runs)', 'Avg attempts')
    fig.tight_layout()
    save_chart(fig, 'v4_attempts_per_success.png')


def main():
    runs = load_history()
    full_runs = [r for r in runs if r.get('per_success')]

    chart_organism_numbers(runs)

    if full_runs:
        max_s = max(len(r['per_success']) for r in full_runs)
        success_labels = [f"S{j+1}" for j in range(max_s)]

        def avg_field(field):
            result = []
            for s in range(max_s):
                vals = [r['per_success'][s][field] for r in full_runs if len(r['per_success']) > s]
                result.append(sum(vals) / len(vals) if vals else 0)
            return result

        chart_steps(success_labels, avg_field('steps'))
        chart_food(success_labels, avg_field('food_found'))
        chart_energy(success_labels, avg_field('energy_remaining'))
        chart_attempts(success_labels, avg_field('attempts'))

    print(f"\n  All charts saved to {OUT_DIR}")


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  VERSION 4 DASHBOARD — SPLIT GRAPHS")
    print("="*50)
    main()