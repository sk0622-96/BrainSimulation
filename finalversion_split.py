"""
Final Versions Dashboard — Split Graphs
=========================================
Saves each chart and the summary table as individual PNGs.
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import defaultdict

os.environ['TK_SILENCE_DEPRECATION'] = '1'

BASE_PATH    = "/Users/tommysupey/Desktop/Brain Simulation"
HISTORY_FILE = os.path.join(BASE_PATH, "simulation_history.json")
OUT_DIR      = os.path.join(BASE_PATH, "charts")

VERSION_ORDER  = ['Perfect LTM', 'Realistic LTM', 'Imperfect STM']
VERSION_COLORS = {
    'Perfect LTM':   '#2E86AB',
    'Realistic LTM': '#6A4C93',
    'Imperfect STM': '#E76F51',
}

DARK = '#222222'
GREY = '#AAAAAA'


def load_history():
    if not os.path.exists(HISTORY_FILE):
        print(f"\n  Could not find {HISTORY_FILE}")
        sys.exit(1)
    with open(HISTORY_FILE) as f:
        data = json.load(f)
    runs = data.get('runs', [])
    by_version = defaultdict(list)
    for i, run in enumerate(runs):
        v = run.get('version', '')
        if v in VERSION_ORDER:
            run['_run_index'] = i + 1
            by_version[v].append(run)
    if not any(v in by_version for v in VERSION_ORDER):
        print("  No tagged runs found.")
        sys.exit(1)
    print(f"  Loaded {len(runs)} total run(s) from {HISTORY_FILE}")
    return by_version


def save_chart(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


def avg_per_success(by_version, key):
    result = {}
    for v in VERSION_ORDER:
        runs = by_version.get(v, [])
        if not runs:
            continue
        all_series = []
        for run in runs:
            ps = run.get('per_success', [])
            if not ps:
                continue
            if key in ps[0]:
                all_series.append({s['success_num']: s[key] for s in ps})
        if not all_series:
            continue
        common = sorted(set.intersection(*[set(s.keys()) for s in all_series]))
        averaged = [sum(s[n] for s in all_series if n in s) / len(all_series) for n in common]
        result[v] = (common, averaged)
    return result


def legend_handles(by_version):
    return [
        plt.Line2D([0], [0], color=VERSION_COLORS[v], linewidth=2,
                   marker='o', markersize=6, markerfacecolor='white',
                   markeredgewidth=1.8, markeredgecolor=VERSION_COLORS[v],
                   label=f"{v}  (n={len(by_version.get(v, []))})")
        for v in VERSION_ORDER if v in by_version
    ]


def chart_organism_numbers(by_version):
    series = avg_per_success(by_version, 'organism_num')
    fig, ax = plt.subplots(figsize=(10, 5))
    for v, (xs, ys) in series.items():
        col = VERSION_COLORS[v]
        ax.plot(xs, ys, color=col, linewidth=1.8, marker='o',
                markersize=5, markerfacecolor='white',
                markeredgewidth=1.8, markeredgecolor=col,
                label=f"{v}  (n={len(by_version[v])})", zorder=3)
        ax.fill_between(xs, ys, alpha=0.07, color=col)
        for j, (x, y) in enumerate(zip(xs, ys)):
            offset = 9 if j % 2 == 0 else -14
            ax.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, offset),
                        textcoords='offset points', ha='center',
                        fontsize=7.5, color=col, fontweight='bold')
    ax.set_title('Organism # at each success — all versions', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Organism #', fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(handles=legend_handles(by_version), fontsize=9, framealpha=0.8)
    fig.tight_layout()
    save_chart(fig, 'final_organism_numbers.png')


def chart_steps(by_version):
    series = avg_per_success(by_version, 'steps')
    versions_present = [v for v in VERSION_ORDER if v in series]
    all_xs = sorted(set.intersection(*[set(series[v][0]) for v in versions_present]))
    x = np.arange(len(all_xs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(versions_present):
        xs, ys = series[v]
        vals = [ys[xs.index(s)] if s in xs else 0 for s in all_xs]
        offset = (i - (len(versions_present)-1)/2) * width
        ax.bar(x + offset, vals, width, color=VERSION_COLORS[v],
               alpha=0.88, edgecolor='white', linewidth=0.8, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in all_xs], fontsize=8)
    ax.set_title('Avg steps per success', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Steps', fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=9, framealpha=0.8)
    fig.tight_layout()
    save_chart(fig, 'final_steps_per_success.png')


def chart_energy(by_version):
    series = avg_per_success(by_version, 'energy_remaining')
    fig, ax = plt.subplots(figsize=(10, 5))
    for v, (xs, ys) in series.items():
        col = VERSION_COLORS[v]
        ax.plot(xs, ys, color=col, linewidth=1.8, marker='o',
                markersize=5, markerfacecolor='white',
                markeredgewidth=1.8, markeredgecolor=col,
                label=f"{v}  (n={len(by_version[v])})", zorder=3)
        ax.fill_between(xs, ys, alpha=0.07, color=col)
        for j, (x, y) in enumerate(zip(xs, ys)):
            offset = 9 if j % 2 == 0 else -14
            ax.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, offset),
                        textcoords='offset points', ha='center',
                        fontsize=7.5, color=col, fontweight='bold')
    ax.set_title('Avg energy remaining per success', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Energy', fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(handles=legend_handles(by_version), fontsize=9, framealpha=0.8)
    fig.tight_layout()
    save_chart(fig, 'final_energy_per_success.png')


def chart_food(by_version):
    series = avg_per_success(by_version, 'food_found')
    versions_present = [v for v in VERSION_ORDER if v in series]
    all_xs = sorted(set.intersection(*[set(series[v][0]) for v in versions_present]))
    x = np.arange(len(all_xs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, v in enumerate(versions_present):
        xs, ys = series[v]
        vals = [ys[xs.index(s)] if s in xs else 0 for s in all_xs]
        offset = (i - (len(versions_present)-1)/2) * width
        ax.bar(x + offset, vals, width, color=VERSION_COLORS[v],
               alpha=0.88, edgecolor='white', linewidth=0.8, label=v)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in all_xs], fontsize=8)
    ax.set_title('Avg food found per success', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Food found', fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=9, framealpha=0.8)
    fig.tight_layout()
    save_chart(fig, 'final_food_per_success.png')


def chart_summary_tables(by_version):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, version in zip(axes, VERSION_ORDER):
        col = VERSION_COLORS.get(version, DARK)
        ax.set_facecolor('#f7f8fc')
        for spine in ax.spines.values():
            spine.set_color(col)
            spine.set_linewidth(2)
        ax.axis('off')
        ax.text(0.5, 0.92, version, transform=ax.transAxes,
                ha='center', va='top', color=col, fontsize=9, fontweight='bold')
        runs = by_version.get(version, [])
        if not runs:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color=GREY, fontsize=9)
            continue
        avg_steps  = sum(r['average_steps'] for r in runs) / len(runs)
        best_steps = min(r['best_steps'] for r in runs)
        avg_energy = sum(r.get('average_energy', 0) for r in runs) / len(runs)
        avg_orgs   = sum(r['total_organisms'] for r in runs) / len(runs)
        avg_sr     = sum(r['success_rate'] for r in runs) / len(runs)
        avg_food   = sum(r.get('final_food_inherited', 0) for r in runs) / len(runs)
        rows = [
            ('Runs',         str(len(runs))),
            ('Avg Steps',    f'{avg_steps:.1f}'),
            ('Best Steps',   str(best_steps)),
            ('Avg Energy',   f'{avg_energy:.1f}'),
            ('Avg Food',     f'{avg_food:.1f}'),
            ('Avg Orgs',     f'{avg_orgs:.0f}'),
            ('Success Rate', f'{avg_sr:.1%}'),
        ]
        y = 0.78
        for label, val in rows:
            ax.text(0.08, y, label, transform=ax.transAxes,
                    ha='left', va='top', color=GREY, fontsize=8)
            ax.text(0.92, y, val, transform=ax.transAxes,
                    ha='right', va='top', color=DARK, fontsize=8, fontweight='bold')
            y -= 0.115
            ax.plot([0.06, 0.94], [y + 0.09, y + 0.09], color='#dde1ea',
                    linewidth=0.6, transform=ax.transAxes)
    fig.tight_layout()
    save_chart(fig, 'final_summary_tables.png')


def main():
    by_version = load_history()
    chart_organism_numbers(by_version)
    chart_steps(by_version)
    chart_energy(by_version)
    chart_food(by_version)
    chart_summary_tables(by_version)
    print(f"\n  All charts saved to {OUT_DIR}")


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  FINAL VERSIONS DASHBOARD — SPLIT GRAPHS")
    print("="*50)
    main()