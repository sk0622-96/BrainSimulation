"""
Final Versions Dashboard — Perfect LTM, Realistic LTM, Imperfect STM
Reads results from simulation_history.json .
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
from datetime import datetime

os.environ['TK_SILENCE_DEPRECATION'] = '1'

BASE_PATH    = "/Users/tommysupey/Desktop/Brain Simulation"
HISTORY_FILE = os.path.join(BASE_PATH, "simulation_history.json")
OUTPUT_FILE  = os.path.join(BASE_PATH, "dashboard.png")

VERSION_ORDER  = ['Perfect LTM', 'Realistic LTM', 'Imperfect STM']
VERSION_COLORS = {
    'Perfect LTM':   '#2E86AB',
    'Realistic LTM': '#6A4C93',
    'Imperfect STM': '#E76F51',
}

DARK = '#222222'
GREY = '#AAAAAA'
RED  = '#E53935'


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


def avg_per_success(by_version, key):
    """For each version, average the per_success field across all runs."""
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


def draw_line_chart(ax, by_version, key, title, ylabel):
    series = avg_per_success(by_version, key)
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
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')


def draw_stat_table(ax, by_version, version):
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
        return

    avg_steps  = sum(r['average_steps'] for r in runs) / len(runs)
    best_steps = min(r['best_steps'] for r in runs)
    avg_energy = sum(r.get('average_energy', 0) for r in runs) / len(runs)
    avg_orgs   = sum(r['total_organisms'] for r in runs) / len(runs)
    avg_sr     = sum(r['success_rate'] for r in runs) / len(runs)
    avg_food   = sum(r.get('final_food_inherited', 0) for r in runs) / len(runs)

    rows = [
        ('Runs',          str(len(runs))),
        ('Avg Steps',     f'{avg_steps:.1f}'),
        ('Best Steps',    str(best_steps)),
        ('Avg Energy',    f'{avg_energy:.1f}'),
        ('Avg Food',      f'{avg_food:.1f}'),
        ('Avg Orgs',      f'{avg_orgs:.0f}'),
        ('Success Rate',  f'{avg_sr:.1%}'),
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


def draw_grouped_bar(ax, by_version, key, title, ylabel):
    series = avg_per_success(by_version, key)
    versions_present = [v for v in VERSION_ORDER if v in series]
    if not versions_present:
        return

    # Get common success numbers
    all_xs = sorted(set.intersection(*[set(series[v][0]) for v in versions_present]))
    x = np.arange(len(all_xs))
    n = len(versions_present)
    width = 0.25

    for i, v in enumerate(versions_present):
        xs, ys = series[v]
        vals = [ys[xs.index(s)] if s in xs else 0 for s in all_xs]
        offset = (i - (n-1)/2) * width
        bars = ax.bar(x + offset, vals, width, color=VERSION_COLORS[v],
                      alpha=0.88, edgecolor='white', linewidth=0.8, label=v)

    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in all_xs], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')


def main():
    by_version = load_history()

    fig = plt.figure(figsize=(15, 14))
    fig.suptitle('Final Versions Dashboard — Perfect LTM  ·  Realistic LTM  ·  Imperfect STM',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.text(0.5, 0.955, datetime.now().strftime('%Y-%m-%d %H:%M'),
             ha='center', fontsize=9, color=GREY)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           top=0.88, bottom=0.05,
                           left=0.08, right=0.97,
                           hspace=0.58, wspace=0.32)

    ax1  = fig.add_subplot(gs[0, :])
    ax2  = fig.add_subplot(gs[1, 0])
    ax3  = fig.add_subplot(gs[1, 1])
    ax4  = fig.add_subplot(gs[2, :])

    gs_tables = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3, :], wspace=0.3)
    ax_t0 = fig.add_subplot(gs_tables[0])
    ax_t1 = fig.add_subplot(gs_tables[1])
    ax_t2 = fig.add_subplot(gs_tables[2])

    draw_line_chart(ax1, by_version, 'organism_num',     'Organism # at each success — all versions', 'Organism #')
    draw_grouped_bar(ax2, by_version, 'steps',           'Avg steps per success',                     'Steps')
    draw_line_chart(ax3, by_version, 'energy_remaining', 'Avg energy remaining per success',           'Energy')
    draw_grouped_bar(ax4, by_version, 'food_found',      'Avg food found per success',                 'Food found')

    for ax_t, v in zip([ax_t0, ax_t1, ax_t2], VERSION_ORDER):
        draw_stat_table(ax_t, by_version, v)

    # Single shared legend
    handles = [
        plt.Line2D([0], [0], color=VERSION_COLORS[v], linewidth=2,
                   marker='o', markersize=6, markerfacecolor='white',
                   markeredgewidth=1.8, markeredgecolor=VERSION_COLORS[v],
                   label=f"{v}  (n={len(by_version.get(v, []))})")
        for v in VERSION_ORDER if v in by_version
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.97),
               fontsize=9, framealpha=0.8, ncol=3)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\n  Saved to {OUTPUT_FILE}")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  FINAL VERSIONS DASHBOARD")
    print("="*50)
    main()