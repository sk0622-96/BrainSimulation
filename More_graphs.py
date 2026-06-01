"""
Cross-Version Comparison Charts
=================================
Compares comparable metrics across Version 2, 3, 4,
Perfect LTM, Realistic LTM, and Imperfect STM.

Saves individual PNGs to the charts folder.
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

os.environ['TK_SILENCE_DEPRECATION'] = '1'

BASE_PATH = "/Users/tommysupey/Desktop/Brain Simulation"
OUT_DIR   = os.path.join(BASE_PATH, "graphs")

# File paths
FILES = {
    'Version 2':      os.path.join(BASE_PATH, "version2_history.json"),
    'Version 3':      os.path.join(BASE_PATH, "version3_history.json"),
    'Version 4':      os.path.join(BASE_PATH, "version4_history.json"),
    'simulation':     os.path.join(BASE_PATH, "simulation_history.json"),
}

# Final three version labels inside simulation_history.json
# Maps JSON version tag -> display label
FINAL_THREE_MAP = {
    'Perfect LTM':   'V5: Perfect LTM',
    'Realistic LTM': 'V5: Realistic LTM',
    'Imperfect STM': 'V5: Imperfect STM',
}
FINAL_THREE = list(FINAL_THREE_MAP.keys())

# Colors per version
COLORS = {
    'Version 2':          '#4CAF82',
    'Version 3':          '#2E86AB',
    'Version 4':          '#F4A261',
    'V5: Perfect LTM':   '#7C4DFF',
    'V5: Realistic LTM': '#6A4C93',
    'V5: Imperfect STM': '#E76F51',
}

DARK = '#222222'
RED  = '#E53935'


# ── Data loading ───────────────────────────────────────────

def load_json(path):
    if not os.path.exists(path):
        print(f"  Warning: could not find {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_all():
    data = {}

    # Version 2 and 3 — per_generation structure
    for label in ('Version 2', 'Version 3'):
        key = label
        raw = load_json(FILES[key])
        if not raw:
            continue
        runs = raw.get('runs', [])
        all_pg = [r.get('per_generation', []) for r in runs if r.get('per_generation')]
        if all_pg:
            data[label] = {'type': 'per_generation', 'all_pg': all_pg, 'n_runs': len(all_pg)}

    # Version 4 — per_success structure
    raw = load_json(FILES['Version 4'])
    if raw:
        runs = [r for r in raw.get('runs', []) if r.get('per_success')]
        if runs:
            data['Version 4'] = {'type': 'per_success', 'runs': runs, 'n_runs': len(runs)}

    # Final three — per_success inside simulation_history.json
    raw = load_json(FILES['simulation'])
    if raw:
        by_version = defaultdict(list)
        for run in raw.get('runs', []):
            v = run.get('version', '')
            if v in FINAL_THREE and run.get('per_success'):
                by_version[v].append(run)
        for v in FINAL_THREE:
            if by_version[v]:
                display = FINAL_THREE_MAP[v]
                data[display] = {'type': 'per_success', 'runs': by_version[v], 'n_runs': len(by_version[v])}

    print(f"  Loaded versions: {list(data.keys())}")
    return data


# ── Metric extraction ──────────────────────────────────────

def get_avg_steps(info):
    """Returns a single average steps value for a version."""
    if info['type'] == 'per_generation':
        all_steps = []
        for pg in info['all_pg']:
            all_steps.extend([g['steps'] for g in pg])
        return np.mean(all_steps) if all_steps else None
    else:
        all_steps = []
        for run in info['runs']:
            all_steps.extend([s['steps'] for s in run['per_success']])
        return np.mean(all_steps) if all_steps else None


def get_avg_energy(info):
    """Returns a single average energy remaining value for a version."""
    if info['type'] == 'per_generation':
        vals = []
        for pg in info['all_pg']:
            vals.extend([g['energy_remaining'] for g in pg])
        return np.mean(vals) if vals else None
    else:
        vals = []
        for run in info['runs']:
            vals.extend([s['energy_remaining'] for s in run['per_success']])
        return np.mean(vals) if vals else None


def get_avg_food(info):
    """Returns a single average food found value. Only valid for V4 and final three."""
    if info['type'] == 'per_generation':
        return None  # V2/V3 food not tracked the same way
    vals = []
    for run in info['runs']:
        vals.extend([s['food_found'] for s in run['per_success']])
    return np.mean(vals) if vals else None


def get_efficiency(info, label):
    """
    Returns a single efficiency number representing how hard it was to succeed.
    V2/V3: avg episodes per generation to reach goal.
    V4+: avg organism number at first success (success 1).
    """
    if info['type'] == 'per_generation':
        vals = []
        for pg in info['all_pg']:
            if pg:
                vals.append(pg[0]['episodes'])
        return np.mean(vals) if vals else None
    else:
        # organism number at success 1
        vals = []
        for run in info['runs']:
            ps = run.get('per_success', [])
            if ps:
                # find success_num == 1 or just take the first entry
                s1 = next((s for s in ps if s.get('success_num', 0) == 1), ps[0])
                org = s1.get('organism_num', s1.get('organism_numbers', None))
                if org is not None:
                    vals.append(org)
        return np.mean(vals) if vals else None


def get_steps_over_time(info):
    """Returns (labels, avg, min, max) for steps plotted over generations/successes."""
    if info['type'] == 'per_generation':
        all_pg = info['all_pg']
        max_g = max(len(pg) for pg in all_pg)
        labels = [f'G{g+1}' for g in range(max_g)]
        avg, mn, mx = [], [], []
        for g in range(max_g):
            v = [pg[g]['steps'] for pg in all_pg if len(pg) > g]
            avg.append(np.mean(v) if v else 0)
            mn.append(np.min(v) if v else 0)
            mx.append(np.max(v) if v else 0)
        return labels, avg, mn, mx
    else:
        runs = info['runs']
        max_s = max(len(r['per_success']) for r in runs)
        labels = [f'S{s+1}' for s in range(max_s)]
        avg, mn, mx = [], [], []
        for s in range(max_s):
            v = [r['per_success'][s]['steps'] for r in runs if len(r['per_success']) > s]
            avg.append(np.mean(v) if v else 0)
            mn.append(np.min(v) if v else 0)
            mx.append(np.max(v) if v else 0)
        return labels, avg, mn, mx


def get_energy_over_time(info):
    """Returns (labels, avg, min, max) for energy plotted over generations/successes."""
    if info['type'] == 'per_generation':
        all_pg = info['all_pg']
        max_g = max(len(pg) for pg in all_pg)
        labels = [f'G{g+1}' for g in range(max_g)]
        avg, mn, mx = [], [], []
        for g in range(max_g):
            v = [pg[g]['energy_remaining'] for pg in all_pg if len(pg) > g]
            avg.append(np.mean(v) if v else 0)
            mn.append(np.min(v) if v else 0)
            mx.append(np.max(v) if v else 0)
        return labels, avg, mn, mx
    else:
        runs = info['runs']
        max_s = max(len(r['per_success']) for r in runs)
        labels = [f'S{s+1}' for s in range(max_s)]
        avg, mn, mx = [], [], []
        for s in range(max_s):
            v = [r['per_success'][s]['energy_remaining'] for r in runs if len(r['per_success']) > s]
            avg.append(np.mean(v) if v else 0)
            mn.append(np.min(v) if v else 0)
            mx.append(np.max(v) if v else 0)
        return labels, avg, mn, mx


def get_food_over_time(info):
    """Returns (labels, avg, min, max) for food found. Only V4 and final three."""
    if info['type'] == 'per_generation':
        return None, None, None, None
    runs = info['runs']
    max_s = max(len(r['per_success']) for r in runs)
    labels = [f'S{s+1}' for s in range(max_s)]
    avg, mn, mx = [], [], []
    for s in range(max_s):
        v = [r['per_success'][s]['food_found'] for r in runs if len(r['per_success']) > s]
        avg.append(np.mean(v) if v else 0)
        mn.append(np.min(v) if v else 0)
        mx.append(np.max(v) if v else 0)
    return labels, avg, mn, mx


# ── Saving ─────────────────────────────────────────────────

def save_chart(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


def style_ax(ax, title, ylabel, xlabel='Version'):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')


# ── Chart 1: Avg steps — all six versions (single bar each) ───

def chart_avg_steps(data):
    versions = [v for v in COLORS if v in data]
    values   = [get_avg_steps(data[v]) for v in versions]
    colors   = [COLORS[v] for v in versions]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(versions, values, color=colors, edgecolor='white',
                  linewidth=1.2, alpha=0.88, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values) * 0.015,
                f'{val:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=DARK)
    ax.axhline(np.mean(values), color=RED, linestyle='--', linewidth=1.3,
               alpha=0.7, label=f'Overall avg: {np.mean(values):.0f}')
    ax.legend(fontsize=9)
    style_ax(ax, 'Avg steps used to reach goal', 'Avg steps')
    ax.tick_params(axis='x', labelsize=8)
    ax.text(0.98, 0.97, 'Note: max steps per episode differs by version\nV2: 500  ·  V3: 150  ·  V4/V5: 200',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    fig.tight_layout()
    save_chart(fig, 'cross_avg_steps.png')


# ── Chart 2: Avg energy remaining — all six versions ──────

def chart_avg_energy(data):
    versions = [v for v in COLORS if v in data]
    values   = [get_avg_energy(data[v]) for v in versions]
    colors   = [COLORS[v] for v in versions]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(versions, values, color=colors, edgecolor='white',
                  linewidth=1.2, alpha=0.88, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values) * 0.015,
                f'{val:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=DARK)
    ax.axhline(np.mean(values), color=RED, linestyle='--', linewidth=1.3,
               alpha=0.7, label=f'Overall avg: {np.mean(values):.0f}')
    ax.legend(fontsize=9)
    style_ax(ax, 'Avg energy remaining at goal', 'Avg energy remaining')
    ax.tick_params(axis='x', labelsize=8)
    ax.text(0.98, 0.97, 'Note: starting energy differs by version\nV2: 300  ·  V3/V4/Final three: 100',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    fig.tight_layout()
    save_chart(fig, 'cross_avg_energy.png')


# ── Chart 3: Avg food found — V4 and final three ──────────

def chart_avg_food(data):
    versions = [v for v in ['Version 4'] + list(FINAL_THREE_MAP.values()) if v in data]
    values   = [get_avg_food(data[v]) for v in versions]
    colors   = [COLORS[v] for v in versions]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(versions, values, color=colors, edgecolor='white',
                  linewidth=1.2, alpha=0.88, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values) * 0.015,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=DARK)
    ax.axhline(np.mean(values), color=RED, linestyle='--', linewidth=1.3,
               alpha=0.7, label=f'Overall avg: {np.mean(values):.2f}')
    ax.legend(fontsize=9)
    style_ax(ax, 'Avg food found per success', 'Avg food found')
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    save_chart(fig, 'cross_avg_food.png')


# ── Chart 4: Efficiency — how hard was first success ──────

def chart_efficiency(data):
    versions = [v for v in COLORS if v in data]
    values   = [get_efficiency(data[v], v) for v in versions]
    colors   = [COLORS[v] for v in versions]

    # Filter out None
    valid = [(v, val, c) for v, val, c in zip(versions, values, colors) if val is not None]
    versions, values, colors = zip(*valid) if valid else ([], [], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(versions, values, color=colors, edgecolor='white',
                  linewidth=1.2, alpha=0.88, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values) * 0.015,
                f'{val:.0f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=DARK)
    style_ax(ax, 'Number of organisms until first success', 'Organisms')
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    save_chart(fig, 'cross_efficiency.png')


# ── Chart 5: Steps over time — line chart all versions ────

def chart_steps_over_time(data):
    fig, ax = plt.subplots(figsize=(12, 5))
    for v, info in data.items():
        labels, avg, mn, mx = get_steps_over_time(info)
        col = COLORS.get(v, '#999999')
        x = range(len(labels))
        ax.fill_between(x, mn, mx, color=col, alpha=0.1)
        ax.plot(x, avg, color=col, linewidth=1.8, marker='o',
                markersize=4, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=col,
                label=f"{v}")
    ax.set_title('Avg steps over time', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Avg steps', fontsize=10)
    ax.set_xlabel('Generation / Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.7, bbox_to_anchor=(1, 0.88))
    ax.text(0.99, 0.02, 'Note: max steps per episode differs by version\nV2: 500  ·  V3: 150  ·  V4/V5: 200',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    fig.tight_layout()
    save_chart(fig, 'cross_steps_over_time.png')


# ── Chart 6: Energy over time — line chart all versions ───

def chart_energy_over_time(data):
    fig, ax = plt.subplots(figsize=(12, 5))
    for v, info in data.items():
        labels, avg, mn, mx = get_energy_over_time(info)
        col = COLORS.get(v, '#999999')
        x = range(len(labels))
        if v != 'Version 2':
            ax.fill_between(x, mn, mx, color=col, alpha=0.1)
        ax.plot(x, avg, color=col, linewidth=1.8, marker='o',
                markersize=4, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=col,
                label=f"{v}")
    ax.set_title('Avg energy remaining over time', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Avg energy remaining', fontsize=10)
    ax.set_xlabel('Generation / Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.7, bbox_to_anchor=(1, 0.88))
    ax.text(0.99, 0.02, 'Note: starting energy differs by version\nV2: 300  ·  V3/V4/V5: 100',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    fig.tight_layout()
    save_chart(fig, 'cross_energy_over_time.png')


# ── Chart 7: Food over time — V4 and final three ──────────

def chart_food_over_time(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    versions = [v for v in ['Version 4'] + list(FINAL_THREE_MAP.values()) if v in data]
    for v in versions:
        labels, avg, mn, mx = get_food_over_time(data[v])
        if labels is None:
            continue
        col = COLORS.get(v, '#999999')
        x = range(len(labels))
        ax.fill_between(x, mn, mx, color=col, alpha=0.1)
        ax.plot(x, avg, color=col, linewidth=1.8, marker='o',
                markersize=4, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=col,
                label=f"{v}")
    ax.set_title('Avg food found per success', fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Avg food found', fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.7)
    fig.tight_layout()
    save_chart(fig, 'cross_food_over_time.png')


# ── Chart 8: Cumulative organisms to each success — V3, V4, V5 ───

def chart_cumulative_organisms(data):
    """
    V3: cumulative sum of episodes across generations (converted to match V4/V5 organism counting).
    V4/V5: organism_num at each success, averaged across runs, capped at 10.
    """
    versions = ['Version 2', 'Version 3'] + ['Version 4'] + list(FINAL_THREE_MAP.values())
    fig, ax = plt.subplots(figsize=(12, 5))

    for v in versions:
        if v not in data:
            continue
        info = data[v]
        col  = COLORS.get(v, '#999999')

        if info['type'] == 'per_generation':
            all_pg = info['all_pg']
            max_g  = min(10, max(len(pg) for pg in all_pg))
            cumvals, cummin, cummax = [], [], []
            for g in range(max_g):
                ep_vals = [pg[g]['episodes'] for pg in all_pg if len(pg) > g]
                cumvals.append(np.mean(ep_vals) if ep_vals else 0)
                cummin.append(np.min(ep_vals) if ep_vals else 0)
                cummax.append(np.max(ep_vals) if ep_vals else 0)
            cumulative     = np.cumsum(cumvals)
            cumulative_min = np.cumsum(cummin)
            cumulative_max = np.cumsum(cummax)
            xs = list(range(1, len(cumulative) + 1))
            ax.fill_between(xs, cumulative_min, cumulative_max, color=col, alpha=0.1)
            ax.plot(xs, cumulative, color=col, linewidth=1.8, marker='o',
                    markersize=5, markerfacecolor='white',
                    markeredgewidth=1.5, markeredgecolor=col,
                    label=f"{v}")
            for x, y in zip(xs, cumulative):
                ax.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, 9),
                            textcoords='offset points', ha='center',
                            fontsize=7.5, color=col, fontweight='bold')

        else:
            runs   = info['runs']
            max_s  = min(10, max(len(r['per_success']) for r in runs))
            vals, mn, mx = [], [], []
            for s in range(max_s):
                v_vals = [r['per_success'][s]['organism_num']
                          for r in runs if len(r['per_success']) > s]
                vals.append(np.mean(v_vals) if v_vals else 0)
                mn.append(np.min(v_vals) if v_vals else 0)
                mx.append(np.max(v_vals) if v_vals else 0)
            xs = list(range(1, len(vals) + 1))
            ax.fill_between(xs, mn, mx, color=col, alpha=0.1)
            ax.plot(xs, vals, color=col, linewidth=1.8, marker='o',
                    markersize=5, markerfacecolor='white',
                    markeredgewidth=1.5, markeredgecolor=col,
                    label=f"{v}")
            for x, y in zip(xs, vals):
                ax.annotate(f'{y:.0f}', xy=(x, y), xytext=(0, 9),
                            textcoords='offset points', ha='center',
                            fontsize=7.5, color=col, fontweight='bold')

    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f'S{i}' for i in range(1, 11)], fontsize=9)
    ax.set_title('Cumulative organisms to each success',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('Cumulative organisms', fontsize=10)
    ax.set_xlabel('Success #', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.7)
    ax.text(0.99, 0.02, 'V2 only ran 3 successes',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    fig.tight_layout()
    save_chart(fig, 'cross_cumulative_organisms.png')




def main():
    data = load_all()
    if not data:
        print("  No data found. Check your JSON file paths.")
        sys.exit(1)

    chart_avg_steps(data)
    chart_avg_energy(data)
    chart_avg_food(data)
    chart_efficiency(data)
    chart_steps_over_time(data)
    chart_energy_over_time(data)
    chart_food_over_time(data)
    chart_cumulative_organisms(data)

    print(f"\n  All cross-version charts saved to {OUT_DIR}")


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  CROSS-VERSION COMPARISON CHARTS")
    print("="*50)
    main()