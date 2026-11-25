"""Plot the independently verified saved-data findings, without retrieval.

New verification figures, 2026-11-26. Existing figures are never overwritten;
use --output-dir with a new directory to check figure regeneration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'sigir_analysis/.plot_deps'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fiqa_oracle_robustness import recompute_oracles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE/'figures/verified')
    args = parser.parse_args()
    dest = args.output_dir
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {dest}')
    dest.mkdir(parents=True, exist_ok=True)
    source = HERE/'VERIFIED_SAVED_DATA_RESULTS.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    if data['status'] != 'PASS':
        raise ValueError('Figures require a verified summary')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'CURA-verified-20261126'})
    conditions = ['50_PERCENT_POSITIVES', '25_PERCENT_POSITIVES', 'ONE_POSITIVE_PER_QUERY']
    labels = ['50%', '25%', 'One per query']
    primary = next(x for x in data['trec_false_safety']['tolerance_sensitivity'] if x['tolerance'] == .025)
    records = []

    def save(fig, name, values):
        fig.tight_layout(rect=(0, .065, 1, 1) if name == 'trec_false_safety_conditional' else (0, 0, 1, 1))
        fig.savefig(dest/(name+'.png'), dpi=180)
        fig.savefig(dest/(name+'.svg'), metadata={'Date': None})
        plt.close(fig)
        records.append({'name': name, 'plotted_values': values,
                        'files': {ext: hashlib.sha256((dest/(name+'.'+ext)).read_bytes()).hexdigest()
                                  for ext in ('png', 'svg')}})

    values = [primary['conditions'][c]['conditional_false_safe_rate']*100 for c in conditions]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(labels, values, color=['#277DA1', '#F8961E', '#B44A54'], width=.58)
    for i, (condition, value) in enumerate(zip(conditions, values)):
        row = primary['conditions'][condition]
        ax.text(i, value+2, f'{value:.2f}%\n{row["false_safe_count"]}/700', ha='center', fontsize=10)
    ax.set(ylim=(0, 104), xlabel='Positive judgments retained',
           ylabel='FULL-unsafe pair instances labeled safe (%)',
           title='TREC DL 2019: judgment thinning masks unsafe labels')
    fig.text(.5, .018, '14 FULL-unsafe pairs x 50 seeds; tolerance 0.025. FULL = available pooled judgments.',
             ha='center', fontsize=9)
    fig.subplots_adjust(bottom=.16)
    save(fig, 'trec_false_safety_conditional', values)

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    tolerance_values = {}
    for item, color in zip(data['trec_false_safety']['tolerance_sensitivity'], ['#277DA1', '#F8961E', '#B44A54']):
        values = [item['conditions'][c]['conditional_false_safe_rate']*100 for c in conditions]
        tolerance_values[str(item['tolerance'])] = values
        ax.plot(labels, values, 'o-', label=f'Tolerance {item["tolerance"]}: {item["full_unsafe_pairs"]} FULL-unsafe pairs', color=color)
    ax.set(ylim=(0, 100), xlabel='Positive judgments retained',
           ylabel='FULL-unsafe pair instances labeled safe (%)',
           title='TREC DL 2019: nDCG-loss tolerance sensitivity')
    ax.grid(axis='y', alpha=.2)
    ax.legend(fontsize=9)
    save(fig, 'trec_tolerance_sensitivity', tolerance_values)

    source_rows = pd.read_csv(ROOT/'beir_fiqa/fiqa_query_budget_results.csv', dtype={'query_id': str})
    saved = pd.read_csv(ROOT/'beir_fiqa/fiqa_oracle_budgets.csv', dtype={'query_id': str})
    oracle = recompute_oracles(source_rows, saved)
    cohort = oracle[(oracle.exact_nDCG10 > 0) & oracle.ann_safe_ef.notna() & oracle.ir_safe_ef.notna()]
    result = data['fiqa_oracle_replication']['results']
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.6))
    for field, label, color in [('ann_safe_ef', 'ANN-safe', '#277DA1'), ('ir_safe_ef', 'IR-safe', '#F8961E')]:
        x = np.sort(cohort[field].to_numpy())
        axes[0].step(x, np.arange(1, len(x)+1)/len(x), where='post', label=label, color=color, lw=2)
    axes[0].set(xlabel='First safe tested efSearch', ylabel='Cumulative query fraction',
                title='303 jointly feasible exact-positive queries', ylim=(0, 1.03))
    axes[0].legend()
    axes[0].grid(alpha=.2)
    counts = [result['IR_safe_lower'], result['equal_budgets'], result['IR_safe_higher']]
    axes[1].bar(['IR lower', 'Equal', 'IR higher'], counts, color=['#277DA1', '#8F9CA6', '#B44A54'])
    for i, count in enumerate(counts):
        axes[1].text(i, count+3, str(count), ha='center')
    axes[1].set(ylim=(0, 195), ylabel='Queries', title=f'Mean ANN-minus-IR gap: {result["mean_ANN_minus_IR_gap"]:.3f} ef')
    fig.suptitle('FiQA oracle replication: distinct from judgment-density false safety', fontsize=12)
    save(fig, 'fiqa_oracle_replication', {'counts': counts, 'mean_gap': result['mean_ANN_minus_IR_gap'],
                                       'ANN_mean': result['mean_ANN_safe_ef'], 'IR_mean': result['mean_IR_safe_ef']})
    (dest/'manifest.json').write_text(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                                'figures': records}, indent=2)+'\n', encoding='utf-8')
    print('PASS: generated three figures from verified data')


if __name__ == '__main__':
    main()
