"""Regenerate paper artifacts from the completed, fixed-protocol six-seed study."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'figures/fresh_confirmation'
MODES = [('initial', 'Independent'), ('conservation_only', 'Conserve'),
         ('self_conserve', 'Self+C'), ('foreign_conserve', 'Foreign+C')]
SUPPORTS = [('native', 'Native'), ('lora', 'LoRA')]


def main():
    records = pd.read_csv(HERE / 'analysis/per_task.csv')
    results = json.loads((HERE / 'analysis/results.json').read_text())
    retention = pd.read_csv(HERE / 'analysis/retention.csv')
    verified = json.loads((HERE / 'analysis/verification.json').read_text())
    assert verified['checks_passed'] and verified['primary_statistics_independently_recomputed']
    seeds = results['protocol']['seeds']
    assert len(seeds) == 6 and set(records.seed) == set(seeds)
    assert len(records) == len(records.drop_duplicates(['support', 'seed', 'mode', 'split', 'task', 'mask']))
    scores = records.query("split == 'test' and mask == 'all'").groupby(['support', 'seed', 'mode']).accuracy.mean().mul(100)
    OUT.mkdir(parents=True, exist_ok=True)
    primary = []
    for support, _ in SUPPORTS:
        paired = (scores.loc[support].unstack().foreign_conserve - scores.loc[support].unstack().self_conserve).reindex(seeds)
        stored = next(x for x in results['primary'] if x['support'] == support)
        interval = stats.t.interval(.95, 5, loc=paired.mean(), scale=stats.sem(paired))
        assert np.allclose(paired, stored['paired_pp']) and np.allclose(interval, stored['ci95_pp'])
        assert abs(stats.ttest_1samp(paired, 0).pvalue - stored['p_two_sided']) < 1e-10
        primary.append(stored)
    ordered = sorted(primary, key=lambda r: r['p_two_sided'])
    adjusted = 0.
    for i, row in enumerate(ordered):
        adjusted = max(adjusted, min(1., (2-i)*row['p_two_sided']))
        assert abs(adjusted-row['p_holm']) < 1e-10 and not row['meets_prespecified_success']
    def values(support, mode, mask='all', split='test'):
        return records[(records.support == support) & (records['mode'] == mode) &
                       (records['mask'] == mask) & (records.split == split)].groupby('seed').accuracy.mean().reindex(seeds).mul(100)
    def cell(values, places=1):
        return f'${values.mean():.{places}f} \\pm {values.std(ddof=1):.{places}f}$'
    lines = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Support & Indep. & C & Self+C & Foreign+C\\', r'\midrule']
    for support, label in SUPPORTS:
        lines.append(label + ' & ' + ' & '.join(f'{values(support, mode).mean():.1f}' for mode, _ in MODES) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'accuracy_table.tex').write_text('\n'.join(lines) + '\n')
    lines = [r'\begin{tabular}{@{}llrrrrr@{}}', r'\toprule',
             r'Support & Condition & Owner & All eight & Without owner & $\Delta$PPL (\%) & MC $\Delta$ (pp)\\', r'\midrule']
    for support, label in SUPPORTS:
        for mode, title in MODES:
            rows = retention[(retention.support == support) & (retention['mode'] == mode)]
            wiki = rows[rows.panel == 'wikitext'].set_index('seed').perplexity_change_percent.reindex(seeds)
            mc = rows[rows.panel != 'wikitext'].copy()
            weighted = mc.assign(weighted=mc.change_pp*mc.n).groupby('seed').apply(
                lambda g: g.weighted.sum()/g.n.sum(), include_groups=False).reindex(seeds)
            cells = [cell(values(support, mode, mask)) for mask in ['owner', 'all', 'without_owner']]
            lines.append(label+' & '+title+' & '+' & '.join(cells+[cell(wiki), cell(weighted)])+r'\\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    (OUT / 'full_summary_table.tex').write_text('\n'.join(lines) + '\n')
    lines = [r'\begin{tabular}{@{}lrrrrrrrl@{}}', r'\toprule',
             'Support & '+' & '.join(map(str, seeds))+r' & Mean & 95\% paired interval\\', r'\midrule']
    for row in primary:
        label = dict(SUPPORTS)[row['support']]
        numbers = [f'${x:+.2f}$' for x in row['paired_pp']]
        lo, hi = row['ci95_pp']
        lines.append(label+' & '+' & '.join(numbers)+f" & ${row['mean_pp']:+.2f}$ & $[{lo:+.2f},{hi:+.2f}]$"+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'paired_table.tex').write_text('\n'.join(lines) + '\n')
    names = ['Arithmetic', 'Pointer', 'Boolean', 'Brackets', 'Sorting', 'Stack', 'Shortest path', 'Sudoku']
    per = records.query("split == 'test' and mask == 'all'").groupby(['support', 'mode', 'task']).accuracy.mean().mul(100)
    base = records.query("split == 'test' and mask == 'base'").groupby('task').accuracy.mean().mul(100)
    lines = [r'\begin{tabular}{@{}lrrrrrrrrr@{}}', r'\toprule',
             r' & & \multicolumn{4}{c}{Native} & \multicolumn{4}{c}{LoRA}\\',
             r'Task & Base & Indep. & C & S+C & F+C & Indep. & C & S+C & F+C\\', r'\midrule']
    for task, name in enumerate(names):
        cells = [f'{base[task]:.1f}'] + [f'{per[s, m, task]:.1f}' for s, _ in SUPPORTS for m, _ in MODES]
        lines.append(name+' & '+' & '.join(cells)+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'task_table.tex').write_text('\n'.join(lines) + '\n')
    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    colours = ['#2563eb', '#d97706']
    for y, (row, colour) in enumerate(zip(primary, colours)):
        ax.scatter(row['paired_pp'], y+np.linspace(-.10, .10, 6), s=20, color=colour, zorder=3)
        lo, hi = row['ci95_pp']
        ax.errorbar(row['mean_pp'], y, xerr=[[row['mean_pp']-lo], [hi-row['mean_pp']]],
                    fmt='D', color='black', ms=4, lw=1.3, capsize=4, zorder=4)
    ax.axvline(0, color='#64748b', ls='--', lw=.9)
    ax.set(yticks=[0, 1], yticklabels=['Native 4', 'LoRA r1'], xlabel='(Foreign + C) − (Self + C) (pp)',
           xlim=(-3.2, 10.5), ylim=(1.45, -.55))
    ax.grid(axis='x', color='#e2e8f0', lw=.5)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#64748b', label='Training/allocation seed'),
                       Line2D([], [], marker='D', color='black', label='Mean and paired 95% t interval')],
              loc='upper left', bbox_to_anchor=(0., 1.45), frameon=False, fontsize=7.5)
    fig.tight_layout(pad=.3)
    fig.savefig(OUT / 'primary_contrasts.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(OUT / 'primary_contrasts.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    paths = [HERE / 'analysis' / n for n in ['per_task.csv', 'results.json', 'retention.csv', 'verification.json']]
    paths.append(Path(__file__).resolve())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = dict(inputs={str(p.relative_to(HERE.parents[1])): sha(p) for p in paths},
                      outputs={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'provenance.json'},
                      seed_count=6, main_contrast='foreign_conserve minus self_conserve',
                      neither_primary_met=True, model_runs=0)
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print('Wrote fresh-bank tables and paired-contrast figure; all declared statistics verified')


if __name__ == '__main__':
    main()
