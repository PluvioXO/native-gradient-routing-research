"""Plot retained experimental summaries; no training or model inference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT/'aaai/figures/report_visuals/data/01_budget_accuracy.csv',
         ROOT/'aaai/figures/report_visuals/data/04_independent_coexistence.csv',
         ROOT/'refusal/experiments/refusal_colocalization_v3/analysis/endpoints.csv',
         ROOT/'refusal/experiments/refusal_colocalization_v4/analysis/endpoints.csv']
budget, coexist, simple, hard = [pd.read_csv(p) for p in paths]
assert len(budget) == 19 and len(coexist) == 4
assert round(coexist.set_index('allocation').loc['Native 4', 'eight'], 1) == 42.1
plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.6))
for name, color, marker in [('Native', '#2563eb', 'o'), ('LoRA', '#d97706', 's')]:
    rows = budget[(budget.method == name) & (budget.params_m < 1)]
    axes[0].errorbar(rows.params_m, rows.accuracy, yerr=rows.sd, color=color,
                     marker=marker, ms=3, lw=1.1, capsize=2, label=name)
axes[0].axhline(38.3, color='grey', ls='--', lw=.8, label='Base')
axes[0].set(xlabel='Trainable scalars (millions)', ylabel='Pointer accuracy (%)',
            title='(a) Low-budget RL', ylim=(0, 105))
axes[0].legend(fontsize=6, loc='lower right')
for allocation, label, color in [('Native 4', 'Native 4', '#2563eb'), ('LoRA r1', 'LoRA r1', '#d97706')]:
    row = coexist.set_index('allocation').loc[allocation]
    axes[1].errorbar([1, 2, 4, 8], row[['one', 'two', 'four', 'eight']].to_numpy(dtype=float),
                     yerr=row[['one_sd', 'two_sd', 'four_sd', 'eight_sd']].to_numpy(dtype=float),
                     label=label, color=color, marker='o', ms=3, capsize=2, lw=1.1)
axes[1].axhline(29.1, color='grey', ls='--', lw=.8, label='Base')
axes[1].set(xlabel='Active specialists', ylabel='Macro task accuracy (%)',
            title='(b) Independent skills', ylim=(20, 75), xticks=[1, 2, 4, 8])
axes[1].legend(fontsize=6, loc='lower left')
for i, (frame, prefix) in enumerate([(simple, ''), (hard, 'task_')]):
    rows = frame[(frame.kind == 'policy') & (frame.condition == 'shared8') & (frame.endpoint == 'best')]
    assert len(rows) == 1
    row = rows.iloc[0]
    for j, (key, label, color) in enumerate([('allowed_correct', 'Allowed correct', '#2563eb'),
                                            ('blocked_refusal', 'Blocked refused', '#0f766e'),
                                            ('false_refusal', 'False refusal', '#d97706')]):
        axes[2].bar(i+(j-1)*.24, 100*row[prefix+key], .24, color=color, label=label if i == 0 else None)
axes[2].set(xticks=[0, 1], xticklabels=['1–2 hop', 'Branching'], ylabel='Rate (%)',
            title='(c) Shared eight: SFT', ylim=(0, 128), yticks=[0, 25, 50, 75, 100])
axes[2].legend(fontsize=6, loc='upper right')
fig.tight_layout(w_pad=1.0)
out = ROOT/'proposal/figures'
out.mkdir(exist_ok=True)
fig.savefig(out/'overview.pdf', bbox_inches='tight')
plt.close(fig)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(out/'overview_provenance.json').write_text(json.dumps({
    'generator_sha256': sha(Path(__file__)),
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in paths},
    'output_sha256': sha(out/'overview.pdf'),
    'no_model_calls': True, 'error_bars': 'sample standard deviation across three seeds in panels a/b',
    'panel_c': 'one seed; different task/curriculum families, not a controlled hardness effect'
}, indent=2)+'\n')
print('Generated proposal overview from four retained summaries.')

# One temporal comparison answers the collapse/recovery question without a graph atlas.
curve_path = ROOT/'refusal/experiments/refusal_removal_v7/analysis/learning_curves.csv'
curves = pd.read_csv(curve_path)
fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.9), sharey=True)
for ax, lr in zip(axes, [1e-6, 5e-5]):
    rows = curves[(curves.parent == 'shared8') & (curves.objective == 'preserve') &
                  (curves.lr == lr)].sort_values('step')
    assert len(rows) == 14 and rows.step.iloc[0] == 0 and rows.step.iloc[-1] == 256
    assert rows.step.is_unique
    assert np.isclose(100*rows.allowed_accuracy.min(), 97.265625 if lr == 1e-6 else 0)
    for field, label, color, style in [
        ('allowed_accuracy', 'Allowed correct', '#2563eb', '-'),
        ('blocked_accuracy', 'Blocked correct', '#d97706', '-'),
        ('blocked_refusal', 'Blocked refusal', '#dc2626', '--'),
        ('general_accuracy', 'Arithmetic/copy', '#64748b', ':'),
    ]:
        ax.plot(rows.step, 100*rows[field], color=color, linestyle=style,
                marker='.', markersize=3, linewidth=1.2, label=label)
    ax.set_xscale('symlog', linthresh=1, base=2)
    ax.set(xlim=(0, 256), ylim=(-3, 103), xlabel='Removal SFT updates',
           title=f'Shared 8, learning rate {lr:.0e}')
    ax.set_xticks([0, 1, 4, 16, 64, 256], ['0', '1', '4', '16', '64', '256'])
    ax.set_yticks([0, 25, 50, 75, 100])
axes[0].set_ylabel('Development rate (%)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0, 1, .88), w_pad=1)
fig.savefig(out/'removal_trajectories.pdf', bbox_inches='tight')
plt.close(fig)
manifest = json.loads((out/'overview_provenance.json').read_text())
manifest['inputs'][str(curve_path.relative_to(ROOT))] = sha(curve_path)
manifest['outputs'] = {name: sha(out/name) for name in ['overview.pdf', 'removal_trajectories.pdf']}
manifest['removal_scope'] = 'Shared-eight preserving attacks; one seed; saved development measurements only'
(out/'overview_provenance.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Generated selected removal trajectories; no model calls.')
