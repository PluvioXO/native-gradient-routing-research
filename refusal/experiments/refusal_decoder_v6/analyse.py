"""Summarise the fixed decoder diagnostic after development thresholds are locked."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import pandas as pd

from diagnostic import read, write, sha, digest, sources, rows_at
from logic import auc, summarize


def roc(rows, key):
    scores = np.array([r[key] for r in rows])
    labels = np.array([r['label'] == 'blocked' for r in rows])
    order = np.argsort(-scores, kind='stable'); y = labels[order]; s = scores[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    return np.r_[0, np.cumsum(~y)[ends]/(~y).sum()], np.r_[0, np.cumsum(y)[ends]/y.sum()]


def main(root):
    manifest, calibration = read(root/'manifest.json'), read(root/'calibration.json')
    assert read(root/'status.json')['phase'] == 'complete' and not manifest['smoke']
    assert manifest['sources'] == sources() and calibration['run'] == digest(manifest)
    assert calibration['development_only'] and calibration['selected_before_heldout']
    out = root.parent/'analysis'; out.mkdir(exist_ok=True)
    summaries, cases, differences, distributions = [], [], [], {}
    done_checks = []
    for key, meta in manifest['models'].items():
        assert sha(meta['checkpoint']) == meta['checkpoint_sha256']
        all_rows = []
        for phase in ['dev', 'heldout']:
            done = read(root/phase/key/'DONE.json')
            assert done['run'] == digest(manifest) and done['parameters_before'] == done['parameters_after']
            if phase == 'heldout':
                assert done['calibration_sha256'] == sha(root/'calibration.json')
                assert done['started_unix'] >= (root/'calibration.json').stat().st_mtime
            for filename, expected in done['chunks'].items():
                assert sha(root/phase/key/filename) == expected
            rows = rows_at(root, phase, key)
            assert len(rows) == done['n']
            for row in rows:
                values = np.array(row['node_logits'], dtype=float)
                assert len(values) == 16 and np.isfinite(values).all()
                lse = float(values.max()+np.log(np.exp(values-values.max()).sum()))
                assert abs(row['ref_logit']-lse-row['margin_sum']) < 1e-4
                assert abs(row['ref_logit']-values.max()-row['margin_max']) < 1e-4
            all_rows.extend(rows)
            done_checks.append(dict(model=key, phase=phase, n=len(rows), parameters_unchanged=True))
        distributions[key] = all_rows
        condition, sampler = key.split('__')
        threshold = calibration['models'][key]['chosen']['threshold']
        for split in ['dev', 'test', 'wording', 'ood']:
            rows = [r for r in all_rows if r['split'] == split]
            assert rows
            tags = dict(model=key, condition=condition, sampler=sampler, split=split)
            for method in ['greedy', 'argmax', 'sum_zero', 'sum_calibrated']:
                t = threshold if method == 'sum_calibrated' else 0.
                result = summarize(rows, method, t)
                groups = result.pop('cases')
                summaries.append(dict(tags, method=method, threshold=t, auc_sum=auc(rows, 'margin_sum'),
                    auc_max=auc(rows, 'margin_max'), retention_ok=calibration['models'][key]['retention_ok'],
                    qualifies_with_original_retention=bool(result['task_gates_passed'] and calibration['models'][key]['retention_ok']),
                    **result))
                cases.extend(dict(tags, method=method, threshold=t, **group) for group in groups)
            allowed_false = [r for r in rows if r['label'] != 'blocked' and r['greedy_refusal']]
            canonical = [r for r in allowed_false if r['raw_top_is_ref'] and r['argmax_refusal']]
            discordant = [r for r in canonical if r['margin_sum'] <= 0]
            differences.append(dict(tags, original_false_refusals=len(allowed_false),
                canonical_refusal_starts=len(canonical), sum_max_disagreements=len(discordant),
                disagreements_fraction_of_false=len(discordant)/len(allowed_false) if allowed_false else 0.,
                disagreements_with_correct_forced_path=sum(r['node_path_correct'] for r in discordant),
                other_first_tokens=sum(not r['raw_top_is_ref'] and not r['raw_top_is_node'] for r in rows),
                literal_refusal_with_noncanonical_start=sum(r['greedy_refusal'] and not r['raw_top_is_ref'] for r in rows)))
    frame = pd.DataFrame(summaries)
    assert len(frame) == 6*4*4
    for filename, values in [('endpoints.csv', summaries), ('case_scores.csv', cases), ('disagreements.csv', differences)]:
        pd.DataFrame(values).to_csv(out/filename, index=False)
    cal_rows = [dict(model=key, **candidate) for key, r in calibration['models'].items() for candidate in r['candidates']]
    pd.DataFrame(cal_rows).to_csv(out/'development_thresholds.csv', index=False)
    plot(out, frame, pd.DataFrame(differences), distributions, calibration)
    result = dict(manifest_sha256=sha(root/'manifest.json'), calibration_sha256=sha(root/'calibration.json'),
        development_only_thresholds=True, models=manifest['models'], endpoints=summaries,
        disagreements=differences, measurement_checks=done_checks,
        optimizer_or_training=False, source_checkpoints_changed=False,
        scope='One V5 seed and inspected graph bank; task-specific decoding controls; no refusal-removal attack',
        retention_scope='Copied from original V5 model; gate not applied to non-navigation tasks')
    write(out/'results.json', result)
    lines = ['# Saved-model decoder diagnostic', '',
        'One seed, inspected V5 bank. Same model weights; development-only thresholds.', '',
        '| Model | Method | Allowed (%) | Refusal (%) | False refusal (%) | Worst joint (%) |',
        '|---|---|---:|---:|---:|---:|']
    for r in frame[frame.split == 'test'].to_dict('records'):
        lines.append(f"| {r['model']} | {r['method']} | {100*r['allowed_accuracy']:.1f} | {100*r['blocked_refusal']:.1f} | {100*r['false_refusal']:.1f} | {100*r['worst_joint']:.1f} |")
    lines += ['', 'Restricted argmax and both sum gates share a forced-literal refusal and the same node-first continuation.',
        'Their comparison to original greedy also changes vocabulary/grammar. None changes model weights or establishes attack resistance.',
        'Retain original V5 greedy outcomes as originally reported. No test/probe threshold selection.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(out/'provenance.json', dict(sources=manifest['sources'], manifest_sha256=sha(root/'manifest.json'),
        calibration_sha256=sha(root/'calibration.json'),
        files={p.name: sha(p) for p in sorted(out.iterdir()) if p.name != 'provenance.json'}))
    print('Completed fixed decoder analysis:', out, flush=True)


def plot(out, frame, differences, distributions, calibration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    order = ['shared4__original','shared4__balanced','shared8__original','shared8__balanced','full__original','full__balanced']
    titles = ['Native 4 / original','Native 4 / balanced','Native 8 / original','Native 8 / balanced','Full / original','Full / balanced']
    colours = ['#64748b','#d97706','#2563eb','#0f766e']
    methods = ['greedy','argmax','sum_zero','sum_calibrated']
    names = ['Original greedy','Restricted argmax','Sum gate (0)','Sum gate (dev threshold)']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    test = frame[frame.split == 'test']
    with PdfPages(out/'decoder_graphs.pdf') as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        for ax, field, title in zip(axes.flat, ['allowed_accuracy','blocked_refusal','false_refusal','worst_joint'],
                                    ['Allowed correct','Blocked refused','False refusal','Worst-case joint quality']):
            for j, (method, colour, name) in enumerate(zip(methods, colours, names)):
                values = [100*test[(test.model == k)&(test.method == method)].iloc[0][field] for k in order]
                ax.bar(np.arange(6)+(j-1.5)*.19, values, width=.18, color=colour, label=name)
            ax.set_xticks(range(6), ['4/O','4/B','8/O','8/B','Full/O','Full/B'])
            ax.set(title=title, ylim=(0,105), ylabel='Rate (%)'); ax.legend(frameon=False, fontsize=8)
        fig.suptitle('Fixed checkpoints: four decoding rules — one seed, inspected bank')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for ax, key, title in zip(axes.flat, order, titles):
            rows = [r for r in distributions[key] if r['split'] == 'test']
            for blocked, colour, label in [(False,'#2563eb','Allowed'),(True,'#d97706','Blocked')]:
                values = [r['margin_sum'] for r in rows if (r['label'] == 'blocked') == blocked]
                ax.hist(values, bins=35, density=True, alpha=.5, color=colour, label=label)
            ax.axvline(0, color='black', ls='--', lw=.8, label='Loss threshold 0')
            ax.axvline(calibration['models'][key]['chosen']['threshold'], color='#0f766e', ls=':', label='Dev-selected threshold')
            ax.set(title=title, xlabel='Refusal logit − logsumexp(node logits)', ylabel='Density')
            ax.legend(frameon=False, fontsize=7)
        fig.suptitle('Held-out score overlap; thresholds selected only on development')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for ax, key, title in zip(axes.flat, order, titles):
            rows = [r for r in distributions[key] if r['split'] == 'test']
            for metric, colour, label in [('margin_sum','#2563eb','Sum margin'),('margin_max','#d97706','Max margin')]:
                x, y = roc(rows, metric)
                ax.plot(x, y, color=colour, label=f'{label}, AUC {auc(rows,metric):.3f}')
            ax.plot([0,1],[0,1],ls='--',color='gray',lw=.7)
            ax.set(title=title, xlabel='False refusal rate', ylabel='Blocked refusal rate', xlim=(0,1), ylim=(0,1))
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle('Test ROC is descriptive; it never chooses a deployed threshold')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        d = differences[differences.split == 'test'].set_index('model').loc[order]
        axes[0].bar(range(6), 100*d.disagreements_fraction_of_false, color='#2563eb')
        axes[0].set(ylabel='Original false refusals (%)', title='Canonical sum-versus-max disagreement', ylim=(0,105))
        axes[1].bar(range(6), d.disagreements_with_correct_forced_path, color='#0f766e')
        axes[1].set(ylabel='Questions', title='Disagreements with a recoverable correct path')
        for ax in axes:ax.set_xticks(range(6), ['4/O','4/B','8/O','8/B','Full/O','Full/B'])
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        for ax, split in zip(axes.flat, ['dev','test','wording','ood']):
            for method, colour, label in zip(methods,colours,names):
                vals = [100*frame[(frame.model == k)&(frame.method == method)&(frame.split == split)].iloc[0].worst_joint for k in order]
                ax.plot(range(6), vals, 'o-', color=colour, label=label)
            ax.set_xticks(range(6), ['4/O','4/B','8/O','8/B','Full/O','Full/B'])
            ax.set(title=split, ylabel='Worst-case joint quality (%)', ylim=(0,105));ax.legend(frameon=False,fontsize=7)
        fig.suptitle('Same locked development threshold on each held-out panel')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root.resolve())
