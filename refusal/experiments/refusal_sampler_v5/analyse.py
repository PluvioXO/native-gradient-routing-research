#!/usr/bin/env python3
"""Summarise completed sampler fits without rerunning models or changing selection."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(report):
    rows = report['task']['rows']
    allowed = [r for r in rows if r['label'] != 'blocked']
    blocked = [r for r in rows if r['label'] == 'blocked']
    mean = lambda pool, key: sum(bool(r[key]) for r in pool) / len(pool)
    groups = report['task']['metrics']['by_case']
    a = [v for k, v in groups.items() if '_blocked_' not in k]
    b = [v for k, v in groups.items() if '_blocked_' in k]
    return dict(allowed_accuracy=mean(allowed, 'path_correct'), blocked_refusal=mean(blocked, 'refusal'),
                false_refusal=mean(allowed, 'refusal'), capability_trace_all=mean(rows, 'path_correct'),
                worst_joint=min(min(v['path_correct'] for v in a), min(v['refusal'] for v in b),
                                1 - max(v['refusal'] for v in a)),
                malformed=1 - mean(rows, 'format_valid'), truncated=mean(rows, 'truncated'),
                general_exact=report['general']['metrics']['path_correct'],
                general_parsed=report['general']['metrics']['answer_parsed_correct'],
                **report['benchmark']['metrics'],
                **{'general_' + k + '_exact': v['path_correct']
                   for k, v in report['general']['metrics']['by_case'].items()})


def analyse(root):
    manifest = read(root / 'manifest.json')
    assert read(root / 'status.json')['phase'] == 'complete'
    jobs = read(root / 'jobs.json')
    recipes = read(root / 'recipes.json')['recipes']
    output = root.parent / ('smoke_analysis' if manifest['protocol']['hardware_smoke'] else 'analysis')
    output.mkdir(exist_ok=True)
    endpoints, curves, cases, candidates, losses = [], [], [], [], []
    selected = {r['job'] for r in recipes.values()}
    for key, job in jobs.items():
        directory = root / 'jobs' / key
        tags = dict(job=key, condition=job.get('condition', key), sampler=job.get('sampler', ''),
                    lr_factor=job.get('lr_factor', ''), seed=job['seed'])
        if job['kind'] == 'policy':
            for path in sorted((directory / 'policy').glob('dev_*.json')):
                report = read(path)
                curves.append(dict(tags, step=int(path.stem[4:]), **metrics(report)))
                for case, values in report['task']['metrics']['by_case'].items():
                    cases.append(dict(tags, panel='dev', step=int(path.stem[4:]), case=case, **values))
            best = read(directory / 'policy/best.json')
            history = read(directory / 'policy/history.json')
            schedule = [json.loads(line) for line in (root / f"sampling_schedule_{job['sampler']}.jsonl").read_text().splitlines()]
            candidates.append(dict(tags, selected=key in selected, selected_step=best['step'],
                                   feasible=best['selection']['feasible'],
                                   train_seconds=sum(x['train_seconds'] for x in history),
                                   **{k: v for k, v in best['selection'].items() if k not in ['rank', 'step', 'feasible']}))
            assert len(history) == manifest['protocol']['policy_steps']
            for row, expected in zip(history, schedule):
                assert len(row['examples']) == 2 and all('teacher' in r for r in row['examples'])
                assert row['conservation'] == expected['conservation']
                assert row['examples'] == [{k: r[k] for k in ['allowed', 'blocked', 'depth', 'teacher']}
                                           for r in expected['examples']]
                losses.append(dict(tags, **{k: v for k, v in row.items() if isinstance(v, (int, float))}))
        for path in sorted(directory.glob('test_*.json')):
            report = read(path)
            values = metrics(report['panels'])
            endpoints.append(dict(tags, endpoint=report['endpoint'], step=report['step'], **values))
            for panel in ['task', 'wording', 'ood']:
                for case, values in report['panels'][panel]['metrics']['by_case'].items():
                    cases.append(dict(tags, panel=panel, step=report['step'], endpoint=report['endpoint'],
                                      case=case, **values))
    contrasts = []
    for condition in ['shared4', 'shared8', 'full']:
        for endpoint in ['best', 'final']:
            pair = {r['sampler']: r for r in endpoints if r['condition'] == condition and r['endpoint'] == endpoint}
            assert set(pair) == {'original', 'balanced'}
            contrasts.append(dict(condition=condition, endpoint=endpoint,
                **{k + '_change_pp': 100 * (pair['balanced'][k] - pair['original'][k])
                   for k in ['allowed_accuracy', 'blocked_refusal', 'false_refusal', 'worst_joint', 'general_exact']}))
    for name, rows in [('endpoints', endpoints), ('learning_curves', curves), ('case_scores', cases),
                       ('all_candidates', candidates), ('paired_contrasts', contrasts), ('losses', losses)]:
        write_csv(output / (name + '.csv'), rows)
    summary = dict(protocol=manifest['protocol'], recipes=recipes, endpoints=endpoints, contrasts=contrasts,
                   scientific_result=not manifest['protocol']['hardware_smoke'],
                   limitations=['one seed', 'previously inspected V4 bank', 'no removal attack',
                                'joint depth/position sampling change; FLOPs not measured'])
    (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(output, endpoints, curves, cases, recipes, losses)
    lines = ['# Selective-refusal sampling pilot', '',
             'One seed; original versus balanced positive sampling. All candidates and failed gates are retained.', '',
             '| Model | Sampler | Endpoint | Allowed correct | Blocked refused | False refusal | Worst joint |',
             '|---|---|---|---:|---:|---:|---:|']
    for r in endpoints:
        if r['sampler']:
            lines.append(f"| {r['condition']} | {r['sampler']} | {r['endpoint']} | {r['allowed_accuracy']:.1%} | "
                         f"{r['blocked_refusal']:.1%} | {r['false_refusal']:.1%} | {r['worst_joint']:.1%} |")
    lines.extend(['', 'See the PDF, all-candidate curves, grouped case scores and paired contrasts.',
                  'This pilot does not estimate across-seed uncertainty or demonstrate attack resistance.'])
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(analysis=str(output), endpoint_reports=len(endpoints), policy_candidates=len(candidates))))


def plot(output, endpoints, curves, cases, recipes, losses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    conditions = ['shared4', 'shared8', 'full']
    colours = {'original': '#c77416', 'balanced': '#2563eb'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    with PdfPages(output / 'sampling_graphs.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, metric, title in zip(axes, ['allowed_accuracy', 'blocked_refusal', 'false_refusal'],
                                    ['Allowed correct', 'Blocked refused', 'False refusal']):
            for offset, sampler in [(-.19, 'original'), (.19, 'balanced')]:
                values = [next(r for r in endpoints if r['condition'] == c and r['sampler'] == sampler
                               and r['endpoint'] == 'final')[metric] for c in conditions]
                ax.bar([x + offset for x in range(3)], [100 * x for x in values], width=.36,
                       color=colours[sampler], label=sampler)
            ax.set_xticks(range(3), conditions)
            ax.set_ylim(0, 105)
            ax.set_title(title)
            ax.legend(frameon=False)
        axes[0].set_ylabel('Final held-out rate (%)')
        fig.suptitle('Selected learning rates, fixed final update — one seed')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for metric, title in [('allowed_accuracy', 'Allowed development accuracy'),
                              ('blocked_refusal', 'Blocked development refusal'),
                              ('false_refusal', 'False refusal on allowed development cases'),
                              ('worst_joint', 'Worst development case quality')]:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
            for ax, condition in zip(axes, conditions):
                for sampler in ['original', 'balanced']:
                    chosen = recipes[condition + '__' + sampler]['job']
                    for job in sorted({r['job'] for r in curves if r['condition'] == condition and r['sampler'] == sampler}):
                        values = sorted([r for r in curves if r['job'] == job], key=lambda r: r['step'])
                        ax.plot([r['step'] for r in values], [100 * r[metric] for r in values],
                                color=colours[sampler], alpha=1 if job == chosen else .25,
                                lw=2 if job == chosen else 1, label=sampler if job == chosen else None)
                ax.set_title(condition); ax.set_xlabel('Policy update'); ax.set_ylim(0, 105)
                ax.legend(frameon=False)
            axes[0].set_ylabel('Rate (%)'); fig.suptitle(title + ' — all three rates shown')
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        groups = ['h2_blocked_1', 'h3_blocked_1', 'h3_blocked_2', 'h4_blocked_1', 'h4_blocked_2', 'h4_blocked_3']
        for ax, condition in zip(axes, conditions):
            for sampler in ['original', 'balanced']:
                job = recipes[condition + '__' + sampler]['job']
                rows = [r for r in cases if r['job'] == job and r['panel'] == 'task' and r.get('endpoint') == 'final']
                ax.plot(range(6), [100 * next(r['refusal'] for r in rows if r['case'] == c) for c in groups],
                        'o-', color=colours[sampler], label=sampler)
            ax.set_xticks(range(6), groups, rotation=55, ha='right'); ax.set_title(condition)
            ax.set_ylim(0, 105); ax.legend(frameon=False)
        axes[0].set_ylabel('Correct refusal (%)'); fig.suptitle('Final held-out blocked groups')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        for column, condition in enumerate(conditions):
            for row, metric in enumerate(['capability_nll', 'positive_policy', 'negative_policy']):
                ax = axes[row, column]
                for sampler in ['original', 'balanced']:
                    key = recipes[condition + '__' + sampler]['job']
                    values = [v for v in losses if v['job'] == key]
                    # Mean adjacent windows to retain trends without dense step noise.
                    windows = [values[i:i+16] for i in range(0, len(values), 16)]
                    ax.plot([w[-1]['step'] for w in windows],
                            [sum(v[metric] for v in w)/len(w) for w in windows],
                            color=colours[sampler], label=sampler)
                ax.set_title(condition + ': ' + metric); ax.set_xlabel('Update'); ax.legend(frameon=False)
        fig.suptitle('Training losses at selected rates — 16-update window means')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fields = ['general_addition_exact', 'general_copy_exact', 'arc_easy_correct',
                  'winogrande_correct', 'arc_easy_choice_mass', 'winogrande_choice_mass']
        base = next(r for r in endpoints if r['condition'] == 'base')
        parents = dict(shared4='cap_native4_s31', shared8='cap_native8_s31', full='cap_full_s31')
        for ax, field in zip(axes.flat, fields):
            for offset, sampler in [(-.19, 'original'), (.19, 'balanced')]:
                values = [next(r[field] for r in endpoints if r['condition'] == c and r['sampler'] == sampler
                               and r['endpoint'] == 'final') - base[field] for c in conditions]
                ax.bar([x + offset for x in range(3)], [100*v for v in values], width=.36,
                       color=colours[sampler], label=sampler)
            before = [100*(next(r[field] for r in endpoints if r['condition'] == parents[c]) - base[field]) for c in conditions]
            ax.scatter(range(3), before, marker='D', color='black', zorder=4, label='capability parent')
            ax.axhline(0, color='gray', ls='--', lw=1); ax.set_xticks(range(3), conditions)
            ax.set_title(field); ax.set_ylabel('Change from base (pp)'); ax.legend(frameon=False, fontsize=8)
        fig.suptitle('Collateral changes, including damage before policy training')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    analyse(parser.parse_args().root.resolve())
