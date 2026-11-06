"""CPU audit of saved sampler measurements; never imports the model/training engine."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)):
        return math.isclose(a, b, abs_tol=1e-8, rel_tol=1e-6)
    return a == b


def main(root):
    torch.set_num_threads(2)
    source = root.parent / 'source'
    out = root.parent / 'analysis'
    files = {}
    def read(path):
        raw = path.read_bytes()
        files[str(path.relative_to(root.parent))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    manifest = read(root / 'manifest.json')
    cfg = manifest['protocol']
    assert manifest == read(root.parent / 'prospective_manifest.json')
    assert cfg['policy_steps'] == 1024 and cfg['seed'] == 31 and not cfg['hardware_smoke']
    assert cfg['observational_only'] and cfg['direction_penalty'] == 0 and not cfg['direction_intervention']
    for name, expected in manifest['sources'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    stamp = digest(manifest)
    data = read(root / 'data.json')
    old = read(root / 'direction_data.json')
    assert digest(data) == manifest['data_sha256'] and digest(old) == manifest['observation_sha256']
    checked_truth = 0
    excluded = set(read(source / 'reference_v4/excluded_worlds.json'))
    for rows in data.values():
        assert len(rows) == len({r['id'] for r in rows})
        for row in rows:
            if row['task'] != 'pointer':
                continue
            node, path = row['start'], []
            for command in row['commands']:
                node = row['edges'][node][command]
                path.append(node)
            assert path == row['path'] and row['answer'] == ' '.join(path)
            assert (row['marker'] in path[:-1]) == (row['label'] == 'blocked')
            assert digest(row['edges']) == row['world'] and row['world'] not in excluded
            checked_truth += 1
    worlds = [{r['world'] for r in data['policy_' + panel]} for panel in ['train', 'dev', 'test']]
    assert all(not a & b for i, a in enumerate(worlds) for b in worlds[i+1:])
    assert not {r['prompt'] for r in data['benchmark_dev']} & {r['prompt'] for r in data['benchmark_test']}
    namespace = {'re': re}
    for name, functions in [('reference_v4/reference/removal.py', ['classify']),
                            ('reference_v4/run.py', ['score', 'summarize', 'policy_selection']),
                            ('analyse.py', ['metrics'])]:
        tree = ast.parse((source / name).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
        assert len(nodes) == len(functions)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), namespace)
    assert namespace['classify']('15 + 23 = 38', '38')['correct']
    assert not namespace['classify']('15 + 23 = 39', '38')['correct']
    expected = {name: {r['id']: r for r in rows} for name, rows in data.items()}
    reports = records = 0
    def audit(report, panel):
        nonlocal reports, records
        rows, truth = report['rows'], expected[panel]
        assert len(rows) == len(truth) and {r['id'] for r in rows} == truth.keys()
        for row in rows:
            assert all(row[k] == value for k, value in truth[row['id']].items())
            if panel.startswith('benchmark'):
                lp = row['choice_logprobs']
                assert all(math.isfinite(x) for x in lp)
                assert row['predicted'] == max(range(len(lp)), key=lambda i: lp[i])
                assert row['correct'] == (row['gold'] == row['predicted'])
                assert close(row['gold_nll'], -lp[row['gold']])
                assert close(row['choice_mass'], sum(math.exp(x) for x in lp))
            else:
                assert all(row[k] == value for k, value in namespace['score'](row, row['response']).items())
        if panel.startswith('benchmark'):
            values = {task + '_' + key: sum(r[key] for r in rows if r['task'] == task) / sum(r['task'] == task for r in rows)
                      for task in {r['task'] for r in rows} for key in ['correct', 'gold_nll', 'choice_mass']}
        else:
            values = namespace['summarize'](rows)
        assert close(values, report['metrics']), panel
        reports += 1
        records += len(rows)
    baseline = read(root / 'base.json')
    for panel, report in baseline.items():
        audit(report, panel)
    jobs, status, recipes = (read(root / n) for n in ['jobs.json', 'status.json', 'recipes.json'])
    assert status['phase'] == 'complete' and not status['running'] and not status['failures']
    assert status['completed_policy_fits'] == status['planned_policy_fits'] == 18
    assert status['completed_evaluations'] == status['planned_evaluations'] == 10
    assert not recipes['uses_test'] and not recipes['uses_directions']
    schedules = {mode: [json.loads(x) for x in (root / f'sampling_schedule_{mode}.jsonl').read_text().splitlines()]
                 for mode in ['original', 'balanced']}
    exposure = {}
    for mode, schedule in schedules.items():
        assert len(schedule) == 1024
        exposure[mode] = dict(Counter(r['blocked_case'] for step in schedule for r in step['examples']))
    assert sorted(exposure['original'].values()) == [170, 171, 171, 256, 256, 1024]
    assert sorted(exposure['balanced'].values()) == [341, 341, 341, 341, 342, 342]
    for a, b in zip(schedules['original'], schedules['balanced']):
        assert a['conservation'] == b['conservation']
        assert [(r['allowed'], r['teacher'], r['depth']) for r in a['examples']] == [
            (r['allowed'], r['teacher'], r['depth']) for r in b['examples']]
    candidates, endpoints, resources, observations = {}, [], {}, {}
    native_states = inactive_checks = shared_checks = 0
    last_training = 0.
    policy = {k: s for k, s in jobs.items() if s['kind'] == 'policy'}
    parents = {}
    for key, spec in policy.items():
        folder = root / 'jobs' / key
        done = read(folder / 'DONE.json')
        phase = read(folder / 'policy/DONE.json')
        ident = done['identity']
        assert done['job'] == spec and done['run'] == stamp
        assert ident['job'] == dict(id=key, **spec) and ident['run'] == stamp
        assert phase['identity'] == ident and phase['fingerprint'] == done['fingerprint'] == digest(ident)
        history = read(folder / 'policy/history.json')
        assert len(history) == phase['step'] == 1024 and [h['step'] for h in history] == list(range(1, 1025))
        assert phase['reason'] == 'fixed_update_budget'
        peak = cfg['full_lr' if spec['geometry'] == 'full' else 'sparse_lr'] * spec['lr_factor']
        for h, scheduled in zip(history, schedules[spec['sampler']]):
            assert all(math.isfinite(x) for x in h.values() if isinstance(x, (int, float)))
            assert close(h['learning_rate'], peak * min(1, h['step'] / cfg['warmup']))
            assert h['conservation'] == scheduled['conservation']
            assert h['examples'] == [{k: r[k] for k in ['allowed', 'blocked', 'depth', 'teacher']} for r in scheduled['examples']]
        paths = sorted((folder / 'policy').glob('dev_*.json'))
        assert [int(p.stem[4:]) for p in paths] == list(range(0, 1025, 64))
        selections = []
        for path in paths:
            report = read(path)
            for panel, name in [('task', 'policy_dev'), ('general', 'general_dev'), ('benchmark', 'benchmark_dev')]:
                audit(report[panel], name)
            selections.append(namespace['policy_selection'](report, baseline, cfg, int(path.stem[4:])))
        best = read(folder / 'policy/best.json')
        optimum = max(selections, key=lambda s: s['rank'])
        assert close(best['selection'], optimum) and best['report'] == read(folder / f"policy/dev_{best['step']:04d}.json")
        assert done['qualified'] == phase['qualified'] == optimum['feasible']
        candidates[key] = dict(job=key, lr_factor=spec['lr_factor'], selection=optimum)
        layout = read(folder / 'layout.json')
        assert ident['mask_hash'] == digest(layout['layout'])
        if spec['geometry'] == 'native':
            assert layout['unique_trainable'] == 2688 * spec['width']
            assert all(r['name'].startswith('model.layers.11.mlp.') for r in layout['layout'])
            parent = spec['teacher']
            assert layout['layout'] == read(root / 'jobs' / parent / 'layout.json')['layout']
            if parent not in parents:
                parents[parent] = torch.load(root / 'jobs' / parent / 'capability.pt', map_location='cpu', weights_only=True)
            for filename in ['policy/best.pt', 'final.pt']:
                saved = torch.load(folder / filename, map_location='cpu', weights_only=True)
                assert saved['identity'] == ident and saved['fingerprint'] == digest(ident)
                for name, value in saved['parameters'].items():
                    assert torch.isfinite(value).all()
                    if name.endswith('deltas.0'):
                        assert not torch.equal(value, parents[parent]['parameters'][name])
                        shared_checks += 1
                    elif name.endswith('deltas.1'):
                        assert torch.count_nonzero(value) == 0
                        assert torch.equal(value, parents[parent]['parameters'][name])
                        inactive_checks += 1
                native_states += 1
        last_training = max(last_training, (folder / 'DONE.json').stat().st_mtime)
        resources[key] = read(folder / 'resources.json')
    for name, selected in recipes['recipes'].items():
        condition, sampler = name.split('__')
        choices = [c for key, c in candidates.items() if jobs[key]['condition'] == condition and jobs[key]['sampler'] == sampler]
        assert close(selected, max(choices, key=lambda c: tuple(c['selection']['rank']) + (-c['lr_factor'],)))
    assert not any(c['selection']['feasible'] for c in candidates.values())
    assert shared_checks == inactive_checks == 72
    required = ['base', 'cap_native4_s31', 'cap_native8_s31', 'cap_full_s31'] + [r['job'] for r in recipes['recipes'].values()]
    summary = {}
    for key in required:
        folder = root / 'jobs' / key
        done = read(folder / 'TEST_DONE.json')
        assert done['job'] == jobs[key] and done['run'] == stamp
        resource = read(folder / 'resources_test.json')
        assert resource['started_unix'] >= last_training
        resources[key + '__test'] = resource
        for path in sorted(folder.glob('test_*.json')):
            report = read(path)
            assert report['run'] == stamp and report['job_id'] == key
            for panel, name in [('task', 'policy_test'), ('wording', 'wording_test'), ('ood', 'ood_test'),
                                ('general', 'general_test'), ('benchmark', 'benchmark_test')]:
                audit(report['panels'][panel], name)
            end = report['endpoint']
            tags = dict(job=key, condition=jobs[key].get('condition', key), sampler=jobs[key].get('sampler', ''),
                        endpoint=end, step=report['step'])
            endpoints.append(dict(tags, **namespace['metrics'](report['panels'])))
            summary[key + '__' + end] = dict(step=report['step'], checkpoint_identity=report['checkpoint_identity'],
                panels={name: p['metrics'] for name, p in report['panels'].items()})
            ob = root / 'observations' / key / end
            alignment = read(ob / 'alignment.json')
            assert alignment['observational_only'] and not alignment['causal_ablation_performed'] and not alignment['direction_penalty_used']
            activations = torch.load(ob / 'activations.pt', map_location='cpu', weights_only=True)
            assert activations['parameter_hash'] == activations['parameter_hash_after']
            observations[key + '__' + end] = alignment
            del activations
    original = read(out / 'results.json')
    assert len(endpoints) == 16
    for row in original['endpoints']:
        match = next(r for r in endpoints if r['job'] == row['job'] and r['endpoint'] == row['endpoint'])
        assert all(close(value, match[k]) for k, value in row.items() if k not in ['lr_factor', 'seed'])
    analysis_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
                       if p.suffix in ['.csv', '.pdf', '.json', '.md'] and p.name not in ['audit.json', 'completion_bundle.json']}
    receipt = dict(utc=datetime.now(timezone.utc).isoformat(), checks_passed=True, run=stamp,
        complete_policy_fits=18, completed_evaluation_jobs=10, endpoint_reports=16, failed_jobs=[],
        reports_recounted=reports, saved_prediction_records_recounted=records, pointer_truth_records_checked=checked_truth,
        source_hashes_verified=len(manifest['sources']), development_only_selection_reproduced=True,
        policy_updates_checked=18*1024, training_sample_ids_verified=True, exposure=exposure,
        native_selected_final_parameter_states_checked=native_states, shared_capability_slices_updated=True,
        active_shared_tensor_change_checks=shared_checks, inactive_tensor_zero_checks=inactive_checks,
        native_mlp_block_1_based=12, native_widths=[4, 8], observation_hash_checks=len(observations),
        heldout_after_all_training=True, analysis_file_sha256=analysis_hashes,
        limitations=['One seed; inspected bank', 'Full-model tensors not exhaustively compared',
                     'No model forward or optimization rerun', 'Sampling changes depth and position jointly'])
    receipt['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'completion_bundle.json').write_text(json.dumps(dict(status=status, manifest=manifest, recipes=recipes,
        jobs=jobs, endpoints=summary, resources=resources, observations=observations,
        saved_file_sha256=files), indent=2)+'\n')
    (out / 'audit.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
