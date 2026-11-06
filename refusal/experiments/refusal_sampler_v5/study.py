#!/usr/bin/env python3
"""Original versus balanced refusal sampling from verified V4 capability parents."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'reference_v4'))
import run as engine
import sampling

REFERENCE_SOURCES = engine.source_hashes
PARENTS = {'shared4': 'cap_native4_s31', 'shared8': 'cap_native8_s31', 'full': 'cap_full_s31'}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    values = {'reference_v4/' + k: v for k, v in REFERENCE_SOURCES().items()}
    for path in sorted(HERE.glob('*.py')):
        values[path.name] = digest_file(path)
    for name in ['PROTOCOL.md', 'launch/v4_source.tar.gz', 'launch/v4_manifest.json',
                 'launch/engine_lineage.json', 'launch/import_sources.json']:
        values[name] = digest_file(HERE / name)
    return values


engine.source_hashes = source_hashes


def config(smoke=False):
    cfg = copy.deepcopy(engine.CFG)
    cfg.update(version=5, study='original-versus-balanced-refusal-sampling', hardware_smoke=smoke,
               samplers=['original', 'balanced'], policy_steps=1024, eval_every=64,
               imported_acquisitions=True, inference='exploratory; inspected V4 bank; one seed')
    if smoke:
        cfg.update(policy_steps=3, eval_every=1, lr_factors=[.1], dev_worlds=1, test_worlds=1,
                   ood_worlds=1, wording_worlds=1, observe_worlds=1, eval_max_new_tokens=12)
    return cfg


def make_jobs(old_jobs, cfg):
    jobs = {'base': dict(kind='base', seed=31)}
    jobs.update({name: old_jobs[name] for name in PARENTS.values()})
    for condition, parent in PARENTS.items():
        for sampler in cfg['samplers']:
            for factor in cfg['lr_factors']:
                key = f'policy_{condition}_{sampler}_s31_lr{round(100 * factor):03d}'
                jobs[key] = dict(kind='policy', condition=condition, sampler=sampler, seed=31,
                                 geometry='full' if condition == 'full' else 'native',
                                 width=0 if condition == 'full' else int(condition[6:]),
                                 placement='same', teacher=parent, lr_factor=factor, calibration=True)
    return jobs


def freeze_json(path, value):
    if path.exists():
        if engine.read(path) != value:
            raise ValueError(f'Frozen input differs: {path}')
    else:
        engine.core.atomic_json(path, value)


def small_panels(data, old):
    data = copy.deepcopy(data)
    for key, rows in data.items():
        if not rows or key in ['policy_train', 'preserve_train', 'rehearsal']:
            continue
        if 'world' in rows[0]:
            world = rows[0]['world']
            data[key] = [r for r in rows if r['world'] == world]
        elif 'task' in rows[0]:
            data[key] = [r for task in sorted({r['task'] for r in rows})
                         for r in [q for q in rows if q['task'] == task][:2]]
    old = copy.deepcopy(old)
    for key in ['fit', 'heldout']:
        old[key] = [r for label in ['safe', 'unsafe'] for r in [q for q in old[key] if q['label'] == label][:4]]
    return data, old


def initialize(root, imports, smoke=False):
    cfg = config(smoke)
    root.mkdir(parents=True, exist_ok=True)
    engine.free_space(root)
    expected = engine.read(HERE / 'launch/import_sources.json')
    verified = {}
    for name, record in expected['files'].items():
        path = imports / name
        if path.stat().st_size != record['bytes'] or digest_file(path) != record['sha256']:
            raise ValueError(f'Imported file checksum differs: {name}')
        verified[name] = record
    old_manifest = engine.read(imports / 'results/manifest.json')
    assert old_manifest == engine.read(HERE / 'launch/v4_manifest.json')
    data = engine.read(imports / 'results/data.json')
    old = engine.read(imports / 'results/direction_data.json')
    assert engine.core.digest(data) == old_manifest['data_sha256']
    assert engine.core.digest(old) == old_manifest['observation_sha256']
    exposure = sampling.audit(data, config(False), engine.pick)
    assert exposure['counts']['original'] == {
        'h2_blocked_1': 1024, 'h3_blocked_1': 256, 'h3_blocked_2': 256,
        'h4_blocked_1': 170, 'h4_blocked_2': 171, 'h4_blocked_3': 171}
    # The exact historical sampler is also independently recounted in tests/preflight.
    if smoke:
        data, old = small_panels(data, old)
    jobs = make_jobs(engine.read(imports / 'results/jobs.json'), cfg)
    manifest = dict(protocol=cfg, sources=source_hashes(), data_sha256=engine.core.digest(data),
                    observation_sha256=engine.core.digest(old), parents=verified,
                    parent_manifest_sha256=digest_file(imports / 'results/manifest.json'))
    for name, value in [('manifest.json', manifest), ('data.json', data), ('direction_data.json', old),
                        ('jobs.json', jobs), ('sampling_audit.json', exposure)]:
        freeze_json(root / name, value)
    for mode in cfg['samplers']:
        path = root / f'sampling_schedule_{mode}.jsonl'
        payload = ''.join(json.dumps(row, sort_keys=True) + '\n'
                          for row in sampling.schedule(data, cfg, mode, engine.pick))
        if path.exists():
            assert path.read_text() == payload
        else:
            path.write_text(payload)
    for parent in PARENTS.values():
        directory = root / 'jobs' / parent
        directory.mkdir(parents=True, exist_ok=True)
        for name in ['capability.pt', 'teacher.pt', 'layout.json', 'acquisition.json']:
            source = imports / 'results/jobs' / parent / name
            destination = directory / name
            if not destination.exists():
                os.link(source, destination)
        acquisition = engine.read(directory / 'acquisition.json')
        assert acquisition['qualified']
        freeze_json(directory / 'DONE.json', dict(job=jobs[parent], run=engine.core.digest(manifest),
            qualified=True, imported=True, source=expected['files'][f'results/jobs/{parent}/capability.pt']))
    return cfg, manifest


def recipes(root, jobs):
    result = {}
    for condition in PARENTS:
        for sampler in ['original', 'balanced']:
            candidates = []
            for key, job in jobs.items():
                if job.get('condition') == condition and job.get('sampler') == sampler:
                    report = engine.read(root / 'jobs' / key / 'policy/best.json')
                    assert report['identity']['job'] == dict(id=key, **job)
                    candidates.append(dict(job=key, lr_factor=job['lr_factor'], selection=report['selection']))
            result[condition + '__' + sampler] = max(
                candidates, key=lambda c: tuple(c['selection']['rank']) + (-c['lr_factor'],))
    return dict(recipes=result, uses_test=False, uses_directions=False,
                failures_retained=True, no_all_conditions_success_gate=True)


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(row.split(',')[0]): int(row.split(',')[1]) for row in output.strip().splitlines()}


def launch(root, gpus, smoke=False):
    cfg = config(smoke)
    manifest = engine.read(root / 'manifest.json')
    assert manifest['protocol'] == cfg and manifest['sources'] == source_hashes()
    lock = (root / 'launcher.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = engine.read(root / 'jobs.json')
    stamp = engine.core.digest(manifest)
    (root / 'logs').mkdir(exist_ok=True)
    active, failures = {}, []
    policy = [k for k, v in jobs.items() if v['kind'] == 'policy']
    def done(key, test=False):
        path = root / 'jobs' / key / ('TEST_DONE.json' if test else 'DONE.json')
        if not path.exists():
            return False
        report = engine.read(path)
        assert report['job'] == jobs[key] and report['run'] == stamp
        return True
    while True:
        for pid, (process, key, test, assigned, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[pid]
            print(json.dumps(dict(event='exit', job=key, test=test, code=code)), flush=True)
            if code or not done(key, test):
                failures.append(dict(job=key, test=test, exit_code=code))
        trained = done('base') and all(done(k) for k in policy)
        if trained:
            freeze_json(root / 'recipes.json', recipes(root, jobs))
            selected = [c['job'] for c in engine.read(root / 'recipes.json')['recipes'].values()]
            required = ['base', *PARENTS.values(), *selected]
            tasks = [(k, True) for k in required if not done(k, True) and
                     (jobs[k]['kind'] != 'policy' or done(jobs[k]['teacher'], True))]
        else:
            tasks = [('base', False)] if not done('base') else [(k, False) for k in policy if not done(k)]
        occupied = {g for _, _, _, assigned, _ in active.values() for g in assigned}
        memory = gpu_memory()
        available = [g for g in gpus if g not in occupied and memory[g] < 200]
        busy = {(key, test) for _, key, test, _, _ in active.values()}
        if not failures:
            # Start two-GPU controls before they can be starved by one-GPU fits.
            tasks.sort(key=lambda t: (not (jobs[t[0]].get('geometry') == 'full' and not t[1]), t[0]))
            for key, test in tasks:
                count = 2 if jobs[key].get('geometry') == 'full' and not test else 1
                if (key, test) in busy or len(available) < count:
                    continue
                assigned, available = available[:count], available[count:]
                command = [sys.executable, '-u', str(HERE / 'study.py'), 'worker', '--root', str(root), '--job', key]
                if smoke:
                    command.append('--smoke')
                if test:
                    command.append('--test')
                handle = (root / 'logs' / (key + ('__test' if test else '') + '.log')).open('a')
                process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, assigned))),
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[process.pid] = (process, key, test, assigned, handle)
                print(json.dumps(dict(event='launch', job=key, test=test, pid=process.pid, gpus=assigned)), flush=True)
        status = dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      phase='held_out_evaluation' if trained else 'training',
                      planned_policy_fits=len(policy), completed_policy_fits=sum(done(k) for k in policy),
                      running=[dict(pid=pid, job=k, test=t, gpus=g) for pid, (_, k, t, g, _) in active.items()],
                      failures=failures, free_gib=shutil.disk_usage(root).free / 2**30)
        if trained:
            status.update(planned_evaluations=len(required), completed_evaluations=sum(done(k, True) for k in required))
        engine.core.atomic_json(root / 'status.json', status)
        if failures and not active:
            raise RuntimeError(f'Worker failure; checkpoints retained: {failures}')
        if trained and all(done(k, True) for k in required) and not active:
            status['phase'] = 'complete'
            engine.core.atomic_json(root / 'status.json', status)
            break
        time.sleep(5)


def worker(root, job_id, smoke, test):
    cfg = config(smoke)
    manifest = engine.read(root / 'manifest.json')
    assert manifest['protocol'] == cfg and manifest['sources'] == source_hashes()
    spec = engine.read(root / 'jobs.json')[job_id]
    if spec['kind'] == 'policy':
        for name in ['capability.pt', 'teacher.pt']:
            expected = manifest['parents'][f"results/jobs/{spec['teacher']}/{name}"]['sha256']
            assert digest_file(root / 'jobs' / spec['teacher'] / name) == expected
    start = time.time()
    output = root / 'jobs' / job_id / ('resources_test.json' if test else 'resources.json')
    try:
        engine.worker(root, cfg, job_id, test)
    finally:
        engine.core.atomic_json(output, dict(started_unix=start, ended_unix=time.time(),
            wall_seconds=time.time()-start, visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
            host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_allocated_bytes=[engine.torch.cuda.max_memory_allocated(i)
                                       for i in range(engine.torch.cuda.device_count())],
            cuda_peak_reserved_bytes=[engine.torch.cuda.max_memory_reserved(i)
                                      for i in range(engine.torch.cuda.device_count())]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['initialize', 'run', 'worker', 'status'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--imports', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3,4,5')
    parser.add_argument('--job')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'initialize':
        initialize(root, args.imports.resolve(), args.smoke)
        print(json.dumps(dict(initialized=str(root), smoke=args.smoke)))
    elif args.command == 'run':
        gpus = [int(x) for x in args.gpus.split(',')]
        assert len(gpus) >= 2 and len(set(gpus)) == len(gpus)
        launch(root, gpus, args.smoke)
    elif args.command == 'worker':
        worker(root, args.job, args.smoke, args.test)
    else:
        print(json.dumps(engine.read(root / 'status.json'), indent=2))


if __name__ == '__main__':
    main()
