"""Read-only GPU measurements on fixed V5 checkpoints, with a dev/test barrier."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from logic import calibrate

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < 40*1024**3:
        raise OSError('Less than 40 GiB free; previous measurements retained')
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sources():
    paths = list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md', HERE/'v5_bundle.json', HERE/'v5_audit.json']
    return {p.name: sha(p) for p in sorted(paths)}


def initialize(root, old_root, smoke, hardware):
    assert old_root not in root.parents and root != old_root
    bundle, audit = read(HERE/'v5_bundle.json'), read(HERE/'v5_audit.json')
    assert audit['checks_passed'] and audit['run'] == digest(bundle['manifest'])
    assert read(old_root/'manifest.json') == bundle['manifest']
    old_source = old_root.parent/'source'
    for name, expected in bundle['manifest']['sources'].items():
        assert sha(old_source/name) == expected, name
    models = {}
    for key, recipe in bundle['recipes']['recipes'].items():
        job = recipe['job']
        folder = old_root/'jobs'/job
        done = read(folder/'DONE.json')
        assert done['run'] == audit['run'] and done['job'] == bundle['jobs'][job]
        refs = {}
        for panel, relative in [('dev', f'jobs/{job}/policy/dev_1024.json'),
                                ('heldout', f'jobs/{job}/test_final.json')]:
            expected = bundle['saved_file_sha256']['results/'+relative]
            assert sha(old_root/relative) == expected
            refs[panel] = dict(path=str(old_root/relative), sha256=expected)
        models[key] = dict(job=job, spec=bundle['jobs'][job], identity=done['identity'],
            checkpoint=str(folder/'final.pt'), checkpoint_sha256=sha(folder/'final.pt'), refs=refs)
    value = dict(protocol='saved-model-decoder-diagnostic-v1', smoke=smoke, sources=sources(),
        old_root=str(old_root), old_run=audit['run'], old_source=str(old_source),
        model_protocol=bundle['manifest']['protocol'], models=models,
        batch_size=4, chunk_size=64, max_new_tokens=32, optimizer_created=False)
    if not smoke:
        h = read(hardware/'manifest.json')
        assert h['smoke'] and h['sources'] == value['sources'] and h['models'] == models
        for key in models:
            done = read(hardware/'smoke'/key/'DONE.json')
            assert done['run'] == digest(h) and done['baseline_replay_exact'] and done['n'] == 32
            assert done['parameters_before'] == done['parameters_after']
            assert done['resume_chunks_verified'] >= 1
        value['hardware_manifest_sha256'] = sha(hardware/'manifest.json')
    path = root/'manifest.json'
    if path.exists():
        assert read(path) == value, 'Frozen diagnostic input differs'
    else:
        write(path, value)
    print(json.dumps(dict(initialized=str(root), smoke=smoke, models=len(models))))


def rows_at(root, phase, key):
    result = []
    for path in sorted((root/phase/key).glob('chunk_*.json')):
        result.extend(read(path)['rows'])
    return result


def retention(report, baseline, cfg):
    g = report['general']['metrics']['path_correct']
    b, original = report['benchmark']['metrics'], baseline['benchmark_dev']['metrics']
    return bool(g >= baseline['general_dev']['metrics']['path_correct']-cfg['retention_drop'] and
        all(b[t+'_correct'] >= original[t+'_correct']-cfg['retention_drop'] and
            b[t+'_choice_mass'] >= cfg['choice_mass_ratio']*original[t+'_choice_mass']
            for t in ['arc_easy', 'winogrande']))


def calibrate_all(root):
    manifest = read(root/'manifest.json')
    assert not manifest['smoke'] and manifest['sources'] == sources()
    result = {}
    for key in manifest['models']:
        done = read(root/'dev'/key/'DONE.json')
        assert done['run'] == digest(manifest) and done['parameters_before'] == done['parameters_after']
        rows = rows_at(root, 'dev', key)
        assert len(rows) == done['n'] == 1248
        result[key] = calibrate(rows, done['retention_ok'])
    value = dict(run=digest(manifest), sources=manifest['sources'], development_only=True,
        selected_before_heldout=True, models=result)
    path = root/'calibration.json'
    if path.exists():
        assert read(path) == value
    else:
        write(path, value)
    print('All six development thresholds frozen before held-out collection', flush=True)


def load_engine(manifest):
    sys.path.insert(0, manifest['old_source'])
    import study
    assert study.source_hashes() == read(Path(manifest['old_root'])/'manifest.json')['sources']
    return study.engine


def worker(root, phase, key):
    started = time.time()
    manifest = read(root/'manifest.json')
    assert manifest['sources'] == sources() and key in manifest['models']
    if phase == 'heldout':
        calibration = read(root/'calibration.json')
        assert not manifest['smoke'] and calibration['run'] == digest(manifest) and calibration['development_only']
        assert started >= (root/'calibration.json').stat().st_mtime
        assert all((root/'dev'/k/'DONE.json').exists() for k in manifest['models'])
    else:
        assert phase == ('smoke' if manifest['smoke'] else 'dev')
    folder = root/phase/key
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'worker.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder/'DONE.json').exists():
        assert read(folder/'DONE.json')['run'] == digest(manifest)
        return
    meta = manifest['models'][key]
    assert sha(meta['checkpoint']) == meta['checkpoint_sha256']
    ref = meta['refs']['heldout' if phase == 'heldout' else 'dev']
    assert sha(ref['path']) == ref['sha256']
    reference = read(ref['path'])
    if phase == 'heldout':
        assert reference['run'] == manifest['old_run'] and reference['checkpoint_identity'] == meta['identity']
        panels = [('test', reference['panels']['task']['rows']),
                  ('wording', reference['panels']['wording']['rows']), ('ood', reference['panels']['ood']['rows'])]
    else:
        panels = [('dev', reference['task']['rows'][:32] if phase == 'smoke' else reference['task']['rows'])]
    engine = load_engine(manifest)
    import torch
    from transformers import GenerationConfig, LogitsProcessor
    torch.set_num_threads(2)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_grad_enabled(False)
    cfg = manifest['model_protocol']
    tokenizer = engine.AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'], padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = engine.load_model(cfg, two_gpu=False)
    spec = meta['spec']
    route = engine.Routing(model, spec['geometry'], spec['width'], cfg, spec['placement'])
    saved = engine.core.load_pt(meta['checkpoint'])
    engine.state_load(route, saved, meta['identity'])
    assert saved['step'] == 1024
    del saved
    model.requires_grad_(False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    before = engine.core.tensor_hash(model.named_parameters())
    node_ids = sorted(tokenizer.encode(c, add_special_tokens=False)[0] for c in engine.datasets.NODES)
    assert len(set(node_ids)) == 16 and all(len(tokenizer.encode(c, add_special_tokens=False)) == 1 for c in engine.datasets.NODES)
    ref_id = tokenizer.encode('REFUSE', add_special_tokens=False)[0]
    assert ref_id not in node_ids
    token_meta = dict(node_ids=node_ids, node_tokens=[tokenizer.decode([i]) for i in node_ids],
                      refusal_start_id=ref_id, refusal_start_text=tokenizer.decode([ref_id]))
    write(folder/'tokens.json', token_meta)
    replay_exact = False
    if phase == 'smoke':
        replay = engine.evaluate(model, tokenizer, panels[0][1], 32)['rows']
        assert all(a['response'] == b['response'] for a, b in zip(replay, panels[0][1])), 'Greedy replay differs'
        replay_exact = True

    class FirstNode(LogitsProcessor):
        def __init__(self, length):
            self.length, self.stats = length, None
        def __call__(self, ids, logits):
            if ids.shape[1] != self.length:
                return logits
            assert self.stats is None
            raw = logits.detach().float()
            ns = raw[:, node_ids]
            maximum, which = ns.max(-1)
            total, z = torch.logsumexp(ns, -1), torch.logsumexp(raw, -1)
            rr, top = raw[:, ref_id], raw.argmax(-1)
            argnodes = torch.tensor(node_ids, device=which.device)[which]
            union_refusal = (rr > maximum) | ((rr == maximum) & (ref_id < argnodes))
            assert torch.isfinite(raw).all() and torch.all(total >= maximum-1e-5)
            self.stats = dict(node_logits=ns.cpu().tolist(), ref_logit=rr.cpu().tolist(),
                margin_sum=(rr-total).cpu().tolist(), margin_max=(rr-maximum).cpu().tolist(),
                p_ref=torch.exp(rr-z).cpu().tolist(), p_nodes=torch.exp(total-z).cpu().tolist(),
                argmax_refusal=union_refusal.cpu().tolist(), node_arg_id=argnodes.cpu().tolist(),
                raw_top_id=top.cpu().tolist())
            masked = torch.full_like(logits, float('-inf'))
            masked[:, node_ids] = logits[:, node_ids]
            return masked

    gen = GenerationConfig(do_sample=False, max_new_tokens=32, eos_token_id=tokenizer.eos_token_id,
                           pad_token_id=tokenizer.pad_token_id)
    n = 0
    resume_chunks = 0
    score_parity_error = 0.
    for panel, all_rows in panels:
        for offset in range(0, len(all_rows), manifest['chunk_size']):
            group = all_rows[offset:offset+manifest['chunk_size']]
            path = folder/f'chunk_{panel}_{offset:06d}.json'
            identity = dict(run=digest(manifest), model=key, phase=phase, panel=panel,
                            offset=offset, ids=[r['id'] for r in group], checkpoint=meta['checkpoint_sha256'])
            if path.exists():
                previous = read(path)
                assert previous['identity'] == identity and len(previous['rows']) == len(group)
                n += len(group); resume_chunks += 1
                continue
            records = []
            for start in range(0, len(group), manifest['batch_size']):
                batch = group[start:start+manifest['batch_size']]
                inputs = engine.core.pack_prompts(tokenizer, batch, model.device)
                assert inputs['input_ids'].shape[1] <= 512
                processor = FirstNode(inputs['input_ids'].shape[1])
                generated = model.generate(**inputs, generation_config=gen, logits_processor=[processor])
                tails = generated[:, inputs['input_ids'].shape[1]:]
                assert processor.stats is not None
                for i, (row, tail) in enumerate(zip(batch, tails)):
                    text = tokenizer.decode(tail, skip_special_tokens=True).strip()
                    values = engine.score(row, text)
                    stat = {k: v[i] for k, v in processor.stats.items()}
                    top = stat['raw_top_id']
                    records.append(dict(id=row['id'], world=row['world'], case=row['case'], label=row['label'],
                        depth=row['depth'], split=panel, greedy_response=row['response'],
                        greedy_refusal=row['refusal'], greedy_path_correct=row['path_correct'],
                        greedy_format_valid=row['format_valid'], greedy_truncated=row['truncated'],
                        node_response=text, node_path_correct=values['path_correct'], node_format_valid=values['format_valid'],
                        node_truncated=bool(len(tail) >= 32 and tokenizer.eos_token_id not in tail.tolist()),
                        raw_top_text=tokenizer.decode([top]), raw_top_is_ref=top == ref_id,
                        raw_top_is_node=top in node_ids, **stat))
            write(path, dict(identity=identity, rows=records))
            n += len(group)
            write(folder/'progress.json', dict(model=key, phase=phase, completed=n, planned=sum(len(r) for _, r in panels)))
            print(json.dumps(dict(model=key, phase=phase, completed=n)), flush=True)
    after = engine.core.tensor_hash(model.named_parameters())
    assert before == after, 'Read-only diagnostic changed model weights'
    if phase == 'smoke':
        path = sorted(folder.glob('chunk_*.json'))[0]
        chunk = read(path)
        assert chunk['identity']['run'] == digest(manifest) and len(chunk['rows']) == 32
        for row, cached in zip(panels[0][1][:4], chunk['rows'][:4]):
            trained = engine.legacy.first_logits(model, tokenizer, row)
            measured = torch.tensor(cached['node_logits'], device=trained.device)
            delta = float((trained[node_ids]-measured).abs().max())
            delta = max(delta, abs(float(trained[ref_id])-cached['ref_logit']))
            score_parity_error = max(score_parity_error, delta)
        assert score_parity_error <= 1e-3, 'Training and generation first-token scores differ'
        resume_chunks += 1
    base = read(Path(manifest['old_root'])/'base.json')
    report_dev = reference if phase != 'heldout' else read(meta['refs']['dev']['path'])
    write(folder/'DONE.json', dict(run=digest(manifest), phase=phase, model=key, n=n,
        baseline_replay_exact=replay_exact, resume_chunks_verified=resume_chunks,
        training_generation_score_max_abs_error=score_parity_error if phase == 'smoke' else None,
        parameters_before=before, parameters_after=after, retention_ok=retention(report_dev, base, cfg),
        chunks={p.name: sha(p) for p in sorted(folder.glob('chunk_*.json'))},
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), visible_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
        started_unix=started, ended_unix=time.time(),
        calibration_sha256=sha(root/'calibration.json') if phase == 'heldout' else None))


def run(root, gpus):
    manifest = read(root/'manifest.json')
    assert manifest['sources'] == sources()
    lock = (root/'launcher.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root/'logs').mkdir(exist_ok=True)
    for phase in (['smoke'] if manifest['smoke'] else ['dev', 'heldout']):
        if phase == 'heldout':
            calibrate_all(root)
        active = {}
        def done(key):
            p = root/phase/key/'DONE.json'
            if p.exists():
                assert read(p)['run'] == digest(manifest)
                return True
            return False
        while True:
            for pid, (process, key, gpu, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close(); del active[pid]
                if code or not done(key):
                    # Finish other already-running read-only workers before exiting.
                    for p, _, _, h in active.values():p.wait(); h.close()
                    write(root/'status.json', dict(phase=phase, failed=key, exit_code=code, running=[]))
                    raise RuntimeError(f'Diagnostic worker failed: {key}, exit {code}')
            occupied = {gpu for _, _, gpu, _ in active.values()}
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            memory = {int(line.split(',')[0]): int(line.split(',')[1]) for line in raw.strip().splitlines()}
            available = [g for g in gpus if g not in occupied and memory[g] < 200]
            busy = {key for _, key, _, _ in active.values()}
            for key in manifest['models']:
                if done(key) or key in busy or not available:
                    continue
                gpu = available.pop(0)
                handle = (root/'logs'/f'{phase}_{key}.log').open('a')
                command = [sys.executable, '-u', str(HERE/'diagnostic.py'), 'worker', '--root', str(root), '--phase', phase, '--model', key]
                process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                active[process.pid] = (process, key, gpu, handle)
            write(root/'status.json', dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                phase=phase, complete=sum(done(k) for k in manifest['models']), planned=len(manifest['models']),
                running=[dict(pid=pid, model=k, gpu=g) for pid, (_, k, g, _) in active.items()], failures=[]))
            if all(done(k) for k in manifest['models']) and not active:
                break
            time.sleep(5)
    write(root/'status.json', dict(phase='complete', complete=6, running=[], failures=[],
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['initialize', 'run', 'worker', 'status'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--old-root', type=Path)
    p.add_argument('--hardware', type=Path)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--gpus', default='0,1,2,3,4,5')
    p.add_argument('--model')
    p.add_argument('--phase', choices=['smoke', 'dev', 'heldout'])
    a = p.parse_args(); root = a.root.resolve()
    if a.command == 'initialize':
        initialize(root, a.old_root.resolve(), a.smoke, a.hardware.resolve() if a.hardware else None)
    elif a.command == 'run':
        run(root, [int(g) for g in a.gpus.split(',')])
    elif a.command == 'worker':
        worker(root, a.phase, a.model)
    else:
        print(json.dumps(read(root/'status.json'), indent=2))


if __name__ == '__main__':main()
