"""Freeze the previous specialists, Adam states and train-only replay evidence."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reference'))
sys.path.insert(0, str(ROOT / 'reference/src'))
import torch
import run_study as previous

OLD = Path('/var/tmp/casr-mn866-composition/outputs/composition01_repaired')
WIKI_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def build_banks(target, seed):
    data = json.loads((ROOT / 'reference/dataset.json').read_text())
    train_ids = {e['id']: e for e in data['train']}
    forbidden = {e['id'] for split in ('dev', 'test', 'depth_probe') for e in data[split]}
    assert not set(train_ids) & forbidden
    plan = json.loads((OLD / 'experiment_plan.json').read_text())
    result = []
    for support, budget, width in previous.CELLS:
        destination = target / f'bank_{support}_{budget}_s{seed}.pt'
        state, optimizers, banks, owners, manifest = {}, [], {}, {}, None
        for task in range(8):
            key = f'main_{support}_{budget}_s{seed}_t{task}'
            spec = next(j for j in plan['jobs'] if j['id'] == key)
            assert previous.validate_completed(OLD, spec, plan)
            folder = OLD / 'fits' / key
            owner = torch.load(folder / 'owner_final.pt', map_location='cpu', weights_only=True)
            if manifest is None: manifest = owner['manifest']
            assert owner['manifest'] == manifest and owner['step'] == 320
            assert not set(state) & set(owner['deltas'])
            state.update(owner['deltas'])
            saved = torch.load(folder / 'durable/fit/continuation.pt', map_location='cpu', weights_only=True)['latest']
            assert saved['step'] == 320 and not saved['pending_dev']
            assert all(torch.equal(v, saved['deltas'][k]) for k, v in owner['deltas'].items())
            optimizers.append(saved['optimizers'][task])
            selected = {}
            for event in reversed(saved['history']):
                path = folder / 'durable/fit' / event['rollout_file']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == event['rollout_sha256']
                with gzip.open(path, 'rt') as stream: record = json.load(stream)
                assert record['task_id'] == task and record['step'] == event['step']
                for item in record['items']:
                    ex = item['example']
                    assert ex['id'] in train_ids and ex == train_ids[ex['id']]
                    if item['correct'] and item['valid'] and item['answer'] == ex['answer'] and ex['id'] not in selected:
                        selected[ex['id']] = dict(item, replay_source_step=event['step'],
                                                replay_source_sha256=event['rollout_sha256'])
            assert len(selected) >= 8 and not set(selected) & forbidden
            banks[task] = [selected[key] for key in sorted(selected)]
            owners[key] = dict(sha256=previous.hash_file(folder/'owner_final.pt'),
                               correct_replay_examples=len(selected), train_examples=240)
            del saved, owner
        payload = dict(manifest=manifest, deltas=state, optimizers=optimizers, replay=banks,
                       source_owners=owners, source_plan_sha256=previous.digest(plan), seed=seed,
                       support=support, budget=budget, width=width, source_step_per_task=320)
        from nr5_rl import atomic_torch
        atomic_torch(destination, payload)
        result.append(dict(path=destination.name, sha256=previous.hash_file(destination), owners=owners))
        print('BANK READY', destination.name, [len(v) for v in banks.values()], flush=True)
    previous.atomic(target / 'banks_manifest.json', dict(seed=seed, banks=result))


def build_neutral(target):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    benchmark = json.loads((ROOT / 'reference/benchmarks.json').read_text())
    sequences = list(benchmark['wiki_chunks']) + [e['context_ids'] + choice for e in benchmark['examples'] for choice in e['choice_ids']]
    blocked = {tuple(ids[i:i+16]) for ids in sequences for i in range(max(0,len(ids)-15))}
    tokenizer = AutoTokenizer.from_pretrained(previous.MODEL, revision=previous.REVISION,
                                               local_files_only=True, trust_remote_code=False)
    path = hf_hub_download('Salesforce/wikitext', 'wikitext-2-raw-v1/train-00000-of-00001.parquet',
                           repo_type='dataset', revision=WIKI_REVISION)
    rows, rejected = [], 0
    for text in pq.read_table(path, columns=['text'])['text'].to_pylist():
        ids = tokenizer(text, add_special_tokens=False).input_ids
        for start in range(0, len(ids)-128, 129):
            chunk = ids[start:start+129]
            if any(tuple(chunk[i:i+16]) in blocked for i in range(114)):
                rejected += 1
                continue
            rows.append(dict(id=f'wiki_train_{len(rows):04d}', prompt_ids=chunk[:64],
                             generated_ids=chunk[64:]))
            if len(rows) == 1024: break
        if len(rows) == 1024: break
    assert len(rows) == 1024
    previous.atomic(target / 'neutral.json', dict(rows=rows, rejected_overlap_windows=rejected,
        source='Salesforce/wikitext', revision=WIKI_REVISION, split='wikitext-2-raw-v1/train',
        parquet_sha256=previous.hash_file(path), exclusion='No shared contiguous 16-token span with frozen retention inputs.'))
    print('NEUTRAL BANK READY', len(rows), 'overlap windows excluded', rejected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--neutral', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    build_banks(args.output, {'ogg':31, 'garlick':47, 'weatherwax':71}[socket.gethostname()])
    if args.neutral: build_neutral(args.output)
