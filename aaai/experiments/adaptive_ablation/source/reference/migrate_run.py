#!/usr/bin/env python3
"""Copy a stopped original run and migrate signatures without changing training state."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import run_study as study


def convert(value, previous, current):
    """Change only the trainer fingerprint inside current resume signatures."""
    result=copy.deepcopy(value)
    def visit(item):
        if isinstance(item,dict):
            if 'trainer_sha256' in item:
                if item['trainer_sha256'] != previous: raise ValueError('Unexpected trainer fingerprint')
                item['trainer_sha256']=current
            for child in item.values(): visit(child)
        elif isinstance(item,(list,tuple)):
            for child in item: visit(child)
    visit(result)
    return result


def migrate(source,target,old_source):
    import torch
    from nr5_rl import _load,atomic_torch
    from nr6_checks import equal
    if target.exists() or source.resolve()==target.resolve(): raise ValueError('Destination must be a fresh directory')
    # A stopped launcher must not leave children writing while the copy is made.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try: command=(entry/'cmdline').read_bytes().split(b'\0')
        except (OSError,PermissionError): continue
        if b'--output' in command and str(source.resolve()).encode() in command and any(b'run_study.py' in arg for arg in command):
            raise RuntimeError(f'Original run still has live process {entry.name}')
    plan=json.loads((source/'experiment_plan.json').read_text()); current=study.make_plan()
    assert {k:v for k,v in plan.items() if k!='source_hashes'}=={k:v for k,v in current.items() if k!='source_hashes'}
    assert all(study.hash_file(old_source/name)==value for name,value in plan['source_hashes'].items())
    for job in plan['jobs']: study.validate_completed(source,job,plan)
    study.checked_space(source)
    print('Copying original outputs to',target,flush=True)
    shutil.copytree(source,target)
    previous=plan['source_hashes']['src/nr5_rl.py']; latest=study.hash_file(ROOT/'src/nr5_rl.py')
    converted=[]; resumed=[]
    for job in plan['jobs']:
        folder=target/'fits'/job['id']
        if (folder/'completed.json').exists() or not folder.exists(): continue
        for path in sorted(folder.glob('*/fit/*.pt')):
            if path.name not in ('continuation.pt','latest.pt','best.pt'): continue
            before=_load(path); after=convert(before,previous,latest)
            assert equal(before,convert(after,latest,previous)), 'Weights, optimizer/RNG state or history changed'
            old_hash=study.hash_file(path); atomic_torch(path,after)
            reloaded=_load(path); assert equal(after,reloaded)
            converted.append(dict(path=str(path.relative_to(target)),old_sha256=old_hash,new_sha256=study.hash_file(path),training_state_identical=True))
        for path in folder.glob('*/fit/config.json'):
            before=json.loads(path.read_text()); study.atomic(path,convert(before,previous,latest))
        resume=folder/'durable/fit/resume.json'
        if resume.exists(): resumed.append(dict(id=job['id'],step=json.loads(resume.read_text())['step']))
    revision=dict(scientific_plan_sha256=study.digest(plan),source_hashes=current['source_hashes'],
        previous_source_hashes=plan['source_hashes'],original_source=str(old_source),original_output=str(source),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        changes=['per-fit locks and signal cleanup','validated completion records','validate-before-copy checkpoint recovery',
                 'shared ASCII/Unicode arrow acceptance','equivalent owner-only forward gates and deduplicated development predictions'],
        checkpoint_migration=converted, resumed_fits=resumed,
        timing_policy='History without execution_revision is original code. New rows carry revision ID; use timing_by_revision rather than pooling implementations.')
    revision['id']=study.digest(revision)
    study.atomic(target/'execution_revision.json',revision)
    for record in resumed:
        study.atomic(target/'fits'/record['id']/'repair_resume.json',dict(execution_revision=revision['id'],**record,optimizer_rng_weights_preserved=True))
    if (target/'status.json').exists():
        shutil.copy2(target/'status.json',target/'status_before_repair.json')
        status=json.loads((target/'status.json').read_text()); status.update(active=[],launcher_state='ready_to_resume',execution_revision=revision['id'])
        study.atomic(target/'status.json',status)
    for job in plan['jobs']: study.validate_completed(target,job,plan)
    print(json.dumps(dict(revision=revision['id'],converted_files=len(converted),resumed=resumed)),flush=True)
    return revision


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--old-source',type=Path,required=True); args=parser.parse_args()
    migrate(args.source.resolve(),args.target.resolve(),args.old_source.resolve())
