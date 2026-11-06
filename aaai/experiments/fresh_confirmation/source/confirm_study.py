"""Fresh-bank replication, using the unchanged independent and continuation engines."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run_aware as aware
previous=aware.previous
atomic,digest,hash_file,exclusive,checked_space=aware.atomic,aware.digest,aware.hash_file,aware.exclusive,aware.checked_space


def protocol():return json.loads((ROOT/'protocol.json').read_text())


def bank_name(job):return f"bank_{job['support']}_small_s{job['seed']}.pt"


def make_plan(smoke=False):
    p=protocol();seeds=[31] if smoke else p['seeds'];jobs=[]
    for seed in seeds:
        for support,width in p['supports'].items():
            common=dict(support=support,budget='small',width=width,seed=seed,smoke=smoke)
            owners=[]
            for task in ([7] if smoke else range(8)):
                key=f'independent_{support}_s{seed}_t{task}';owners.append(key)
                jobs.append(dict(common,id=key,kind='independent',task=task,steps=2 if smoke else p['independent_steps'],lr=p['independent_lr'][support]))
            bank=f'bank_{support}_s{seed}'
            if not smoke:jobs.append(dict(common,id=bank,kind='bank',dependencies=owners,steps=0))
            for mode in p['modes']:
                key=f'train_{support}_{mode}_s{seed}'
                jobs.append(dict(common,id=key,kind='main',mode=mode,steps=2 if smoke else p['continuation_steps'],lr=p['continuation_lr'][support],dependencies=[] if smoke else [bank]))
            for mode in (['foreign_conserve'] if smoke else ['initial']+p['modes']):
                jobs.append(dict(common,id=f'eval_{support}_{mode}_s{seed}',kind='merge',mode=mode,
                    steps=2 if smoke else (320 if mode=='initial' else 640),lr=p['continuation_lr'][support],
                    bank_checkpoint=None if mode=='initial' else f'train_{support}_{mode}_s{seed}'))
    # The test set is evaluated only after ALL production training is finished.
    barrier=[j['id'] for j in jobs if j['kind']!='merge']
    for j in jobs:
        if j['kind']=='merge':j['dependencies']=barrier
    files=[*ROOT.glob('*.py'),*ROOT.glob('reference/*.py'),*ROOT.glob('reference/src/*.py'),
           ROOT/'protocol.json',ROOT/'data_audit.json',ROOT/'neutral_manifest.json',ROOT/'engine_lineage.json',ROOT/'launch.sh',
           ROOT/'reference/dataset.json',ROOT/'reference/original_dataset.json',ROOT/'reference/benchmarks.json',ROOT/'reference/requirements.txt']
    return dict(study=p['study'],smoke=smoke,protocol=p,protocol_sha256=digest(p),model=p['model'],revision=p['revision'],
                neutral_sha256=json.loads((ROOT/'neutral_manifest.json').read_text())['sha256'],
                source_hashes={str(f.relative_to(ROOT)):hash_file(f) for f in sorted(files)},jobs=jobs)


def execution_revision(output,plan):return dict(id=digest(plan['source_hashes']))


def runtime_plan(job,plan):
    path=ROOT.parent/'inputs'/bank_name(job)
    return dict(plan,scientific_plan_sha256=digest(plan),
                inputs=dict(banks={path.name:hash_file(path)},neutral_sha256=plan['neutral_sha256']))


def independent_spec(job):
    return dict(job,kind='smoke' if job['smoke'] else 'main')


def configure_runtime(smoke):
    # Only the smoke branch changes data: held-out examples are replaced by train examples.
    def init_runtime():
        import multi_run
        import nr5_rl as rl
        import nr6_train as train
        import multi_tasks as tasks
        import compound_tasks
        multi_run.setup()
        data=json.loads((ROOT/'reference/dataset.json').read_text())
        if smoke:
            sample=[e for t in range(8) for e in [x for x in data['train'] if x['task_id']==t][:3]]
            data={**data,**{s:sample for s in ('train','dev','test','depth_probe')}}
        compound=compound_tasks.build(data)
        def format_prompt(e):
            return compound_tasks.prompt(e,compound['demos']) if 'family' in e else tasks.format_prompt(e)
        rl.format_prompt=train.format_prompt=format_prompt
        train.RECIPES['independent_routed']=('routed','all_ab')
        return data,compound
    previous.init_runtime=init_runtime
    previous.execution_revision=execution_revision


def validate_completed(output,job,plan):
    folder=output/'fits'/job['id'];path=folder/'completed.json'
    if not path.exists():return False
    r=json.loads(path.read_text())
    if job['kind']=='independent':
        assert r['spec']==independent_spec(job) and r['plan_sha256']==digest(plan)
        assert r['final_step']==job['steps'] and r['unused_slots_unchanged'] and r['all_off_equal']
        assert hash_file(folder/'owner_final.pt')==r['owner_sha256']
    elif job['kind']=='bank':
        assert r['spec']==job and r['plan_sha256']==digest(plan)
        assert r['qualified'] and r['minimum_correct_replays']>=8
        assert hash_file(ROOT.parent/'inputs'/bank_name(job))==r['bank_sha256']
    elif job['kind']=='main':
        assert r['spec']==job and r['plan_sha256']==digest(runtime_plan(job,plan))
        assert r['final_step']==job['steps'] and r['status']['complete'] and r['all_off_equal']
        assert hash_file(folder/'bank_final.pt')==r['bank_sha256']
    elif job['kind']=='merge':
        assert r['spec']==job and r['plan_sha256']==digest(plan) and r['complete']
        for name,expected in r['artifact_hashes'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert hash_file(folder/name)==expected
    else:raise ValueError(job['kind'])
    return True


def build_bank(job,plan,output):
    import torch
    from nr5_rl import atomic_torch
    data=json.loads((ROOT/'reference/dataset.json').read_text())
    train_ids={e['id']:e for e in data['train']}
    state={};optimizers=[];replays={};owners={};manifest=None
    lookup={j['id']:j for j in plan['jobs']}
    for task,key in enumerate(job['dependencies']):
        assert validate_completed(output,lookup[key],plan)
        folder=output/'fits'/key
        owner=torch.load(folder/'owner_final.pt',map_location='cpu',weights_only=True)
        saved=torch.load(folder/'durable/fit/continuation.pt',map_location='cpu',weights_only=True)['latest']
        if manifest is None:manifest=owner['manifest']
        assert owner['manifest']==manifest and owner['step']==saved['step']==320 and not saved['pending_dev']
        assert not set(state)&set(owner['deltas'])
        assert all(torch.equal(v,saved['deltas'][k]) for k,v in owner['deltas'].items())
        state.update(owner['deltas']);optimizers.append(saved['optimizers'][task]);selected={}
        for event in reversed(saved['history']):
            path=folder/'durable/fit'/event['rollout_file'];assert hash_file(path)==event['rollout_sha256']
            with gzip.open(path,'rt') as f:record=json.load(f)
            assert record['task_id']==task
            for item in record['items']:
                e=item['example'];assert e['id'] in train_ids and e==train_ids[e['id']]
                if item['valid'] and item['correct'] and item['answer']==e['answer']:
                    selected.setdefault(e['id'],dict(item,replay_source_step=event['step'],replay_source_sha256=event['rollout_sha256']))
        if len(selected)<8:
            atomic(output/'fits'/job['id']/'acquisition_failed.json',dict(task=task,correct_replays=len(selected),required=8,no_seed_replacement=True))
            raise RuntimeError(f'Insufficient verified-correct train replay for {key}: {len(selected)}. Do not substitute a seed or gold answers.')
        replays[task]=[selected[k] for k in sorted(selected)]
        owners[key]=dict(sha256=hash_file(folder/'owner_final.pt'),correct_replay_examples=len(selected),train_examples=240)
    payload=dict(manifest=manifest,deltas=state,optimizers=optimizers,replay=replays,source_owners=owners,
                 source_plan_sha256=digest(plan),seed=job['seed'],support=job['support'],budget='small',width=job['width'],source_step_per_task=320)
    target=ROOT.parent/'inputs'/bank_name(job);atomic_torch(target,payload)
    atomic(output/'fits'/job['id']/'completed.json',dict(spec=job,plan_sha256=digest(plan),qualified=True,
        bank_sha256=hash_file(target),minimum_correct_replays=min(map(len,replays.values())),source_owners=owners))
    print('BANK READY',target.name,[len(v) for v in replays.values()],flush=True)


def evaluate(job,plan,output):
    import torch
    import nr5_rl as rl
    import retention
    from multi_run import witnesses
    data,_=previous.init_runtime()
    lookup={j['id']:j for j in plan['jobs']}
    assert all(validate_completed(output,lookup[k],plan) for k in job['dependencies']), 'Global training barrier not satisfied'
    folder=output/'fits'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    module,tok=previous.build_module(job);cfg=previous.config(dict(job,kind='smoke' if job['smoke'] else 'main'),plan)
    try:
        examples=[next(e for e in data['train'] if e['task_id']==t) for t in range(8)]
        with module.gate((False,)*8):before=witnesses(module.model,tok,examples)
        if job['bank_checkpoint']:
            path=output/'fits'/job['bank_checkpoint']/'bank_final.pt'
            payload=rl._load(path);assert payload['plan_sha256']==digest(runtime_plan(job,plan))
        else:
            path=ROOT.parent/'inputs'/bank_name(job);payload=rl._load(path)
            assert payload['source_plan_sha256']==digest(plan)
        assert payload['manifest']==module.manifest();module.load_state_dict(payload['deltas'])
        module.independent_task=None
        with module.gate((False,)*8):after=witnesses(module.model,tok,examples)
        assert all(torch.equal(a,b) for a,b in zip(before,after))
        scores=[]
        for split in ('test','depth_probe'):
            for task in range(8):
                examples=[e for e in data[split] if e['task_id']==task]
                if job['smoke']:examples=examples[:1]
                for mask,indices in [('base',()),('owner',(task,)),('all',tuple(range(8))),('without_owner',tuple(t for t in range(8) if t!=task))]:
                    if split=='depth_probe' and mask=='without_owner':continue
                    rows=previous.predictions(module,tok,examples,indices,cfg,folder/'evaluation'/f'{split}_t{task}_{mask}.json',split)
                    scores.append(dict(split=split,task=task,mask=mask,n=len(rows),correct=sum(r['correct'] for r in rows),accuracy=sum(r['correct'] for r in rows)/len(rows)))
        benchmarks=json.loads((ROOT/'reference/benchmarks.json').read_text())
        if job['smoke']:
            benchmarks=dict(benchmarks,examples=[next(e for e in benchmarks['examples'] if e['benchmark']==name)
                for name in benchmarks['protocol']['primary_metrics']],wiki_chunks=benchmarks['wiki_chunks'][:1])
        from nr6_io import runtime_info
        base=retention.evaluate(module,benchmarks,folder/'retention/base.json',(False,)*8,runtime_info())
        all_on=retention.evaluate(module,benchmarks,folder/'retention/all.json',(True,)*8,runtime_info())
        atomic(folder/'summary.json',dict(scores=scores,retention=retention.changes(base,all_on,benchmarks)))
        assert module.assert_frozen_base()
        atomic(folder/'completed.json',dict(spec=job,plan_sha256=digest(plan),source_bank_sha256=hash_file(path),complete=True,all_off_equal=True,
            artifact_hashes={str(p.relative_to(folder)):hash_file(p) for p in sorted(folder.rglob('*.json')) if p.name!='completed.json'}))
    finally:module.close()


def worker(job,plan,output):
    configure_runtime(plan['smoke'])
    if job['kind']=='independent':previous.train_fit(independent_spec(job),plan,output)
    elif job['kind']=='bank':build_bank(job,plan,output)
    elif job['kind']=='main':
        if not job['smoke']:
            import nr5_rl as rl
            bank=rl._load(ROOT.parent/'inputs'/bank_name(job))
            assert bank['source_plan_sha256']==digest(plan) and bank['source_step_per_task']==320
            del bank
        aware.train_bank(job,runtime_plan(job,plan),output)
    elif job['kind']=='merge':evaluate(job,plan,output)
    else:raise ValueError(job['kind'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','smoke','worker','status'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--job');parser.add_argument('--gpus',default='0,1,2,3,4,5')
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='status':print((output/'shard_status.json').read_text());return
    output.mkdir(parents=True,exist_ok=True)
    if args.action=='worker':
        plan=json.loads((output/'experiment_plan.json').read_text())
        assert plan==make_plan(plan['smoke']), 'Pinned scientific code/data changed'
        job=next(j for j in plan['jobs'] if j['id']==args.job)
        try:
            with exclusive(output/'fits'/job['id']/'worker.lock'):worker(job,plan,output)
        except BlockingIOError:raise SystemExit(75)
        return
    with exclusive(output/'launcher.lock'):
        plan=make_plan(args.action=='smoke');target=output/'experiment_plan.json'
        if target.exists():assert json.loads(target.read_text())==plan, 'Choose a new output for a changed protocol'
        else:atomic(target,plan)
        if args.action=='run':
            smoke=json.loads((output.parent/'smoke/SHARD_COMPLETE.json').read_text())
            assert smoke['completed']==10 and not smoke['failed']
            assert smoke['execution_revision']==execution_revision(output,plan)['id']
        devices=args.gpus.split(',');assert devices==list(map(str,range(6)))
        def terminate(signum,frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,terminate)
        import scheduler
        scheduler.supervise(sys.modules[__name__],plan,output,[j['id'] for j in plan['jobs']],devices)


if __name__=='__main__':main()
