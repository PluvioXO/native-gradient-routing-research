"""Coordinated continuation of eight frozen-checkpoint specialists on three hosts."""
import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'reference'), str(ROOT/'reference/src')]
import independent_engine as previous

atomic, digest, hash_file = previous.atomic, previous.digest, previous.hash_file
exclusive, checked_space = previous.exclusive, previous.checked_space
MODES = ('self_only', 'foreign_only', 'self_conserve', 'foreign_conserve')
HOST_SEEDS = {'ogg':31, 'garlick':47, 'weatherwax':71}


def make_plan(smoke=False):
    inputs = json.loads((ROOT/'inputs_manifest.json').read_text())
    files = [*ROOT.glob('*.py'), *ROOT.glob('tests/*.py'), *ROOT.glob('reference/*.py'),
             *ROOT.glob('reference/src/*.py'), ROOT/'reference/dataset.json', ROOT/'reference/benchmarks.json',
             ROOT/'reference/requirements.txt', ROOT/'controls_manifest.json']
    jobs = []
    for seed in ((31,) if smoke else (31,47,71)):
        for support,budget,width in previous.CELLS:
            if budget != 'small':
                continue
            for mode in MODES:
                key=f'{support}_{budget}_{mode}_s{seed}'
                steps=2 if smoke else 640
                jobs.append(dict(id='train_'+key,kind='main',support=support,budget=budget,width=width,
                    seed=seed,mode=mode,steps=steps,smoke=smoke,
                    lr=.3*({'native':1e-4,'lora':3e-4}[support])))
                if not smoke:
                    jobs.append(dict(id='eval_'+key,kind='merge',support=support,budget=budget,width=width,
                        seed=seed,mode=mode,steps=steps,lr=.3*({'native':1e-4,'lora':3e-4}[support]),
                        bank_checkpoint='train_'+key,dependencies=['train_'+key]))
    jobs.sort(key=lambda j:j['kind']!='main')
    return dict(study='small-budget-self-foreign-conservation-factorial-extension-v1',model=previous.MODEL,revision=previous.REVISION,
        smoke=smoke,updates_per_task=80,initial_updates_per_task=320,precision='float32',
        controls=json.loads((ROOT/'controls_manifest.json').read_text()),
        modes=list(MODES),skills=8,main_endpoint='Final additional step 640; no test selection.',
        reward='Correctness RLOO, 4 prompts x 4 samples. Owner-only gradients. All eight enabled for aware/preserve; only owner for independent.',
        preservation='Frozen initial owner teacher -> current owner-alone student on one verified-correct train replay; frozen initial foreign teacher -> current all-eight student on one rotating foreign-task replay. Mean of these KL terms.',
        conservation='Frozen original model -> current all-eight student on one WikiText training window per update. Original reward-prefix reference KL .02 also retained.',
        loss='RLOO + .02 student||original prefix KL + preservation*(.5 self KL + .5 foreign KL) + conservation*neutral KL. New (self,foreign,neutral) weights: self_only=(.5,0,0), foreign_only=(0,.5,0), self_conserve=(.5,0,1), foreign_conserve=(0,.5,1). Together with archived controls this completes a three-factor factorial. Each term keeps its original coefficient; total auxiliary weight is intentionally not renormalized. All eight skills active in every arm. Same three auxiliary passes, including zero-weight controls.',
        guards='Reward-prefix old||new mean <= .02, max sequence <= .1; 3 backtracks; clip 1. Auxiliary drift measured but never controls acceptance.',
        compute='Same reward responses, proposed updates and three auxiliary teacher/student/backward passes in every arm. Native/LoRA work and generated lengths differ; report actual time, tokens and memory, no exact FLOP match.',
        optimizer='Carry each specialist owner Adam state into an eight-owner bank; one owner optimizer per update, summed objective. New deterministic joint-stage RNG; exact checkpoint RNG/Adam continuation within this stage.',
        lr='Fixed 0.3 x independently selected LR for every mode; no test tuning. Anchor coefficient fixed 1, not tuned.',
        evaluation='Reuse previous fixed 1/2/4/8 masks, all 28 pairs and eight cyclic quadruples, retention, compounds, and secondary coefficient calibration. Reused banks: exploratory.',
        inputs=inputs,source_hashes={str(p.relative_to(ROOT)):hash_file(p) for p in sorted(files)},jobs=jobs)


def execution_revision(output,plan):
    return dict(id=digest(plan['source_hashes']))


def validate_completed(output,job,plan):
    p=output/'fits'/job['id']/'completed.json'
    if not p.exists(): return False
    record=json.loads(p.read_text())
    assert record['spec']==job and record['plan_sha256']==digest(plan),job['id']
    if job['kind']=='main':
        assert record['final_step']==job['steps'] and record['status']['complete']
        assert hash_file(p.parent/'bank_final.pt')==record['bank_sha256']
    else:
        assert record['artifact_hashes']
        for name,sha in record['artifact_hashes'].items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert hash_file(p.parent/name)==sha
    return True


def auxiliary_items(bank,neutral,seed,step,task):
    rng=random.Random(f'aware-aux-v1:{seed}:{step}')
    iteration=(step-1)//8
    foreign=[k for k in range(8) if k!=task][iteration%7]
    own=dict(rng.choice(bank['replay'][task]),kind='self',teacher_task=task)
    other=dict(rng.choice(bank['replay'][foreign]),kind='foreign',teacher_task=foreign)
    general=dict(rng.choice(neutral['rows']),kind='neutral',teacher_task=None)
    return [own,other,general]


def train_bank(spec,plan,output,stop_after=None):
    import numpy as np
    import torch
    import nr5_rl as rl
    from multi_run import Config, witnesses
    from nr6_io import Workspace
    from aware_update import update
    data,_=previous.init_runtime()
    bank_name=f"bank_{spec['support']}_{spec['budget']}_s{spec['seed']}.pt"
    path=ROOT.parent/'inputs'/bank_name
    assert hash_file(path)==plan['inputs']['banks'][bank_name]
    neutral_path=ROOT.parent/'inputs/neutral.json'
    assert hash_file(neutral_path)==plan['inputs']['neutral_sha256']
    bank=rl._load(path)
    neutral=json.loads(neutral_path.read_text())
    module,tok=previous.build_module(spec)
    assert module.manifest()==bank['manifest']
    module.load_state_dict(bank['deltas'])
    module.independent_task=None
    initial={name:value.to(module.device) for name,value in bank['deltas'].items()}
    @dataclass
    class AwareConfig(Config):
        mode:str='preserve'
        anchor_weight:float=1.
    cfg=AwareConfig(max_steps=spec['steps'],lr=spec['lr'],seed=spec['seed'],arm_name=spec['id'],task_count=8,
        mode=spec['mode'],anchor_weight=1.,prompts_per_update=1 if spec['smoke'] else 4,
        group_size=2 if spec['smoke'] else 4,generation_batch=2,eval_batch=8,
        max_new_tokens=8 if spec['smoke'] else 96,max_prompt_tokens=1536,
        max_mean_kl=10. if spec['smoke'] else .02,max_sequence_kl=10. if spec['smoke'] else .1,
        save_every=1 if spec['smoke'] else 16,eval_every=640 if spec['smoke'] else 320,
        snapshot_every=80,head_chunk=8,kl_head_chunk=8,preservation_mode='all_ab',audit_steps=[],
        cluster_plan_sha256=digest(plan)).validate()
    identity=dict(plan_sha256=digest(plan),spec=spec,config=asdict(cfg),initial_bank=plan['inputs']['banks'][bank_name])
    ws=Workspace(output/'fits'/spec['id']/'local',output/'fits'/spec['id']/'durable',identity)
    ws.restore_training('fit',cfg)
    folder=ws.path('fit')
    signature=dict(identity=identity,manifest=module.manifest(),base_hash=bank['manifest']['base_hash'])
    optimizers=[torch.optim.AdamW(module.parameters(k),lr=cfg.lr,weight_decay=0.) for k in range(8)]
    for optimizer,state in zip(optimizers,bank['optimizers']): optimizer.load_state_dict(state)
    rng=random.Random(spec['seed']+20261022)
    random.seed(spec['seed']+20261022);np.random.seed(spec['seed']+20261022)
    torch.manual_seed(spec['seed']+20261022);torch.cuda.manual_seed_all(spec['seed']+20261022)
    history,dev_history=[],[]
    step,best_step,best_key,pending_dev=0,0,None,not spec['smoke']
    if (folder/'latest.pt').exists():
        state=rl._load(folder/'latest.pt');assert state['signature']==signature
        module.load_state_dict(state['deltas'])
        for opt,value in zip(optimizers,state['optimizers']):opt.load_state_dict(value)
        rl._restore_rng(state['rng'],rng)
        from nr6_checks import equal
        assert equal(rl._rng_state(rng),state['rng']), 'RNG state was not restored exactly'
        assert equal(rl._cpu([o.state_dict() for o in optimizers]),state['optimizers']), 'Adam state was not restored exactly'
        assert equal(rl._cpu(module.state_dict()),state['deltas']), 'Weights were not restored exactly'
        step,best_step,best_key,pending_dev=(state[k] for k in ['step','best_step','best_key','pending_dev'])
        history,dev_history=state['history'],state['dev_history']
        print('RESUMED',spec['id'],step,flush=True)
    before_witness=None
    if spec['smoke']:
        with module.gate((False,)*8): before_witness=witnesses(module.model,tok,data['dev'][:1])
    else:
        witness_examples=[next(e for e in data['dev'] if e['task_id']==t) for t in range(8)]
        with module.gate((False,)*8): before_witness=witnesses(module.model,tok,witness_examples)
    retained=set()
    def save():
        checked_space(output)
        rl.atomic_torch(folder/'latest.pt',dict(signature=signature,deltas=module.state_dict(),
            optimizers=[o.state_dict() for o in optimizers],rng=rl._rng_state(rng),step=step,
            best_step=best_step,best_key=best_key,pending_dev=pending_dev,history=history,dev_history=dev_history))
        atomic(folder/'config.json',signature)
        (folder/'history.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in history))
        atomic(folder/'dev_history.json',dev_history)
        ws.save_training('fit',cfg)
        for name,pattern in [('rollouts','*.json.gz'),('snapshots','*.pt')]:
            for p in (folder/name).glob(pattern):
                rel=str(p.relative_to(ws.local))
                if rel not in retained:ws.publish(rel,evidence=False);retained.add(rel)
    started=time.monotonic()
    try:
        save()
        while True:
            if pending_dev:
                rows=previous.predictions(module,tok,data['dev'],tuple(range(8)),cfg,
                    folder/'dev_predictions'/f'step_{step:05d}.json','dev')
                accuracy=sum(r['correct'] for r in rows)/len(rows)
                if best_key is None or accuracy>best_key[0]:
                    best_step,best_key=step,[accuracy]
                    rl.atomic_torch(folder/'best.pt',dict(signature=signature,deltas=module.state_dict(),step=step))
                dev_history=[r for r in dev_history if r['step']!=step]+[dict(step=step,all_eight_accuracy=accuracy)]
                pending_dev=False;save()
                print('DEV',spec['id'],step,accuracy,flush=True)
            if step==cfg.max_steps:break
            tick=time.monotonic()
            task=(0 if step==0 else 7) if spec['smoke'] else step%8
            train=[e for e in data['train'] if e['task_id']==task]
            prompts=rl._sample_prompts(train,step//8,cfg,rng)
            expanded=[e for e in prompts for _ in range(cfg.group_size)]
            items=[]
            mask=f'slot_{task}' if cfg.mode=='independent' else 'all'
            for start in range(0,len(expanded),cfg.generation_batch):
                items.extend(rl._generate(module,tok,expanded[start:start+cfg.generation_batch],cfg,mask,sample=True))
            rewards=np.asarray([float(i['correct']) for i in items]).reshape(-1,cfg.group_size)
            advantages=rewards-(rewards.sum(1,keepdims=True)-rewards)/(cfg.group_size-1)
            if spec['smoke']:advantages=np.asarray([[1.,-1.]])
            aux=auxiliary_items(bank,neutral,spec['seed'],step+1,task)
            prior_steps=[float(next(iter(o.state.values()))['step']) for o in optimizers]
            metrics=update(module,tok,items,advantages.reshape(-1),cfg,optimizers,task,initial,aux)
            after_steps=[float(next(iter(o.state.values()))['step']) for o in optimizers]
            assert all(a==b for k,(a,b) in enumerate(zip(prior_steps,after_steps)) if k!=task)
            assert after_steps[task]-prior_steps[task] in (0,1)
            step+=1
            record=dict(step=step,task_id=task,reward=float(rewards.mean()),seconds=time.monotonic()-tick,
                response_tokens=sum(len(i['generated_ids']) for i in items),auxiliary_tokens=sum(len(i['generated_ids']) for i in aux),
                foreign_task=aux[1]['teacher_task'],foreign_optimizers_unchanged=True,**metrics)
            history.append(record)
            import uuid
            rollout=folder/'rollouts'/f'{step:05d}-{uuid.uuid4().hex}.json.gz';rollout.parent.mkdir(exist_ok=True)
            temp=rollout.with_suffix('.partial')
            with gzip.open(temp,'wt') as stream:json.dump(dict(metrics=record,items=items,auxiliary=aux,
                rewards=rewards.tolist(),advantages=advantages.tolist()),stream,allow_nan=False)
            temp.replace(rollout);record['rollout_file']=str(rollout.relative_to(folder));record['rollout_sha256']=hash_file(rollout)
            if step%cfg.snapshot_every==0:
                rl.atomic_torch(folder/'snapshots'/f'step_{step:05d}.pt',dict(step=step,deltas=module.state_dict()))
            pending_dev=not spec['smoke'] and (step%cfg.eval_every==0 or step==cfg.max_steps)
            if step%cfg.save_every==0 or pending_dev or step==cfg.max_steps:
                save();print('STEP',spec['id'],step,'reward',record['reward'],'accepted',metrics['accepted'],flush=True)
            if stop_after is not None and step==stop_after:raise InterruptedError('Injected resume check')
            smoke_marker=output/'fits'/spec['id']/'smoke_resume_interruption.json'
            if spec['smoke'] and step==1 and not smoke_marker.exists():
                atomic(smoke_marker,dict(step=step,reason='Intentional subprocess resume test after a committed update'))
                raise InterruptedError('Intentional smoke-only checkpoint/resume test; launcher retries')
        if spec['smoke']:
            from torch.nn.utils.parametrize import cached
            longest=max(data['train'],key=lambda e:len(tok(rl.format_prompt(e),add_special_tokens=False).input_ids))
            # Calibration covers the longest train prompt and a full 96-token backward graph.
            synthetic=dict(example=longest,prompt_ids=tok(rl.format_prompt(longest),add_special_tokens=False).input_ids,
                           generated_ids=[20]*96)
            saved_deltas=module.state_dict()
            import copy
            saved_adam=[copy.deepcopy(o.state_dict()) for o in optimizers]
            calibration_task=longest['task_id']
            calibration=update(module,tok,[synthetic,synthetic],[1.,-1.],cfg,optimizers,calibration_task,initial,
                               auxiliary_items(bank,neutral,spec['seed'],8,calibration_task))
            module.load_state_dict(saved_deltas)
            for optimizer,value in zip(optimizers,saved_adam):optimizer.load_state_dict(value)
            assert calibration['accepted']
            del saved_deltas,saved_adam
            enc=tok(rl.format_prompt(longest),add_special_tokens=False,return_tensors='pt',return_token_type_ids=False).to(module.device)
            with torch.no_grad(),module.gate((True,)*8),cached():
                generated=module.model.generate(**enc,min_new_tokens=96,max_new_tokens=96,eos_token_id=None,
                    pad_token_id=tok.pad_token_id,do_sample=False,temperature=None,top_p=None,top_k=None)
            assert generated.shape[1]-enc.input_ids.shape[1]==96
            with module.gate((False,)*8): after=witnesses(module.model,tok,data['dev'][:1])
            assert all(torch.equal(a,b) for a,b in zip(before_witness,after))
            assert all(r['accepted'] and r['delta_max_move']>0 for r in history)
            assert torch.cuda.max_memory_allocated()/2**30<7.25
            del enc,generated
        else:
            with module.gate((False,)*8): after=witnesses(module.model,tok,witness_examples)
            assert all(torch.equal(a,b) for a,b in zip(before_witness,after)), 'All-off witness changed'
        assert module.assert_frozen_base()
        assert len(history)==spec['steps'] and all(r['foreign_optimizers_unchanged'] for r in history)
        if not spec['smoke']:assert [sum(r['task_id']==t for r in history) for t in range(8)]==[spec['steps']//8]*8
        payload=dict(deltas=module.state_dict(),manifest=module.manifest(),plan_sha256=digest(plan),spec=spec,step=step)
        fit=output/'fits'/spec['id'];rl.atomic_torch(fit/'bank_final.pt',payload)
        atomic(fit/'completed.json',dict(spec=spec,plan_sha256=digest(plan),bank_sha256=hash_file(fit/'bank_final.pt'),
            final_step=step,status=dict(complete=True),all_off_equal=True,owner_gradient_audits=True,
            accepted_updates=sum(r['accepted'] for r in history),training_seconds=sum(r['seconds'] for r in history),
            response_tokens=sum(r['response_tokens'] for r in history),auxiliary_tokens=sum(r['auxiliary_tokens'] for r in history),
            initial_source_owners=bank['source_owners'],final_dev=dev_history[-1] if dev_history else None,
            elapsed_this_session=time.monotonic()-started,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30))
        print('COMPLETE',spec['id'],flush=True)
    finally:module.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','smoke','worker','status'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--job')
    parser.add_argument('--gpus',default='0,1,2,3,4,5')
    args=parser.parse_args();output=args.output.resolve()
    if args.action=='status':print((output/'shard_status.json').read_text());return
    output.mkdir(parents=True,exist_ok=True)
    path=output/'experiment_plan.json'
    if args.action=='worker':
        plan=json.loads(path.read_text())
        assert plan==make_plan(plan['smoke']),'Protocol or source changed'
        spec=next(j for j in plan['jobs'] if j['id']==args.job)
        try:
            with exclusive(output/'fits'/spec['id']/'worker.lock'):
                if spec['kind']=='main':train_bank(spec,plan,output)
                else:
                    previous.execution_revision=execution_revision
                    previous.merge_fit(spec,plan,output)
        except BlockingIOError:raise SystemExit(75)
        return
    with exclusive(output/'launcher.lock'):
        plan=make_plan(args.action=='smoke')
        if path.exists():assert json.loads(path.read_text())==plan,'Protocol changed: choose a fresh output'
        else:atomic(path,plan)
        assigned=[j['id'] for j in plan['jobs']]
        devices=args.gpus.split(',')
        assert devices == [str(i) for i in range(6)], 'Use the six Ogg GPUs assigned by the user'
        if args.action == 'run':
            smoke_record=json.loads((output.parent/'smoke/SHARD_COMPLETE.json').read_text())
            assert smoke_record['completed']==8 and not smoke_record['failed']
            assert smoke_record['execution_revision']==execution_revision(output,plan)['id']
        def terminate(signum,frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,terminate)
        import scheduler
        scheduler.supervise(sys.modules[__name__],plan,output,assigned,devices)


if __name__=='__main__':main()
