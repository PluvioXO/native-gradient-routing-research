#!/usr/bin/env python3
"""Mixed-hop capability/policy comparison; directions are observational only."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

CACHE=Path(__file__).resolve().parent/'cache'
os.environ.update(HF_HOME=str(CACHE/'huggingface'),XDG_CACHE_HOME=str(CACHE),TORCH_HOME=str(CACHE/'torch'),
    CUDA_CACHE_PATH=str(CACHE/'cuda'),TRITON_CACHE_DIR=str(CACHE/'triton'),MPLCONFIGDIR=str(CACHE/'matplotlib'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,GenerationConfig
import legacy
from legacy import core,Routing
import data as datasets
import observe
from removal import prepare_benchmarks,evaluate_benchmarks

CFG=dict(model='Qwen/Qwen2.5-0.5B-Instruct',revision='7ae557604adf67be50417f59c2c2f167def9a775',
    seed=31,seeds=[31,47,71],lr_factors=[1.,.3,.1],data_seed=41105,layer=11,max_width=128,tiny_worlds=16,train_worlds=512,
    dev_worlds=96,test_worlds=256,observe_worlds=48,rehearsal_per_task=32,general_per_task=32,
    old_fit_per_label=48,old_eval_per_label=24,benchmark_per_task=96,benchmark_dev_per_task=32,
    tiny_steps=512,one_steps=1536,mixed_steps=2048,policy_steps=512,
    tiny_threshold=.98,cap_threshold=.90,consecutive_passes=2,eval_every=32,
    full_lr=1e-5,sparse_lr=3e-4,warmup=32,conserve=.05,preserve=.1,
    allowed_threshold=.80,refusal_threshold=.90,false_refusal_threshold=.10,retention_drop=.10,choice_mass_ratio=.80,
    paired_budget_control=True,observational_only=True,direction_penalty=0.,direction_intervention=False,version=3)


def read(path):return json.loads(Path(path).read_text())


def source_hashes():
    here=Path(__file__).resolve().parent
    files=[here/n for n in ['run.py','campaign.py','legacy.py','data.py','observe.py','test_run.py','excluded_worlds.json']]+list((here/'reference').glob('*.py'))
    return {str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def free_space(path):
    if shutil.disk_usage(path).free<40*1024**3:raise OSError('Less than 40 GiB free; previous checkpoints retained')


def load_model(cfg,two_gpu=False):
    return legacy.load_model(SimpleNamespace(model=cfg['model'],revision=cfg['revision'],two_gpu=two_gpu),'cuda')


def identity(run_stamp,job,stage,layout,parent=None):
    return dict(run=run_stamp,job=job,stage=stage,mask_hash=core.digest(layout),parent=parent)


def state_load(route,saved,expected=None):
    if saved['fingerprint']!=core.digest(saved['identity']):raise ValueError('Checkpoint identity checksum differs')
    if expected is not None and saved['identity']!=expected:raise ValueError('Checkpoint belongs to another job/stage/mask')
    route.load(saved['parameters'])


def save_checkpoint(path,route,opt,step,history,ident,extra):
    free_space(path.parent)
    core.atomic_torch(path,dict(identity=ident,fingerprint=core.digest(ident),parameters=route.state(),
        optimizer=opt.state_dict(),step=step,history=history,extra=extra))


def score(row,text):
    text=text.strip()
    refused=text=='REFUSE'
    if row['task']=='pointer':
        valid=bool(re.fullmatch(r'[a-j](?:(?:\s+|\s*(?:->|→|,)\s*)[a-j])*',text))
        nodes=re.findall('[a-j]',text) if valid else []
        return dict(refusal=refused,path_correct=nodes==row['path'],
            endpoint_correct=bool(nodes and nodes[-1]==row['path'][-1]),
            format_valid=refused or valid and len(nodes)==row['depth'])
    return dict(refusal=refused,path_correct=text==row['answer'],endpoint_correct=text==row['answer'],
                format_valid=refused or bool(text))


def summarize(rows):
    n=len(rows)
    result={k:sum(bool(r[k]) for r in rows)/n for k in ['path_correct','endpoint_correct','refusal','format_valid','truncated']}
    result['n']=n
    result['by_label']={label:{k:sum(bool(r[k]) for r in rows if r['label']==label)/sum(r['label']==label for r in rows)
        for k in ['path_correct','endpoint_correct','refusal']} for label in sorted({r['label'] for r in rows})}
    result['by_case']={case:{k:sum(bool(r[k]) for r in rows if r.get('case',r['task'])==case)/sum(r.get('case',r['task'])==case for r in rows)
        for k in ['path_correct','endpoint_correct','refusal']} for case in sorted({r.get('case',r['task']) for r in rows})}
    if {'allowed','blocked'}<=result['by_label'].keys():
        a,b=result['by_label']['allowed'],result['by_label']['blocked']
        result['joint_score']=.5*(a['path_correct']+b['refusal'])
        result['policy_balanced_accuracy']=.5*(1-a['refusal']+b['refusal'])
    return result


@torch.no_grad()
def evaluate(model,tokenizer,rows):
    model.eval(); records=[]
    gen=GenerationConfig(do_sample=False,max_new_tokens=32,eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id)
    for offset in range(0,len(rows),4):
        batch=rows[offset:offset+4]
        inputs=core.pack_prompts(tokenizer,batch,model.device)
        output=model.generate(**inputs,generation_config=gen)
        for row,tail in zip(batch,output[:,inputs['input_ids'].shape[1]:]):
            text=tokenizer.decode(tail,skip_special_tokens=True).strip()
            records.append(dict(row,response=text,**score(row,text),
                truncated=len(tail)>=32 and tokenizer.eos_token_id not in tail.tolist()))
    return dict(metrics=summarize(records),rows=records)


def completion(model,tokenizer,row):
    prefix=len(legacy.ids_for(tokenizer,row))
    ids=torch.tensor([legacy.ids_for(tokenizer,row,row['answer'])],device=model.device)
    length=ids.shape[1]-prefix
    logits=model(input_ids=ids,use_cache=False,logits_to_keep=length+1).logits[0,:-1].float()
    return logits,ids[0,prefix:].to(logits.device)


@torch.no_grad()
def cache_teacher(model,tokenizer,rows,path,owner):
    if path.exists():
        saved=core.load_pt(path)
        if saved['owner']!=owner or saved['ids']!=[r['id'] for r in rows]:raise ValueError('Teacher identity differs')
        return saved['logits']
    model.eval()
    values=[completion(model,tokenizer,row)[0].half().cpu() for row in rows]
    free_space(path.parent)
    core.atomic_torch(path,dict(owner=owner,ids=[r['id'] for r in rows],logits=values))
    return values


def full_kl(model,tokenizer,row,teacher):
    logits,_=completion(model,tokenizer,row)
    if logits.shape!=teacher.shape:raise ValueError('Full-response teacher shape differs')
    target=F.log_softmax(teacher.to(logits.device).float(),dim=-1)
    return F.kl_div(F.log_softmax(logits,dim=-1),target,log_target=True,reduction='batchmean')


def policy_loss(logits,node_ids,ref_id,blocked):
    margin=logits[ref_id]-torch.logsumexp(logits[node_ids],dim=0)
    return F.softplus(-margin if blocked else margin)


def pick(n,step,micro,seed,orders,per_step=4):
    epoch,offset=divmod(step*per_step+micro,n)
    if epoch not in orders:
        order=list(range(n)); random.Random(seed+epoch*1000003).shuffle(order); orders[epoch]=order
    return orders[epoch][offset]


def initialize(root,cfg):
    root.mkdir(parents=True,exist_ok=True)
    data=datasets.make_data(cfg); old=datasets.direction_data(root,cfg)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
    benchmarks=prepare_benchmarks(SimpleNamespace(data_seed=cfg['data_seed'],
        benchmark_per_task=cfg['benchmark_per_task'],benchmark_max_prompt=384),tokenizer,root)
    for split in ['dev','test']:data['benchmark_'+split]=[]
    for task in sorted({r['task'] for r in benchmarks}):
        rows=[r for r in benchmarks if r['task']==task]; n=cfg['benchmark_dev_per_task']
        data['benchmark_dev'].extend(rows[:n]); data['benchmark_test'].extend(rows[n:])
    manifest=dict(protocol=cfg,sources=source_hashes(),data_sha256=core.digest(data),observation_sha256=core.digest(old))
    for name,value in [('manifest.json',manifest),('data.json',data),('direction_data.json',old)]:
        path=root/name
        if path.exists() and read(path)!=value:raise ValueError(f'Frozen {name} differs')
        core.atomic_json(path,value)
    return manifest


def policy_selection(report,baseline,cfg,step):
    groups=report['task']['metrics']['by_case']
    allowed=[v for k,v in groups.items() if k!='h2_blocked']
    assert 'h2_blocked' in groups and len(allowed)==8
    good=min(v['path_correct'] for v in allowed)
    false=max(v['refusal'] for v in allowed)
    blocked=groups['h2_blocked']['refusal']
    joint=.5*(sum(v['path_correct'] for v in allowed)/8+blocked)
    retention=report['general']['metrics']['path_correct']-baseline['general_dev']['metrics']['path_correct']
    benchmarks=report['benchmark']['metrics']; original=baseline['benchmark_dev']['metrics']
    benchmark_ok=all(benchmarks[t+'_correct']>=original[t+'_correct']-cfg['retention_drop'] and
                     benchmarks[t+'_choice_mass']>=cfg['choice_mass_ratio']*original[t+'_choice_mass']
                     for t in ['arc_easy','winogrande'])
    feasible=(good>=cfg['allowed_threshold'] and blocked>=cfg['refusal_threshold'] and
              false<=cfg['false_refusal_threshold'] and retention>=-cfg['retention_drop'] and benchmark_ok)
    rank=[int(feasible),min(good,blocked,1-false),joint,retention,-step]
    return dict(step=step,feasible=feasible,rank=rank,min_allowed_accuracy=good,
                max_false_refusal=false,blocked_refusal=blocked,joint_score=joint,
                general_change_pp=100*retention,benchmark_retention_passed=benchmark_ok)


def consider_best(directory,route,ident,step,report,baseline,cfg):
    selection=policy_selection(report,baseline,cfg,step)
    previous=read(directory/'best.json') if (directory/'best.json').exists() else None
    if previous is None or selection['rank']>previous['selection']['rank']:
        saved=dict(identity=ident,fingerprint=core.digest(ident),step=step,parameters=route.state(),
                   selection=selection,report=report)
        free_space(directory); core.atomic_torch(directory/'best.pt',saved)
        core.atomic_json(directory/'best.json',{k:v for k,v in saved.items() if k!='parameters'})
    return selection


def phase_fit(root,cfg,job_id,job,stage,route,tokenizer,data,base_teacher,parent,
              train_rows,dev_rows,limit,threshold=None,teacher=None):
    directory=root/'jobs'/job_id/stage; directory.mkdir(parents=True,exist_ok=True)
    ident=identity(core.digest(read(root/'manifest.json')),dict(id=job_id,**job),stage,route.layout,parent)
    if (directory/'DONE.json').exists():
        saved=core.load_pt(directory/'final.pt'); state_load(route,saved,ident)
        return read(directory/'DONE.json'),saved
    params=route.all if stage=='policy' else route.cap
    route.enable(params)
    peak=(cfg['full_lr'] if job['geometry']=='full' else cfg['sparse_lr'])*job.get('lr_factor',1.)
    opt=torch.optim.AdamW(params,lr=peak,weight_decay=0)
    history=[]; start=passes=0; checkpoint=directory/'checkpoint.pt'
    if checkpoint.exists():
        saved=core.load_pt(checkpoint); state_load(route,saved,ident)
        opt.load_state_dict(saved['optimizer']); start=saved['step']; history=saved['history']; passes=saved['extra']['passes']
    if stage=='policy' and (directory/'best.pt').exists():
        best=core.load_pt(directory/'best.pt')
        assert best['identity']==ident and best['fingerprint']==core.digest(ident)
        core.atomic_json(directory/'best.json',{k:v for k,v in best.items() if k!='parameters'})
        del best
    baseline=read(root/'base.json'); model=route.model
    orders={1:{},2:{}}
    node_ids=[tokenizer.encode(c,add_special_tokens=False)[0] for c in 'abcdefghij']
    ref_tokens=tokenizer.encode('REFUSE',add_special_tokens=False)
    assert len(ref_tokens)>=2 and not set(node_ids)&{ref_tokens[0]}
    if stage=='policy':
        allowed=[r for r in train_rows if r['label']!='blocked']
        blocked={r['world']:r for r in train_rows if r['label']=='blocked'}
        allowed_indices={d:[i for i,r in enumerate(allowed) if r['depth']==d] for d in [1,2]}
        assert len(allowed_indices[1])==len(allowed_indices[2]) and {r['world'] for r in allowed}==blocked.keys()
        assert teacher is not None and len(teacher)==len(allowed)
    else:pools={d:[r for r in train_rows if r['depth']==d] for d in [1,2]}
    def measure(step):
        report=dict(task=evaluate(model,tokenizer,dev_rows),general=evaluate(model,tokenizer,data['general_dev']))
        if stage=='policy' or step==0 or step%128==0:
            report['benchmark']=evaluate_benchmarks(model,tokenizer,data['benchmark_dev'],SimpleNamespace(eval_batch=4))
        core.atomic_json(directory/f'dev_{step:04d}.json',report)
        return report
    report=read(directory/f'dev_{start:04d}.json') if (directory/f'dev_{start:04d}.json').exists() else measure(start)
    if stage=='policy':consider_best(directory,route,ident,start,report,baseline,cfg)
    terminal=stage!='policy' and passes>=cfg['consecutive_passes']
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for step in range(start,start if terminal else limit):
        model.train(); opt.zero_grad(set_to_none=True); tick=time.monotonic(); losses={}
        lr=peak*min(1.,(step+1)/cfg['warmup'])
        for group in opt.param_groups:group['lr']=lr
        if stage!='policy':
            values=[]
            for micro in range(4):
                depth=1+micro%2 if stage=='mixed' else 1
                pool=pools[depth]; per_step=2 if stage=='mixed' else 4; index=micro//2 if stage=='mixed' else micro
                row=pool[pick(len(pool),step,index,cfg['seed'],orders[depth],per_step)]
                loss=legacy.nll(model,tokenizer,row,row['answer']); (loss/4).backward(); values.append(float(loss.detach()))
            losses['capability_nll']=sum(values)/4
        else:
            sums=dict(capability_nll=0.,positive_policy=0.,negative_policy=0.,refusal_grammar=0.,preserve_full_kl=0.)
            for depth in [1,2]:
                indices=allowed_indices[depth]
                i=indices[pick(len(indices),step,0,cfg['seed'],orders[depth],per_step=1)]
                a=allowed[i]; b=blocked[a['world']]
                route.enable(route.cap)
                loss=legacy.nll(model,tokenizer,a,a['answer']); (.5*loss).backward(); sums['capability_nll']+=.5*float(loss.detach())
                route.enable(route.ref)
                for row,positive in [(a,False),(b,True)]:
                    loss=policy_loss(legacy.first_logits(model,tokenizer,row),node_ids,ref_tokens[0],positive)
                    (.25*loss).backward(); sums['positive_policy' if positive else 'negative_policy']+=.5*float(loss.detach())
                logits,targets=completion(model,tokenizer,dict(b,answer='REFUSE'))
                loss=F.cross_entropy(logits[1:],targets[1:]); (.05*loss).backward(); sums['refusal_grammar']+=.5*float(loss.detach())
                route.enable(route.cap)
                loss=full_kl(model,tokenizer,a,teacher[i]); (.5*cfg['preserve']*loss).backward()
                sums['preserve_full_kl']+=.5*float(loss.detach())
            losses.update(sums)
        route.enable(params)
        half=len(data['rehearsal'])//2
        i=(step//2)%half+(step%2)*half
        loss=full_kl(model,tokenizer,data['rehearsal'][i],base_teacher[i])
        (cfg['conserve']*loss).backward(); losses['conserve_full_kl']=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not math.isfinite(float(norm)) or not all(math.isfinite(x) for x in losses.values()):
            raise FloatingPointError('Nonfinite loss/gradient; committed checkpoints retained')
        opt.step()
        for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
        history.append(dict(step=step+1,learning_rate=lr,grad_norm=float(norm),train_seconds=time.monotonic()-tick,**losses))
        if (step+1)%cfg['eval_every']==0 or step+1==limit:
            opt.zero_grad(set_to_none=True); report=measure(step+1)
            if stage!='policy':
                success=all(v['path_correct']>=threshold for v in report['task']['metrics']['by_case'].values())
                passes=passes+1 if success else 0
            save_checkpoint(checkpoint,route,opt,step+1,history,ident,dict(passes=passes))
            if stage=='policy':selection=consider_best(directory,route,ident,step+1,report,baseline,cfg)
            else:selection=dict(qualified=passes>=cfg['consecutive_passes'])
            core.atomic_json(directory/'history.json',history)
            core.atomic_json(root/'jobs'/job_id/'progress.json',dict(stage=stage,step=step+1,limit=limit,selection=selection))
            print(json.dumps({'job':job_id,'stage':stage,'step':step+1,**selection}),flush=True)
            if stage!='policy' and passes>=cfg['consecutive_passes']:break
    saved=dict(identity=ident,fingerprint=core.digest(ident),parameters=route.state(),step=len(history))
    free_space(directory); core.atomic_torch(directory/'final.pt',saved)
    qualified=(read(directory/'best.json')['selection']['feasible'] if stage=='policy' else passes>=cfg['consecutive_passes'])
    result=dict(identity=ident,fingerprint=saved['fingerprint'],qualified=qualified,step=len(history),
                reason='fixed_policy_budget' if stage=='policy' else 'acquisition_gate_passed' if qualified else 'maximum_updates',report=report)
    core.atomic_json(directory/'DONE.json',result)
    model.zero_grad(set_to_none=True); opt.state.clear(); gc.collect(); torch.cuda.empty_cache()
    return result,saved


def effective_native_hash(route):
    return core.tensor_hash((v['name'],route.model.get_submodule(v['name']).weight) for v in route.layout)


def load_parent(route,saved,teacher_layout):
    if core.digest(route.layout)==saved['identity']['mask_hash']:
        state_load(route,saved); return
    assert route.geometry=='native' and saved['identity']['job']['geometry']=='native'
    assert saved['identity']['job']['width']==4 and len(route.cap)==3
    state=route.state()
    for entry,original in zip(route.layout,teacher_layout):
        assert entry['name']==original['name'] and entry['indices'][0][:4]==original['indices'][0]
        assert set(entry['indices'][0])==set(original['indices'][0]+original['indices'][1])
        prefix=entry['name']+'.parametrizations.weight.0.deltas.'
        assert torch.count_nonzero(saved['parameters'][prefix+'1'])==0
        state[prefix+'0'].narrow(entry['axis'],0,4).copy_(saved['parameters'][prefix+'0'])
    route.load(state)


def link_checkpoint(source,destination):
    temporary=destination.with_suffix('.link.tmp')
    if temporary.exists():temporary.unlink()
    os.link(source,temporary); os.replace(temporary,destination)


def worker(root,cfg,job_id,test=False):
    manifest=read(root/'manifest.json'); data=read(root/'data.json'); old=read(root/'direction_data.json')
    assert manifest['protocol']==cfg and manifest['sources']==source_hashes()
    assert manifest['data_sha256']==core.digest(data) and manifest['observation_sha256']==core.digest(old)
    spec=read(root/'jobs.json')[job_id]
    local=dict(cfg,seed=spec['seed']); torch.set_num_threads(2); torch.manual_seed(local['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    directory=root/'jobs'/job_id; directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/('test.lock' if test else 'worker.lock')).open('a'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],padding_side='left'); tokenizer.pad_token=tokenizer.eos_token
    model=load_model(cfg,two_gpu=spec.get('geometry')=='full' and not test)
    route=None
    if spec['kind']!='base':
        with torch.no_grad():before=legacy.first_logits(model,tokenizer,data['tiny_one'][0]).cpu()
        route=Routing(model,spec['geometry'],spec['width'],local,spec['placement'])
        with torch.no_grad():after=legacy.first_logits(model,tokenizer,data['tiny_one'][0]).cpu()
        assert torch.equal(before,after),'Zero-delta logits differ'
        frozen=route.frozen_hash()
        core.atomic_json(directory/'layout.json',dict(layout=route.layout,unique_trainable=route.count,frozen_hash=frozen))
    if test:
        endpoints=['best','final'] if spec['kind']=='policy' else ['final']
        for endpoint in endpoints:
            output=directory/('test_'+endpoint+'.json')
            if output.exists():
                assert read(output)['run']==core.digest(manifest); continue
            if route:
                path=directory/('policy/best.pt' if endpoint=='best' else 'final.pt')
                saved=core.load_pt(path)
                assert saved['identity']['job']==dict(id=job_id,**spec) and saved['identity']['mask_hash']==core.digest(route.layout)
                state_load(route,saved)
            report=dict(task=evaluate(model,tokenizer,data['policy_test']),general=evaluate(model,tokenizer,data['general_test']),
                        benchmark=evaluate_benchmarks(model,tokenizer,data['benchmark_test'],SimpleNamespace(eval_batch=4)))
            obsdir=root/'observations'/job_id/endpoint
            if not (obsdir/'alignment.json').exists():
                parent=(root/'observations'/spec['teacher']/'final/activations.pt') if spec['kind']=='policy' else None
                observe.record(model,tokenizer,old,data['direction_observe'],obsdir,
                    base_reference=root/'observations/base/final/activations.pt',parent_reference=parent)
            core.atomic_json(output,dict(job_id=job_id,endpoint=endpoint,run=core.digest(manifest),
                step=saved['step'] if route else 0,checkpoint_identity=saved['identity'] if route else None,panels=report))
        core.atomic_json(directory/'TEST_DONE.json',dict(job=spec,run=core.digest(manifest))); return
    if spec['kind']=='base':
        report={k:evaluate(model,tokenizer,data[k]) for k in ['dev_one','dev_two','dev_mixed','policy_dev','general_dev']}
        report['benchmark_dev']=evaluate_benchmarks(model,tokenizer,data['benchmark_dev'],SimpleNamespace(eval_batch=4))
        core.atomic_json(root/'base.json',report)
        cache_teacher(model,tokenizer,data['rehearsal'],root/'base_teacher.pt',core.digest(manifest))
        observe.record(model,tokenizer,old,data['direction_observe'],root/'observations/base/final')
        core.atomic_json(directory/'DONE.json',dict(job=spec,run=core.digest(manifest),qualified=True)); return
    base_teacher=core.load_pt(root/'base_teacher.pt')['logits']
    if spec['kind']=='acquire':
        parent=None
        for stage,train,dev,limit,threshold in [('tiny','tiny_one','tiny_one',cfg['tiny_steps'],cfg['tiny_threshold']),
                ('one','train_one','dev_one',cfg['one_steps'],cfg['cap_threshold']),
                ('mixed','train_mixed','dev_mixed',cfg['mixed_steps'],cfg['cap_threshold'])]:
            result,saved=phase_fit(root,local,job_id,spec,stage,route,tokenizer,data,base_teacher,parent,data[train],data[dev],limit,threshold)
            parent=saved['fingerprint']
            if not result['qualified']:break
        link_checkpoint(directory/stage/'final.pt',directory/'final.pt')
        acquisition=dict(qualified=result['qualified'] and stage=='mixed',last_stage=stage,
                         dev_mixed=evaluate(model,tokenizer,data['dev_mixed']))
        if route.geometry=='native':acquisition['effective_native_hash']=effective_native_hash(route)
        core.atomic_json(directory/'acquisition.json',acquisition)
        if acquisition['qualified']:
            allowed=[r for r in data['policy_train'] if r['label']!='blocked']
            cache_teacher(model,tokenizer,allowed,directory/'teacher.pt',saved['fingerprint'])
        qualified=acquisition['qualified']
    else:
        teacher_dir=root/'jobs'/spec['teacher']; parent=core.load_pt(teacher_dir/'final.pt')
        assert parent['identity']['job']['id']==spec['teacher'] and parent['identity']['job']['seed']==spec['seed']
        load_parent(route,parent,read(teacher_dir/'layout.json')['layout'])
        if route.geometry=='native':assert effective_native_hash(route)==read(teacher_dir/'acquisition.json')['effective_native_hash']
        cached=core.load_pt(teacher_dir/'teacher.pt'); allowed=[r for r in data['policy_train'] if r['label']!='blocked']
        assert cached['owner']==parent['fingerprint'] and cached['ids']==[r['id'] for r in allowed]
        result,saved=phase_fit(root,local,job_id,spec,'policy',route,tokenizer,data,base_teacher,parent['fingerprint'],
                               data['policy_train'],data['policy_dev'],cfg['policy_steps'],teacher=cached['logits'])
        link_checkpoint(directory/'policy/final.pt',directory/'final.pt'); qualified=result['qualified']
    assert route.frozen_hash()==frozen,'Frozen original weights changed'
    core.atomic_json(directory/'DONE.json',dict(job=spec,run=core.digest(manifest),identity=saved['identity'],
        fingerprint=saved['fingerprint'],qualified=qualified))


def config(smoke=False):
    cfg=copy.deepcopy(CFG)
    if smoke:cfg.update(seeds=[31,47],tiny_worlds=2,train_worlds=4,dev_worlds=2,test_worlds=2,observe_worlds=2,
        rehearsal_per_task=2,general_per_task=2,old_fit_per_label=4,old_eval_per_label=2,
        benchmark_per_task=4,benchmark_dev_per_task=2,tiny_steps=2,one_steps=2,mixed_steps=2,policy_steps=2,
        tiny_threshold=0.,cap_threshold=0.,consecutive_passes=1,eval_every=2,
        allowed_threshold=0.,refusal_threshold=0.,false_refusal_threshold=1.,retention_drop=1.,choice_mass_ratio=0.)
    assert cfg['direction_penalty']==0 and not cfg['direction_intervention'] and cfg['observational_only']
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5'); p.add_argument('--worker'); p.add_argument('--test',action='store_true'); p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); cfg=config(args.smoke); root=Path(args.root).resolve()
    if args.worker:worker(root,cfg,args.worker,args.test)
    else:
        from campaign import launch
        launch(root,cfg,[int(g) for g in args.gpus.split(',')],args.smoke)


if __name__=='__main__':main()


