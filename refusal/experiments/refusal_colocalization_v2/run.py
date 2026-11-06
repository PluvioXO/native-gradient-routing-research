#!/usr/bin/env python3
"""Gated native capability/policy pilot; refusal directions are observation only."""
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
    seed=31,data_seed=41104,layer=11,max_width=128,tiny_worlds=16,train_worlds=512,
    dev_worlds=96,test_worlds=256,observe_worlds=48,rehearsal_per_task=32,general_per_task=32,
    old_fit_per_label=48,old_eval_per_label=24,benchmark_per_task=96,benchmark_dev_per_task=32,
    tiny_steps=256,one_steps=1536,two_steps=2048,policy_steps=512,
    tiny_threshold=.98,cap_threshold=.90,consecutive_passes=2,eval_every=32,
    full_lr=1e-5,sparse_lr=3e-4,warmup=32,conserve=.05,preserve=.1,
    observational_only=True,direction_penalty=0.,direction_intervention=False,version=2)


def read(path):return json.loads(Path(path).read_text())


def source_hashes():
    here=Path(__file__).resolve().parent
    files=[here/n for n in ['run.py','legacy.py','data.py','observe.py','excluded_worlds.json']]+list((here/'reference').glob('*.py'))
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


def initial_jobs():
    return {'full_off':dict(kind='acquire',geometry='full',width=0,placement='same',conserve=0.),
            'full_on':dict(kind='acquire',geometry='full',width=0,placement='same',conserve=CFG['conserve']),
            'native4':dict(kind='acquire',geometry='native',width=4,placement='same',conserve=0.),
            'native128':dict(kind='acquire',geometry='native',width=128,placement='same',conserve=0.)}


def initialize(root,cfg):
    root.mkdir(parents=True,exist_ok=True)
    data=datasets.make_data(cfg)
    old=datasets.direction_data(root,cfg)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
    benchmarks=prepare_benchmarks(SimpleNamespace(data_seed=cfg['data_seed'],
        benchmark_per_task=cfg['benchmark_per_task'],benchmark_max_prompt=384),tokenizer,root)
    data['benchmark_dev']=[]; data['benchmark_test']=[]
    for task in sorted({r['task'] for r in benchmarks}):
        rows=[r for r in benchmarks if r['task']==task]
        data['benchmark_dev'].extend(rows[:cfg['benchmark_dev_per_task']])
        data['benchmark_test'].extend(rows[cfg['benchmark_dev_per_task']:])
    manifest=dict(protocol=cfg,sources=source_hashes(),data_sha256=core.digest(data),observation_sha256=core.digest(old))
    for name,value in [('manifest.json',manifest),('data.json',data),('direction_data.json',old)]:
        p=root/name
        if p.exists() and read(p)!=value:raise ValueError(f'Frozen {name} differs; use a new run directory')
        core.atomic_json(p,value)
    if not (root/'jobs.json').exists():core.atomic_json(root/'jobs.json',initial_jobs())
    return manifest


def phase_fit(root,cfg,job_id,job,stage,route,tokenizer,data,base_teacher,parent,
              train_rows,dev_rows,limit,threshold=None,teacher=None):
    directory=root/'jobs'/job_id/stage; directory.mkdir(parents=True,exist_ok=True)
    ident=identity(core.digest(read(root/'manifest.json')),dict(id=job_id,**job),stage,route.layout,parent)
    final_path=directory/'final.pt'
    if (directory/'DONE.json').exists():
        saved=core.load_pt(final_path); state_load(route,saved,ident)
        return read(directory/'DONE.json'),saved
    params=route.all if stage=='policy' else route.cap
    route.enable(params)
    opt=torch.optim.AdamW(params,lr=cfg['full_lr'] if job['geometry']=='full' else cfg['sparse_lr'],weight_decay=0)
    history,start,passes,bad= [],0,0,0
    checkpoint=directory/'checkpoint.pt'
    if checkpoint.exists():
        saved=core.load_pt(checkpoint); state_load(route,saved,ident)
        opt.load_state_dict(saved['optimizer']); start=saved['step']; history=saved['history']
        passes=saved['extra']['passes']; bad=saved['extra']['bad']
    orders={}
    model=route.model
    node_ids=[tokenizer.encode(c,add_special_tokens=False)[0] for c in 'abcdefghij']
    ref_tokens=tokenizer.encode('REFUSE',add_special_tokens=False)
    assert len(ref_tokens)>=1 and not set(node_ids)&{ref_tokens[0]}
    allowed=[r for r in train_rows if r['label']=='allowed'] if stage=='policy' else []
    blocked=[r for r in train_rows if r['label']=='blocked'] if stage=='policy' else []
    if stage=='policy':
        assert len(allowed)==len(blocked) and all(a['world']==b['world'] for a,b in zip(allowed,blocked))
    teacher_metrics=read(root/'jobs'/job['teacher']/'acquisition.json')['dev_two']['metrics'] if stage=='policy' else None
    baseline_general=read(root/'base.json')['general_dev']['metrics']['path_correct']
    def measure(step):
        value=dict(task=evaluate(model,tokenizer,dev_rows),general=evaluate(model,tokenizer,data['general_dev']))
        if step==0 or step%128==0:
            value['benchmark']=evaluate_benchmarks(model,tokenizer,data['benchmark_dev'],SimpleNamespace(eval_batch=4))
        core.atomic_json(directory/f'dev_{step:04d}.json',value)
        return value
    if start==0:measure(0)
    qualified=False; reason='maximum_updates'; last_report=None
    terminal=(stage!='policy' and passes>=cfg['consecutive_passes']) or (stage=='policy' and bad>=2)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for step in range(start,start if terminal else limit):
        model.train(); opt.zero_grad(set_to_none=True); tick=time.monotonic(); losses={}
        lr=(cfg['full_lr'] if job['geometry']=='full' else cfg['sparse_lr'])*min(1.,(step+1)/cfg['warmup'])
        for group in opt.param_groups:group['lr']=lr
        if stage!='policy':
            route.enable(route.cap); values=[]
            for micro in range(4):
                row=train_rows[pick(len(train_rows),step,micro,cfg['seed'],orders)]
                loss=legacy.nll(model,tokenizer,row,row['answer']); (loss/4).backward(); values.append(float(loss.detach()))
            losses['capability_nll']=sum(values)/4
        else:
            sums={'capability_nll':0.,'positive_policy':0.,'negative_policy':0.,'refusal_grammar':0.}
            for micro in range(2):
                i=pick(len(allowed),step,micro,cfg['seed'],orders,per_step=2)
                a,b=allowed[i],blocked[i]
                route.enable(route.cap)
                loss=legacy.nll(model,tokenizer,a,a['answer']); (.5*loss).backward(); sums['capability_nll']+=.5*float(loss.detach())
                route.enable(route.ref)
                for row,positive in [(a,False),(b,True)]:
                    loss=policy_loss(legacy.first_logits(model,tokenizer,row),node_ids,ref_tokens[0],positive)
                    (.25*loss).backward(); sums['positive_policy' if positive else 'negative_policy']+=.5*float(loss.detach())
                refuse=dict(b,answer='REFUSE')
                logits,targets=completion(model,tokenizer,refuse)
                loss=F.cross_entropy(logits[1:],targets[1:]); (.05*loss).backward(); sums['refusal_grammar']+=.5*float(loss.detach())
            losses.update(sums)
            route.enable(route.cap)
            i=step%len(allowed)
            loss=full_kl(model,tokenizer,allowed[i],teacher[i]); (cfg['preserve']*loss).backward()
            losses['preserve_full_kl']=float(loss.detach())
        route.enable(params)
        if job['conserve']:
            i=step%len(data['rehearsal'])
            loss=full_kl(model,tokenizer,data['rehearsal'][i],base_teacher[i]); (job['conserve']*loss).backward()
            losses['conserve_full_kl']=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not math.isfinite(float(norm)) or not all(math.isfinite(x) for x in losses.values()):
            raise FloatingPointError('Nonfinite training loss/gradient')
        opt.step()
        for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
        history.append(dict(step=step+1,learning_rate=lr,grad_norm=float(norm),train_seconds=time.monotonic()-tick,**losses))
        check=(step+1)%cfg['eval_every']==0 or step+1==limit
        if check:
            opt.zero_grad(set_to_none=True)
            last_report=measure(step+1)
            m=last_report['task']['metrics']
            if stage!='policy':
                success=m['path_correct']>=threshold and all(x['path_correct']>=threshold for x in m['by_label'].values())
                passes=passes+1 if success else 0
                if passes>=cfg['consecutive_passes']:qualified=True; reason='acquisition_gate_passed'
            else:
                a,b=m['by_label']['allowed'],m['by_label']['blocked']
                qualified=b['refusal']>=.9 and a['path_correct']>=.8 and a['refusal']<=.1
                damaged=(a['path_correct']<teacher_metrics['by_label']['allowed']['path_correct']-.2 or
                    last_report['general']['metrics']['path_correct']<baseline_general-.2)
                bad=bad+1 if step+1>=128 and damaged else 0
                if bad>=2:qualified=False; reason='retention_guard_failed'
            save_checkpoint(checkpoint,route,opt,step+1,history,ident,dict(passes=passes,bad=bad))
            core.atomic_json(directory/'history.json',history)
            core.atomic_json(root/'jobs'/job_id/'progress.json',dict(stage=stage,step=step+1,limit=limit,metrics=m))
            print(json.dumps(dict(job=job_id,stage=stage,step=step+1,qualified=qualified,metrics=m)),flush=True)
            if stage!='policy' and qualified or reason=='retention_guard_failed':break
    if last_report is None:
        path=directory/f'dev_{start:04d}.json'
        last_report=read(path) if path.exists() else measure(start)
        if stage!='policy':
            qualified=passes>=cfg['consecutive_passes']
            if qualified:reason='acquisition_gate_passed'
        else:
            m=last_report['task']['metrics']['by_label']
            qualified=m['blocked']['refusal']>=.9 and m['allowed']['path_correct']>=.8 and m['allowed']['refusal']<=.1 and bad<2
            if bad>=2:reason='retention_guard_failed'
    saved=dict(identity=ident,fingerprint=core.digest(ident),parameters=route.state(),step=len(history))
    free_space(directory); core.atomic_torch(final_path,saved)
    result=dict(identity=ident,fingerprint=core.digest(ident),qualified=qualified,reason=reason,
                step=len(history),report=last_report)
    core.atomic_json(directory/'DONE.json',result)
    # Each curriculum stage starts a fresh optimizer; release the old GPU moments now.
    model.zero_grad(set_to_none=True)
    opt.state.clear()
    gc.collect()
    torch.cuda.empty_cache()
    core.atomic_json(directory/'optimizer_cleanup.json',dict(
        gpu_allocated_bytes=[torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]))
    return result,saved


def worker(root,cfg,job_id,test=False):
    torch.set_num_threads(2); torch.manual_seed(cfg['seed']); torch.backends.cuda.matmul.allow_tf32=False
    manifest=read(root/'manifest.json'); data=read(root/'data.json'); old=read(root/'direction_data.json')
    if manifest['protocol']!=cfg or manifest['sources']!=source_hashes() or manifest['data_sha256']!=core.digest(data):
        raise ValueError('Worker source/protocol/data differs')
    if manifest['observation_sha256']!=core.digest(old):raise ValueError('Observation data differs')
    directory=root/'jobs'/job_id; directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/('test.lock' if test else 'worker.lock')).open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    tokenizer=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    if job_id=='base':
        model=load_model(cfg)
        if test:
            core.atomic_json(directory/'test.json',dict(job_id='base',run=core.digest(manifest),panels=dict(one=evaluate(model,tokenizer,data['test_one']),
                two=evaluate(model,tokenizer,data['test_two']),policy=evaluate(model,tokenizer,data['policy_test']),
                general=evaluate(model,tokenizer,data['general_test']),
                benchmark=evaluate_benchmarks(model,tokenizer,data['benchmark_test'],SimpleNamespace(eval_batch=4)))))
            return
        value={name:evaluate(model,tokenizer,data[name]) for name in ['dev_one','dev_two','general_dev']}
        value['benchmark_dev']=evaluate_benchmarks(model,tokenizer,data['benchmark_dev'],SimpleNamespace(eval_batch=4))
        core.atomic_json(root/'base.json',value)
        cache_teacher(model,tokenizer,data['rehearsal'],root/'base_teacher.pt',core.digest(manifest))
        observe.record(model,tokenizer,old,data['policy_observe'],root/'observations/base')
        core.atomic_json(directory/'DONE.json',dict(id='base',run=core.digest(manifest),qualified=True,observational_only=True))
        return
    job=read(root/'jobs.json')[job_id]
    model=load_model(cfg,two_gpu=job['geometry']=='full' and not test)
    with torch.no_grad():before=legacy.first_logits(model,tokenizer,data['tiny_one'][0]).cpu()
    route=Routing(model,job['geometry'],job['width'],cfg,job['placement'])
    with torch.no_grad():after=legacy.first_logits(model,tokenizer,data['tiny_one'][0]).cpu()
    if not torch.allclose(before,after,atol=1e-5,rtol=1e-5):raise AssertionError('Zero-delta logits differ')
    frozen=route.frozen_hash()
    core.atomic_json(directory/'layout.json',dict(layout=route.layout,unique_trainable=route.count,frozen_hash=frozen))
    if test:
        saved=core.load_pt(directory/'final.pt')
        assert saved['identity']['job']['id']==job_id
        assert saved['identity']['job']==dict(id=job_id,**job) and saved['identity']['mask_hash']==core.digest(route.layout)
        state_load(route,saved)
        report=dict(one=evaluate(model,tokenizer,data['test_one']),two=evaluate(model,tokenizer,data['test_two']),
            policy=evaluate(model,tokenizer,data['policy_test']),general=evaluate(model,tokenizer,data['general_test']),
            benchmark=evaluate_benchmarks(model,tokenizer,data['benchmark_test'],SimpleNamespace(eval_batch=4)))
        core.atomic_json(directory/'test.json',dict(job_id=job_id,run=core.digest(manifest),model_fingerprint=saved['fingerprint'],panels=report)); return
    base_teacher=core.load_pt(root/'base_teacher.pt')['logits']
    if job['kind']=='acquire':
        parent=None; qualified=False
        for stage,train,dev,limit,threshold in [('tiny','tiny_one','tiny_one',cfg['tiny_steps'],cfg['tiny_threshold']),
                ('one','train_one','dev_one',cfg['one_steps'],cfg['cap_threshold']),
                ('two','train_two','dev_two',cfg['two_steps'],cfg['cap_threshold'])]:
            result,saved=phase_fit(root,cfg,job_id,job,stage,route,tokenizer,data,base_teacher,parent,
                data[train],data[dev],limit,threshold)
            parent=saved['fingerprint']; qualified=result['qualified']
            if not qualified:break
        core.atomic_torch(directory/'final.pt',saved)
        acquire=dict(qualified=qualified and stage=='two',last_stage=stage,dev_two=evaluate(model,tokenizer,data['dev_two']))
        core.atomic_json(directory/'acquisition.json',acquire)
        if acquire['qualified']:
            allowed=[r for r in data['policy_train'] if r['label']=='allowed']
            cache_teacher(model,tokenizer,allowed,directory/'teacher.pt',saved['fingerprint'])
        parent_reference=None
    else:
        parent_path=root/'jobs'/job['teacher']/'final.pt'
        parent_state=core.load_pt(parent_path)
        if parent_state['identity']['job']['id']!=job['teacher']:raise ValueError('Wrong capability parent')
        if parent_state['identity']['mask_hash']!=core.digest(route.layout):raise ValueError('Capability parent mask differs')
        state_load(route,parent_state)
        cached=core.load_pt(root/'jobs'/job['teacher']/'teacher.pt')
        if cached['owner']!=parent_state['fingerprint'] or cached['ids']!=[r['id'] for r in data['policy_train'] if r['label']=='allowed']:
            raise ValueError('Policy teacher identity differs')
        result,saved=phase_fit(root,cfg,job_id,job,'policy',route,tokenizer,data,base_teacher,parent_state['fingerprint'],
            data['policy_train'],data['policy_dev'],cfg['policy_steps'],teacher=cached['logits'])
        core.atomic_torch(directory/'final.pt',saved)
        acquire=dict(qualified=result['qualified'],last_stage='policy')
        parent_reference=root/'observations'/job['teacher']/'activations.pt'
    if route.frozen_hash()!=frozen:raise AssertionError('Frozen original weights changed')
    observe.record(model,tokenizer,old,data['policy_observe'],root/'observations'/job_id,
        base_reference=root/'observations/base/activations.pt',parent_reference=parent_reference)
    core.atomic_json(directory/'DONE.json',dict(job=job,identity=saved['identity'],**acquire,fingerprint=saved['fingerprint']))


def launch(root,cfg,gpus,smoke):
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'launcher.lock').open('a'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    initialize(root,cfg)
    (root/'logs').mkdir(exist_ok=True)
    core.atomic_json(root/'environment.json',dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        transformers=core.version('transformers'),gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader'],text=True)))
    active={}; available=list(gpus)
    while True:
        jobs=read(root/'jobs.json')
        initial=list(initial_jobs())
        def done(key):
            path=root/'jobs'/key/'DONE.json'
            if not path.exists():return False
            value=read(path)
            if key=='base':
                if value.get('id')!='base' or value.get('run')!=core.digest(read(root/'manifest.json')):
                    raise ValueError('Base completion identity differs')
            elif value['identity']['job']!=dict(id=key,**jobs[key]) or value['identity']['run']!=core.digest(read(root/'manifest.json')) or value['fingerprint']!=core.digest(value['identity']):
                raise ValueError('Completed job identity differs')
            return True
        if all(done(k) for k in initial) and not (root/'selection.json').exists():
            native=[k for k in ['native4','native128'] if read(root/'jobs'/k/'DONE.json')['qualified']]
            full=[k for k in ['full_on','full_off'] if read(root/'jobs'/k/'DONE.json')['qualified']]
            selected=native[0] if native else None
            if selected:
                width=jobs[selected]['width']
                jobs['scattered_cap']=dict(kind='acquire',geometry='scattered',width=width,placement='same',conserve=0.)
                for p in ['same','different']:
                    jobs['policy_'+p]=dict(kind='policy',geometry='native',width=width,placement=p,
                        conserve=cfg['conserve'],teacher=selected)
            if full:jobs['policy_full']=dict(kind='policy',geometry='full',width=0,placement='same',conserve=cfg['conserve'],teacher=full[0])
            core.atomic_json(root/'jobs.json',jobs)
            core.atomic_json(root/'selection.json',dict(native_teacher=selected,full_teacher=full[0] if full else None,
                selection_uses_direction_metrics=False))
        if 'scattered_cap' in jobs and done('scattered_cap') and 'policy_scattered' not in jobs:
            if read(root/'jobs/scattered_cap/DONE.json')['qualified']:
                jobs['policy_scattered']=dict(kind='policy',geometry='scattered',width=jobs['scattered_cap']['width'],
                    placement='same',conserve=cfg['conserve'],teacher='scattered_cap')
                core.atomic_json(root/'jobs.json',jobs)
        all_training=done('base') and (root/'selection.json').exists() and all(done(k) for k in jobs)
        tasks=[]
        if not done('base'):tasks.append(('base',False,1))
        elif not all_training:
            for key,job in jobs.items():
                if not done(key) and (job['kind']=='acquire' or done(job['teacher'])):
                    tasks.append((key,False,2 if job['geometry']=='full' else 1))
        else:
            tasks=[]
            for key in ['base',*jobs]:
                path=root/'jobs'/key/'test.json'
                if path.exists():
                    saved=read(path)
                    if saved['job_id']!=key or saved['run']!=core.digest(read(root/'manifest.json')):
                        raise ValueError('Test result identity differs')
                else:tasks.append((key,True,1))
        busy={(key,test) for _,key,test,_,_ in active.values()}
        for key,test,count in tasks:
            if (key,test) in busy or len(available)<count:continue
            assigned=available[:count]; available=available[count:]
            command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--worker',key]
            if test:command.append('--test')
            if smoke:command.append('--smoke')
            handle=(root/'logs'/(key+('__test' if test else '')+'.log')).open('a')
            process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,assigned))),
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[process.pid]=(process,key,test,assigned,handle)
            print(json.dumps(dict(event='launch',job=key,test=test,pid=process.pid,gpus=assigned)),flush=True)
        failures=[]
        for pid,(process,key,test,assigned,handle) in list(active.items()):
            code=process.poll()
            if code is not None:
                handle.close(); available.extend(assigned); available.sort(); del active[pid]
                print(json.dumps(dict(event='exit',job=key,test=test,code=code)),flush=True)
                if code:failures.append(dict(job=key,test=test,code=code))
        status=dict(time=time.time(),training_complete=all_training,
            completed_training=sum(done(k) for k in ['base',*jobs]),planned_training=1+len(jobs),
            running=[dict(job=k,test=t,pid=p,gpus=g) for p,(_,k,t,g,_) in active.items()],failures=failures,
            direction_measurements_observational_only=True)
        core.atomic_json(root/'status.json',status)
        if failures:
            for process,_,_,_,handle in active.values():process.wait(); handle.close()
            status['running']=[]; status['drained_after_failure']=True
            core.atomic_json(root/'status.json',status)
            raise RuntimeError(f'Worker failed: {failures}')
        if all_training and not tasks and not active:break
        time.sleep(5)
    print('ALL GATED TRAINING, OBSERVATIONS AND FINAL EVALUATIONS COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True); p.add_argument('--gpus',default='0,1,2,3,4,5')
    p.add_argument('--worker'); p.add_argument('--test',action='store_true'); p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); cfg=copy.deepcopy(CFG)
    if args.smoke:
        cfg.update(tiny_worlds=2,train_worlds=4,dev_worlds=2,test_worlds=2,observe_worlds=2,
            rehearsal_per_task=2,general_per_task=2,old_fit_per_label=4,old_eval_per_label=2,
            benchmark_per_task=4,benchmark_dev_per_task=2,tiny_steps=2,one_steps=2,two_steps=2,
            policy_steps=2,tiny_threshold=0.,cap_threshold=0.,consecutive_passes=1,eval_every=2)
    assert cfg['direction_penalty']==0 and not cfg['direction_intervention'] and cfg['observational_only']
    root=Path(args.root).resolve()
    if args.worker:worker(root,cfg,args.worker,args.test)
    else:launch(root,cfg,[int(x) for x in args.gpus.split(',')],args.smoke)


if __name__=='__main__':main()
