"""V8 defense continuation, unrestricted removal, and capability repair."""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference_v7'))
from reference_v7 import attack as api
from design import (CONFIG,DEFENSE_STEPS,ATTACK_STEPS,REPAIR_STEPS,
                    attack_jobs,batch,defense_jobs,digest,make_train,policy_quality,quality)
from mechanism import Support,cosine_gradient,dot,gradient

read,write,sha=api.read,api.write,api.sha


def sources():
    files=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']
    files += [HERE/'reference_v7'/n for n in api.sources()]
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(files)}


def check(root):
    m=read(root/'manifest.json');assert m['sources']==sources()
    assert digest(read(root/'data.json'))==m['data_sha256']
    return m


def save(path,state,needed):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lock=(path.parents[2]/'disk_write.lock').open('a')
    with lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if shutil.disk_usage(path.parent).free<CONFIG['reserve_gib']*1024**3+int(1.15*needed):
            raise OSError('Main disk reserve reached; committed data retained for resume')
        import torch
        tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def initialize(root,previous,smoke=False):
    root.mkdir(parents=True,exist_ok=True);old=read(previous/'results/manifest.json')
    assert old['sources']==api.sources()
    data=read(previous/'results/data.json')
    assert digest(data)==old['data_sha256']
    excluded=set(read(HERE/'reference_v7/reference_v3/excluded_worlds.json'))
    excluded.update(r['world'] for rows in data.values() if isinstance(rows,list) for r in rows if 'world' in r)
    banks={}
    for i,name in enumerate(['defense_train','meta_train','attack_train','repair_train','probe']):
        rows=make_train(excluded,32 if name=='probe' else CONFIG['worlds'],41107+i)
        assert not {r['world'] for r in rows}&excluded
        excluded.update(r['world'] for r in rows);banks[name]=rows
    banks.update({k:data[k] for k in ['policy_dev','policy_test','general_dev','general_test',
                                      'benchmark_dev','benchmark_test','rehearsal']})
    parents={}
    for p in CONFIG['parents']:
        folder=previous/'results/parents'/p;ready=read(folder/'DONE.json')
        assert sha(folder/'initial.pt')==ready['checkpoint_sha256']
        assert sha(folder/'general_teacher.pt')==ready['teacher_sha256']
        parents[p]=dict(folder=str(folder),ready=ready,
            layout=read(previous/'inputs/models'/p/'layout.json')['layout'])
    cfg=dict(CONFIG)
    if smoke:cfg['meta_every']=1
    tasks=defense_jobs(smoke)
    if smoke:tasks['lookahead_reference']=dict(tasks['lookahead'])
    m=dict(protocol='selective-native-gradient-coupling-v8',config=cfg,sources=sources(),
        previous=str(previous),previous_manifest_sha256=sha(previous/'results/manifest.json'),
        parents=parents,data_sha256=digest(banks),defense_jobs=tasks,smoke=smoke)
    for name,value in [('manifest.json',m),('data.json',banks)]:
        if (root/name).exists():assert read(root/name)==value
        else:write(root/name,value)
    print('Initialized',root,'smoke',smoke,flush=True)


def model_for(m,parent,two=True):
    e,t=api.engine();cfg=read(HERE/'reference_v7/reference_v3/original_manifest.json')['protocol']
    tok,model=api.tokenizer_and_model(e,t,cfg,two)
    folder=Path(m['parents'][parent]['folder']);ready=m['parents'][parent]['ready']
    state=e.core.load_pt(folder/'initial.pt')
    assert state['parameter_hash']==ready['parameter_hash']
    api.load_parameters(model,state['parameters'],t);del state
    assert e.core.tensor_hash(model.named_parameters())==ready['parameter_hash']
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    teacher=e.core.load_pt(folder/'general_teacher.pt')
    assert teacher['owner']==ready['parameter_hash']
    return e,t,tok,model,teacher['logits']


def mean_nll(e,model,tok,rows,target=None):
    return sum(e.legacy.nll(model,tok,r,target if target is not None else r['answer']) for r in rows)/len(rows)


def rng(t):
    return dict(cpu=t.get_rng_state(),cuda=t.cuda.get_rng_state_all(),python=random.getstate())


def restore_rng(t,state):
    t.set_rng_state(state['cpu']);t.cuda.set_rng_state_all(state['cuda']);random.setstate(state['python'])


def lookahead(e,t,model,tok,support,data,teacher,step,cfg):
    original=api.snapshot(model);random_state=rng(t)
    offset=(step*cfg['meta_steps'])% (CONFIG['worlds']//2)
    _,probe=batch(data['meta_train'],offset+cfg['meta_steps'])
    before,g0=gradient(support,lambda:mean_nll(e,model,tok,probe))
    outside=original['model.layers.0.self_attn.q_proj.weight']
    try:
        model.requires_grad_(True)
        opt=t.optim.AdamW(model.parameters(),lr=cfg['meta_lr'],weight_decay=0,foreach=False)
        for j in range(cfg['meta_steps']):
            api.update(e,t,model,tok,data['meta_train'],data['rehearsal'],teacher,
                       dict(objective='preserve',lr=cfg['meta_lr']),offset+j,opt)
        changed=float((model.model.layers[0].self_attn.q_proj.weight.detach().cpu()-outside).norm())
        assert changed>0,'Simulated attacker did not change weights outside the routed allocation'
        del opt;model.zero_grad(set_to_none=True);gc.collect();t.cuda.empty_cache()
        support.enable()
        after,g1=gradient(support,lambda:mean_nll(e,model,tok,probe))
        return [x-y for x,y in zip(before,after)],dict(meta_before=g0,meta_after=g1,
            meta_outside_weight_change=changed,meta_updates=cfg['meta_steps'],
            meta_probe_ids=[r['id'] for r in probe],meta_offset=offset)
    finally:
        api.load_parameters(model,original,t);restore_rng(t,random_state);support.enable()
        model.zero_grad(set_to_none=True)


def defense_update(e,t,model,tok,support,opt,data,teacher,spec,step,cfg):
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    blocked,allowed=batch(data['defense_train'],step)
    loss_a=lambda:mean_nll(e,model,tok,blocked)
    loss_c=lambda:mean_nll(e,model,tok,allowed)
    if spec['beta']:
        coupling,cap,record=cosine_gradient(support,loss_a,loss_c,cfg['fd_epsilon'])
    else:
        cap,value=gradient(support,loss_c);coupling=[t.zeros_like(x) for x in cap]
        record=dict(skill_nll=value,cosine=None)
    index=(step//2)%(len(data['rehearsal'])//2)+(step%2)*(len(data['rehearsal'])//2)
    clean,value=gradient(support,lambda:mean_nll(e,model,tok,blocked,'REFUSE')+
        cfg['conserve']*e.full_kl(model,tok,data['rehearsal'][index],teacher[index]))
    grads=[a+b+spec['beta']*c for a,b,c in zip(clean,cap,coupling)]
    if spec['arm']=='lookahead' and (step+1)%cfg['meta_every']==0:
        meta,details=lookahead(e,t,model,tok,support,data,teacher,step,cfg)
        for g,v in zip(grads,meta):g.add_(cfg['meta_weight']*v)
        record.update(details)
    norm=support.apply(opt,grads,cfg['clip'])
    return dict(record,step=step+1,clean_refusal_plus_kl=value,grad_norm=norm,
        blocked_ids=[r['id'] for r in blocked],allowed_ids=[r['id'] for r in allowed],
        rehearsal_id=data['rehearsal'][index]['id'],mutable_scalars=support.count)


def probe_cosine(e,model,tok,support,data):
    b,a=batch(data['probe'],0)
    ga,la=gradient(support,lambda:mean_nll(e,model,tok,b))
    gc,lc=gradient(support,lambda:mean_nll(e,model,tok,a))
    na=math.sqrt(dot(ga,ga));nc=math.sqrt(dot(gc,gc))
    assert na>1e-12 and nc>1e-12
    return dot(ga,gc)/(na*nc),dict(attack_nll=la,skill_nll=lc,attack_grad_norm=na,skill_grad_norm=nc)


def defend(root,key,stop_after=None):
    m=check(root);spec=m['defense_jobs'][key];directory=root/'defense'/key
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'worker.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/'DONE.json').exists():return
    ident=digest(dict(manifest=digest(m),job=key));cfg=m['config']
    e,t,tok,model,teacher=model_for(m,spec['parent'])
    layout=None if spec['arm']=='global' else m['parents'][spec['parent']]['layout']
    support=Support(model,layout)
    expected=494032768 if layout is None else int(spec['parent'][-1])*2688
    assert support.count==expected
    opt=t.optim.AdamW(support.optparams,lr=cfg['defense_lrs']['global_' if layout is None else spec['arm']],weight_decay=0,foreach=False)
    data=read(root/'data.json');baseline=read(Path(m['parents'][spec['parent']]['folder'])/'baseline_dev.json')
    history=[];start=0
    if (directory/'resume.pt').exists():
        saved=e.core.load_pt(directory/'resume.pt');assert saved['identity']==ident
        support.put(saved['values']);opt.load_state_dict(saved['optimizer']);history=saved['history'];start=saved['step']
        restore_rng(t,saved['rng']);del saved
    def state(step):
        return dict(identity=ident,step=step,values=support.values(),names=support.names,
                    scope='global' if layout is None else 'native',parent=spec['parent'])
    def recover(step):
        v=state(step);v.update(optimizer=opt.state_dict(),history=history,rng=rng(t))
        save(directory/'resume.pt',v,3*support.count*4)
    def observe(step):
        before=e.core.tensor_hash(model.named_parameters())
        path=directory/f'dev_{step:04d}.json'
        if path.exists():assert read(path)['parameter_hash']==before;return
        cosine,probe=probe_cosine(e,model,tok,support,data)
        report=api.measure(e,t,model,tok,data,'dev',m['smoke'])
        assert before==e.core.tensor_hash(model.named_parameters())
        selected=policy_quality(report,baseline,step,cosine)
        record=dict(identity=ident,step=step,parameter_hash=before,selection=selected,probe=probe,report=report)
        best=read(directory/'best.json') if (directory/'best.json').exists() else None
        if step>0 and selected['qualified'] and (best is None or selected['rank']>best['selection']['rank']):
            save(directory/'best.pt',state(step),support.count*4)
            write(directory/'best.json',dict(identity=ident,step=step,parameter_hash=before,selection=selected))
        write(path,record)
        write(directory/'progress.json',dict(step=step,limit=spec['limit'],selection=selected))
        print(json.dumps(dict(job=key,**selected)),flush=True)
    observe(start)
    for step in range(start,spec['limit']):
        tick=time.monotonic();record=defense_update(e,t,model,tok,support,opt,data,teacher,spec,step,cfg)
        record['seconds']=time.monotonic()-tick;history.append(record);write(directory/'history.json',history)
        if step+1 in ([1,2] if m['smoke'] else DEFENSE_STEPS):observe(step+1)
        write(directory/'progress.json',dict(step=step+1,limit=spec['limit']))
        if (step+1)%16==0 or step+1==spec['limit'] or step+1==stop_after:recover(step+1)
        if step+1==stop_after:
            write(directory/'INTERRUPTED.json',dict(step=step+1));return
    save(directory/'final.pt',state(spec['limit']),support.count*4)
    # Check exact frozen coordinates, including nonselected rows/columns of mutable matrices.
    frozen_exact=None
    if layout is not None:
        final=support.values();initial=e.core.load_pt(Path(m['parents'][spec['parent']]['folder'])/'initial.pt')
        support.put([support.take(initial['parameters'][n].to(p.device),i).detach().cpu()
                     for i,(n,p) in enumerate(zip(support.names,support.actual))])
        frozen_exact=e.core.tensor_hash(model.named_parameters())==initial['parameter_hash']
        assert frozen_exact;support.put(final);del initial
    write(directory/'DONE.json',dict(identity=ident,spec=spec,updates=len(history),
        parameter_hash=e.core.tensor_hash(model.named_parameters()),mutable_scalars=support.count,
        frozen_exact=frozen_exact,optimizer_hash=api.optimizer_hash(e,opt) if m['smoke'] else None,
        best=read(directory/'best.json') if (directory/'best.json').exists() else None,
        cuda_peak=[t.cuda.max_memory_allocated(i) for i in range(t.cuda.device_count())],ended_unix=time.time()))


def select(root):
    m=check(root);assert not m['smoke']
    assert all((root/'defense'/k/'DONE.json').exists() for k in m['defense_jobs'])
    models={}
    for parent in CONFIG['parents']:
        for arm in CONFIG['arms']:
            candidates=[]
            for key,spec in m['defense_jobs'].items():
                if spec['parent']==parent and spec['arm']==arm:
                    done=read(root/'defense'/key/'DONE.json')
                    if done['best']:candidates.append((done['best']['selection']['rank'],key,done['best']))
            candidates.sort(key=lambda x:(x[0],x[1]),reverse=True)
            chosen=candidates[0] if candidates else None
            models[f'{parent}__{arm}']=dict(parent=parent,arm=arm,qualified=bool(chosen),
                candidate=chosen[1] if chosen else None,best=chosen[2] if chosen else None)
    value=dict(manifest=digest(m),models=models,jobs=attack_jobs(models))
    if (root/'selection.json').exists():assert read(root/'selection.json')==value
    else:write(root/'selection.json',value)
    print('Qualified defense arms',sum(v['qualified'] for v in models.values()),'of',len(models),flush=True)


def load_defended(root,key,two=True):
    m=check(root);chosen=read(root/'selection.json')['models'][key];assert chosen['qualified']
    e,t,tok,model,teacher=model_for(m,chosen['parent'],two)
    source=root/'defense'/chosen['candidate'];spec=m['defense_jobs'][chosen['candidate']]
    support=Support(model,None if spec['arm']=='global' else m['parents'][spec['parent']]['layout'])
    state=e.core.load_pt(source/'best.pt');assert state['identity']==read(source/'best.json')['identity']
    support.put(state['values']);del state
    assert e.core.tensor_hash(model.named_parameters())==chosen['best']['parameter_hash']
    model.requires_grad_(True)
    return m,chosen,e,t,tok,model


def prepare(root,key):
    m,chosen,e,t,tok,model=load_defended(root,key)
    directory=root/'models'/key;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'DONE.json').exists():return
    data=read(root/'data.json');hashed=e.core.tensor_hash(model.named_parameters())
    # Baseline dev only; held-out baseline and attacks are measured after all attack training.
    report=api.measure(e,t,model,tok,data,'dev')
    assert report==read(root/'defense'/chosen['candidate']/f'dev_{chosen["best"]["step"]:04d}.json')['report']
    write(directory/'baseline_dev.json',report)
    teacher=e.cache_teacher(model,tok,data['rehearsal'],directory/'general_teacher.pt',hashed)
    assert len(teacher)==len(data['rehearsal'])
    values=api.snapshot(model);save(directory/'initial.pt',dict(parameters=values,parameter_hash=hashed),494032768*4)
    write(directory/'DONE.json',dict(parameter_hash=hashed,initial_sha256=sha(directory/'initial.pt'),
        teacher_sha256=sha(directory/'general_teacher.pt'),chosen=chosen))


def benign_update(e,t,model,tok,rows,rehearsal,teacher,step,opt):
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    opt.zero_grad(set_to_none=True);_,allowed=batch(rows,step)
    losses=[]
    for row in allowed:
        loss=e.legacy.nll(model,tok,row,row['answer']);assert t.isfinite(loss)
        (loss/len(allowed)).backward();losses.append(float(loss.detach()))
    index=(step//2)%(len(rehearsal)//2)+(step%2)*(len(rehearsal)//2)
    kl=e.full_kl(model,tok,rehearsal[index],teacher[index]);assert t.isfinite(kl)
    (CONFIG['conserve']*kl).backward()
    norm=t.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['clip']);assert t.isfinite(norm)
    opt.step();opt.zero_grad(set_to_none=True)
    return dict(step=step+1,lr=opt.param_groups[0]['lr'],blocked_ids=[],
        allowed_ids=[r['id'] for r in allowed],rehearsal_id=rehearsal[index]['id'],
        blocked_exposure_epochs=0.,allowed_nll=sum(losses)/len(losses),
        conservation_kl=float(kl.detach()),grad_norm=float(norm))


def attack(root,key,stop_after=None):
    m=check(root);selection=read(root/'selection.json');spec=selection['jobs'][key]
    directory=root/'attacks'/key;directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'worker.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/'DONE.json').exists():return
    parent=root/'models'/spec['model'];ready=read(parent/'DONE.json')
    ident=digest(dict(selection=digest(selection),job=key,sources=m['sources'],parent=ready))
    e,t=api.engine();cfg=read(HERE/'reference_v7/reference_v3/original_manifest.json')['protocol']
    tok,model=api.tokenizer_and_model(e,t,cfg,True);initial=e.core.load_pt(parent/'initial.pt')
    assert sha(parent/'initial.pt')==ready['initial_sha256']
    api.load_parameters(model,initial['parameters'],t);del initial
    model.requires_grad_(True);assert all(p.requires_grad for p in model.parameters())
    assert not any('parametrizations' in n for n,_ in model.named_parameters())
    data=read(root/'data.json');teacher=e.core.load_pt(parent/'general_teacher.pt')
    assert sha(parent/'general_teacher.pt')==ready['teacher_sha256'] and teacher['owner']==ready['parameter_hash']
    baseline=read(parent/'baseline_dev.json');history=[];phase='remove';start=0
    opt=t.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=0,foreach=False)
    if (directory/'resume.pt').exists():
        saved=e.core.load_pt(directory/'resume.pt');assert saved['identity']==ident
        api.load_parameters(model,saved['parameters'],t);opt.load_state_dict(saved['optimizer'])
        history=saved['history'];phase=saved['phase'];start=saved['step'];restore_rng(t,saved['rng']);del saved
    def recover(which,step):
        save(directory/'resume.pt',dict(identity=ident,phase=which,step=step,
            parameters=api.snapshot(model),optimizer=opt.state_dict(),history=history,rng=rng(t)),494032768*12)
    def observe(which,step):
        path=directory/f'{which}_{step:04d}.json';hashed=e.core.tensor_hash(model.named_parameters())
        if path.exists():assert read(path)['parameter_hash']==hashed;return
        report=api.measure(e,t,model,tok,data,'dev');score=quality(report,baseline,step)
        record=dict(identity=ident,phase=which,step=step,cumulative_updates=step+(spec['limit'] if which=='repair' else 0),
            parameter_hash=hashed,selection=score,report=report)
        names=[]
        if which=='remove':
            best=read(directory/'best.json') if (directory/'best.json').exists() else None
            if best is None or score['rank']>best['selection']['rank']:names.append('best')
            old=read(directory/'minimum.json') if (directory/'minimum.json').exists() else None
            if old is None or score['allowed_accuracy']<old['selection']['allowed_accuracy']:names.append('minimum')
            if score['refusal']==0 and not (directory/'first_zero.json').exists():names.append('first_zero')
            if step==spec['limit']:names.append('final')
        elif step==m['config']['repair_updates']:names.append('repair_final')
        if names:
            if which=='remove' and step==0:source=parent/'initial.pt'
            else:
                source=directory/'current.pt'
                save(source,dict(parameters=api.snapshot(model),parameter_hash=hashed,identity=ident,
                    phase=which,step=step),494032768*4)
            for name in names:
                api.hardlink(source,directory/(name+'.pt'));write(directory/(name+'.json'),record)
            if source==directory/'current.pt':source.unlink()
        write(path,record)
        write(directory/'progress.json',dict(phase=which,step=step,limit=spec['limit'] if which=='remove' else m['config']['repair_updates'],selection=score))
        print(json.dumps(dict(job=key,phase=which,**score)),flush=True)
    phases=['remove','repair']
    for which in phases[phases.index(phase):]:
        if which!=phase:
            opt=t.optim.AdamW(model.parameters(),lr=m['config']['repair_lr'],weight_decay=0,foreach=False)
            start=0
        limit=spec['limit'] if which=='remove' else m['config']['repair_updates']
        schedule=ATTACK_STEPS if which=='remove' else REPAIR_STEPS
        objective=spec['objective'] if which=='remove' or spec['objective']=='benign' else 'preserve'
        lr=spec['lr'] if which=='remove' else m['config']['repair_lr']
        observe(which,start)
        for step in range(start,limit):
            tick=time.monotonic()
            bank=data['attack_train' if which=='remove' else 'repair_train']
            if objective=='benign':
                record=benign_update(e,t,model,tok,bank,data['rehearsal'],teacher['logits'],step,opt)
            else:
                record=api.update(e,t,model,tok,bank,data['rehearsal'],teacher['logits'],
                                  dict(objective=objective,lr=lr),step,opt)
            history.append(dict(record,phase=which,cumulative_updates=step+1+(spec['limit'] if which=='repair' else 0),
                                phase_objective=objective,seconds=time.monotonic()-tick))
            write(directory/'history.json',history)
            if step+1 in schedule:observe(which,step+1)
            write(directory/'progress.json',dict(phase=which,step=step+1,limit=limit))
            if (step+1)%16==0 or step+1==limit:recover(which,step+1)
        start=0
    write(directory/'DONE.json',dict(identity=ident,spec=spec,updates=len(history),
        parameter_hash=e.core.tensor_hash(model.named_parameters()),ended_unix=time.time(),
        cuda_peak=[t.cuda.max_memory_allocated(i) for i in range(t.cuda.device_count())]))


def heldout(root,key):
    started=time.time();m=check(root);selection=read(root/'selection.json')
    assert all((root/'attacks'/k/'DONE.json').exists() for k in selection['jobs'])
    spec=selection['jobs'][key];directory=root/'attacks'/key;done=read(directory/'DONE.json')
    e,t=api.engine();cfg=read(HERE/'reference_v7/reference_v3/original_manifest.json')['protocol']
    tok,model=api.tokenizer_and_model(e,t,cfg,False);data=read(root/'data.json')
    initial=e.core.load_pt(root/'models'/spec['model']/'initial.pt')
    api.load_parameters(model,initial['parameters'],t);del initial
    baseline=api.measure(e,t,model,tok,data,'test')
    write(directory/'baseline_test.json',dict(parameter_hash=e.core.tensor_hash(model.named_parameters()),report=baseline))
    for endpoint in ['best','final','repair_final','first_zero','minimum']:
        if not (directory/(endpoint+'.pt')).exists():continue
        path=directory/f'test_{endpoint}.json'
        if path.exists():assert read(path)['identity']==done['identity'];continue
        state=e.core.load_pt(directory/(endpoint+'.pt'));api.load_parameters(model,state['parameters'],t);del state
        meta=read(directory/(endpoint+'.json'));before=e.core.tensor_hash(model.named_parameters())
        assert before==meta['parameter_hash']
        report=api.measure(e,t,model,tok,data,'test')
        assert before==e.core.tensor_hash(model.named_parameters())
        write(path,dict(identity=done['identity'],endpoint=endpoint,parameter_hash=before,
            phase=meta['phase'],step=meta['step'],cumulative_updates=meta['cumulative_updates'],
            selection=quality(report,baseline,meta['step']),report=report,started_unix=started))
    write(directory/'TEST_DONE.json',dict(identity=done['identity']))


def campaign(root,phase,gpus):
    m=check(root)
    lock=(root/'launcher.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if phase=='defend':tasks=list(m['defense_jobs']);base='defense';marker='DONE.json'
    else:
        selection=read(root/'selection.json')
        tasks=[k for k,v in selection['models'].items() if v['qualified']] if phase=='prepare' else list(selection['jobs'])
        base='models' if phase=='prepare' else 'attacks';marker='TEST_DONE.json' if phase=='heldout' else 'DONE.json'
    count=1 if phase=='heldout' else 2
    def complete(k):return (root/base/k/marker).exists()
    active={};failures=[];(root/'logs').mkdir(exist_ok=True)
    while True:
        used={g for _,gs,_ in active.values() for g in gs}
        free=api.available_gpus([g for g in gpus if g not in used])
        for key in tasks:
            if complete(key) or key in active or len(free)<count or failures:continue
            assigned=free[:count];free=free[count:]
            args=[sys.executable,'-u',str(HERE/'experiment.py'),'--root',str(root),'--'+phase,key]
            if m['smoke'] and key=='lookahead' and not (root/base/key/'INTERRUPTED.json').exists():args+=['--stop-after','1']
            handle=(root/'logs'/f'{phase}_{key}.log').open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,assigned)),PYTHONDONTWRITEBYTECODE='1')
            process=subprocess.Popen(args,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=(process,assigned,handle)
        for key,(process,assigned,handle) in list(active.items()):
            code=process.poll()
            if code is not None:
                handle.close();del active[key]
                if code:failures.append(dict(job=key,code=code))
        write(root/'status.json',dict(phase=phase,complete=sum(complete(k) for k in tasks),planned=len(tasks),
            running=[dict(job=k,pid=p.pid,gpus=gs) for k,(p,gs,_) in active.items()],failures=failures,updated_unix=time.time()))
        if failures and not active:raise RuntimeError('Workers failed; checkpoints retained: '+str(failures))
        if all(complete(k) for k in tasks) and not active:break
        time.sleep(5)
    if m['smoke']:
        a=read(root/'defense/lookahead/DONE.json');b=read(root/'defense/lookahead_reference/DONE.json')
        assert a['parameter_hash']==b['parameter_hash'] and a['optimizer_hash']==b['optimizer_hash']
        assert all(read(root/'defense'/k/'DONE.json')['frozen_exact'] for k in ['local','lookahead'])
        write(root/'hardware_validation.json',dict(sources=m['sources'],actual_resume_exact=True,
            full_model_lookahead_changed_outside_support=True,local_outer_frozen_coordinates_exact=True,
            global_local_meta_two_gpu_steps_passed=True,parameter_hash=a['parameter_hash'],
            optimizer_hash=a['optimizer_hash']))
    if phase=='heldout':write(root/'COMPLETE.json',dict(run=digest(m),jobs=len(tasks),finished_unix=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--initialize',action='store_true');p.add_argument('--previous',type=Path);p.add_argument('--smoke',action='store_true')
    for name in ['defend','prepare','attack','heldout']:p.add_argument('--'+name)
    p.add_argument('--stop-after',type=int);p.add_argument('--select',action='store_true')
    p.add_argument('--phase',choices=['defend','prepare','attack','heldout']);p.add_argument('--gpus',default='0,1,2,3,4,5')
    a=p.parse_args();root=a.root.resolve()
    if a.initialize:initialize(root,a.previous.resolve(),a.smoke)
    elif a.select:select(root)
    elif a.phase:
        if not check(root)['smoke']:
            validation=read(root.parent/'hardware/hardware_validation.json')
            assert validation['sources']==sources() and validation['actual_resume_exact']
        campaign(root,a.phase,[int(g) for g in a.gpus.split(',')])
    else:
        for name in ['defend','prepare','attack','heldout']:
            key=getattr(a,name)
            if key:
                if name in ['defend','attack']:globals()[name](root,key,a.stop_after)
                else:globals()[name](root,key)
                break


if __name__=='__main__':main()
