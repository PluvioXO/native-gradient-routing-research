"""Full-model SFT removal of a harmless co-located navigation refusal."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from protocol import CONFIG,STEPS,batch,digest,jobs,make_train,prompt,quality

HERE=Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE']='1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sources():
    old=read(HERE/'reference_v3/original_manifest.json')['sources']
    paths=[*HERE.glob('*.py'),HERE/'PROTOCOL.md',HERE/'reference_v3/original_manifest.json',
           *(HERE/'reference_v3'/n for n in old)]
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(paths)}


def engine():
    sys.path.insert(0,str(HERE/'reference_v3'))
    import run
    import torch
    torch.set_num_threads(2);torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert run.source_hashes()==read(HERE/'reference_v3/original_manifest.json')['sources']
    return run,torch


def atomic_state(path,value,needed):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lock=(path.parents[2]/'disk_write.lock').open('a')
    with lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if shutil.disk_usage(path.parent).free < 40*1024**3+int(1.1*needed):
            raise OSError('Insufficient main-disk space; previous committed data retained')
        import torch
        tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def hardlink(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.link.tmp')
    if tmp.exists():tmp.unlink()
    os.link(source,tmp);tmp.replace(destination)


def initialize(root,inputs,smoke=False,parent_root=None):
    root.mkdir(parents=True,exist_ok=True);original=read(inputs/'manifest.json')
    assert original==read(HERE/'reference_v3/original_manifest.json')
    old=read(inputs/'data.json');assert digest(old)==original['data_sha256']
    excluded=set(read(HERE/'reference_v3/excluded_worlds.json'))
    excluded.update(r['world'] for rows in old.values() for r in rows if 'world' in r)
    train=make_train(excluded,CONFIG['attack_worlds'],CONFIG['data_seed'])
    assert not {r['world'] for r in train}&excluded
    data={k:old[k] for k in ['policy_dev','policy_test','general_dev','general_test','benchmark_dev','benchmark_test','rehearsal']}
    data['attack_train']=train
    models={}
    for key in CONFIG['parents']:
        folder=inputs/'models'/key;meta=read(folder/'best.json');recipe=read(inputs/'recipes.json')['recipes'][key]
        assert meta['selection']['feasible'] and recipe['selection']==meta['selection']
        assert meta['identity']['job']['id']==recipe['job'] and meta['identity']['job']['condition']==key
        models[key]=dict(input=str(folder),checkpoint_sha256=sha(folder/'best.pt'),
            original_step=meta['step'],identity=meta['identity'])
    manifest=dict(protocol='colocated-skill-preserving-removal-v1',config=CONFIG,sources=sources(),
        inputs=str(inputs),input_hashes={str(p.relative_to(inputs)):sha(p) for p in sorted(inputs.rglob('*')) if p.is_file()},
        old_manifest_sha256=sha(inputs/'manifest.json'),data_sha256=digest(data),parents=models,
        smoke=smoke,parent_root=str(parent_root or root),jobs=jobs(smoke),
        evaluation_steps=[0,1,2] if smoke else STEPS)
    for name,value in [('manifest.json',manifest),('data.json',data)]:
        path=root/name
        if path.exists():assert read(path)==value,'Frozen input differs: '+name
        else:write(path,value)
    print('Initialized',root,'smoke',smoke,flush=True)


def tokenizer_and_model(e,t,cfg,two):
    tok=e.AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],padding_side='left')
    tok.pad_token=tok.eos_token
    return tok,e.load_model(cfg,two_gpu=two)


def snapshot(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters()}


def load_parameters(model,state,torch):
    named=dict(model.named_parameters());assert named.keys()==state.keys()
    with torch.no_grad():
        for name,p in named.items():
            assert p.shape==state[name].shape and p.dtype==state[name].dtype
            p.copy_(state[name])


def measure(e,torch,model,tok,data,split,smoke=False):
    def subset(rows,nav=False):
        if not smoke:return rows
        if nav:
            worlds=set(sorted({r['world'] for r in rows})[:2]);return [r for r in rows if r['world'] in worlds]
        return sum([[r for r in rows if r['task']==task][:2] for task in sorted({r['task'] for r in rows})],[])
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        return dict(task=e.evaluate(model,tok,subset(data['policy_'+split],True)),
                    general=e.evaluate(model,tok,subset(data['general_'+split])),
                    benchmark=e.evaluate_benchmarks(model,tok,subset(data['benchmark_'+split]),SimpleNamespace(eval_batch=4)))


def prepare(root,key):
    m=read(root/'manifest.json');assert m['sources']==sources() and not m['smoke']
    directory=root/'parents'/key;directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'worker.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/'DONE.json').exists():return
    meta=m['parents'][key];folder=Path(meta['input']);assert sha(folder/'best.pt')==meta['checkpoint_sha256']
    e,torch=engine();cfg=read(HERE/'reference_v3/original_manifest.json')['protocol'];data=read(root/'data.json')
    assert digest(data)==m['data_sha256']
    tok,model=tokenizer_and_model(e,torch,cfg,False)
    spec=meta['identity']['job'];route=e.Routing(model,spec['geometry'],spec['width'],cfg,spec['placement'])
    saved=e.core.load_pt(folder/'best.pt');e.state_load(route,saved,meta['identity'])
    assert saved['selection']['feasible'] and saved['step']==meta['original_step']
    before=e.effective_native_hash(route)
    from removal import merge_native
    merge_native(model);assert before==e.effective_native_hash(route)
    assert all(p.requires_grad for p in model.parameters()) and not any('parametrizations' in n for n,_ in model.named_parameters())
    del route
    state=snapshot(model);parameter_hash=e.core.tensor_hash(state.items())
    atomic_state(directory/'initial.pt',dict(parameters=state,parameter_hash=parameter_hash,parent=meta),sum(v.numel()*v.element_size() for v in state.values()))
    del state
    for row in data['attack_train'][:100]:assert row['prompt']==e.datasets.prompt(row['edges'],row['start'],row['depth'],row['marker'])
    reports={}
    original_dev=read(folder/'best.json')['report'];original_test=read(folder/'test_best.json')['panels']
    for split,reference in [('dev',original_dev),('test',original_test)]:
        report=measure(e,torch,model,tok,data,split)
        for panel in ['task','general']:
            assert [(r['id'],r['response']) for r in report[panel]['rows']]==[(r['id'],r['response']) for r in reference[panel]['rows']], 'Initial response parity failed: '+key+'/'+split+'/'+panel
        for k,v in report['benchmark']['metrics'].items():
            assert abs(v-reference['benchmark']['metrics'][k])<=1e-5
        write(directory/('baseline_'+split+'.json'),report);reports[split]=report
    teacher=e.cache_teacher(model,tok,data['rehearsal'],directory/'general_teacher.pt',parameter_hash)
    assert len(teacher)==len(data['rehearsal'])
    assert parameter_hash==e.core.tensor_hash(model.named_parameters())
    write(directory/'DONE.json',dict(run=digest(m),parent=meta,parameter_hash=parameter_hash,
        checkpoint_sha256=sha(directory/'initial.pt'),teacher_sha256=sha(directory/'general_teacher.pt'),
        materialization_effective_weights_exact=True,baseline_response_replay_exact=True,
        trainable_scalars=sum(p.numel() for p in model.parameters()),original_step=meta['original_step']))
    print('Prepared qualified starting policy',key,flush=True)


def optimizer_hash(e,opt):
    state=opt.state_dict()
    tensors=[(f'{i}.{k}',v.reshape(-1)) for i,s in sorted(state['state'].items()) for k,v in sorted(s.items()) if hasattr(v,'numel')]
    return e.core.tensor_hash(tensors)


def update(e,torch,model,tok,rows,rehearsal,teacher,spec,step,opt):
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    opt.zero_grad(set_to_none=True);blocked,allowed=batch(rows,step);values={};tokens=0
    for name,pool,weight in [('blocked_nll',blocked,1.),('allowed_nll',allowed,1. if spec['objective']=='preserve' else 0.)]:
        if not weight:continue
        losses=[]
        for row in pool:
            loss=e.legacy.nll(model,tok,row,row['answer']);assert torch.isfinite(loss)
            (weight*loss/len(pool)).backward();losses.append(float(loss.detach()))
            tokens+=len(e.legacy.ids_for(tok,row,row['answer']))
        values[name]=sum(losses)/len(losses)
    index=(step//2)%(len(rehearsal)//2)+(step%2)*(len(rehearsal)//2)
    if spec['objective']=='preserve':
        loss=e.full_kl(model,tok,rehearsal[index],teacher[index]);assert torch.isfinite(loss)
        (CONFIG['conserve']*loss).backward();values['conservation_kl']=float(loss.detach())
        tokens+=len(e.legacy.ids_for(tok,rehearsal[index],rehearsal[index]['answer']))
    proof={}
    if step==0:
        for name,p in [('attention_block1',model.model.layers[0].self_attn.q_proj.weight),
                       ('mlp_block24',model.model.layers[23].mlp.up_proj.weight)]:
            proof[name]=float(p.grad.detach().norm());assert proof[name]>0
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['clip']);assert torch.isfinite(norm)
    opt.step();opt.zero_grad(set_to_none=True)
    for device in range(torch.cuda.device_count()):torch.cuda.synchronize(device)
    return dict(step=step+1,lr=spec['lr'],blocked_ids=[r['id'] for r in blocked],
        allowed_ids=[r['id'] for r in allowed] if spec['objective']=='preserve' else [],
        rehearsal_id=rehearsal[index]['id'] if spec['objective']=='preserve' else None,
        blocked_exposure_epochs=(step+1)*2/sum(r['label']=='blocked' for r in rows),
        processed_tokens=tokens,grad_norm=float(norm),outside_support_gradient_proof=proof,**values)


def train(root,key,stop_after=None):
    started=time.time();m=read(root/'manifest.json');assert m['sources']==sources()
    if not m['smoke']:
        check=read(root.parent/'hardware/hardware_validation.json')
        assert check['sources']==m['sources'] and check['actual_process_resume_exact']
    spec=m['jobs'][key];directory=root/'jobs'/key;directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'worker.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/'DONE.json').exists():return
    parent=Path(m['parent_root'])/'parents'/spec['parent'];ready=read(parent/'DONE.json')
    assert ready['parent']==m['parents'][spec['parent']] and sha(parent/'initial.pt')==ready['checkpoint_sha256']
    assert sha(parent/'general_teacher.pt')==ready['teacher_sha256']
    ident=dict(run=digest(m),job=spec,parent=ready['checkpoint_sha256'],data=m['data_sha256']);fingerprint=digest(ident)
    e,torch=engine();cfg=read(HERE/'reference_v3/original_manifest.json')['protocol']
    tok,model=tokenizer_and_model(e,torch,cfg,True);saved=e.core.load_pt(parent/'initial.pt')
    load_parameters(model,saved['parameters'],torch);del saved
    model.requires_grad_(True)
    assert all(p.requires_grad for p in model.parameters()) and sum(p.numel() for p in model.parameters())==ready['trainable_scalars']
    assert not any('parametrizations' in n for n,_ in model.named_parameters())
    teacher=e.core.load_pt(parent/'general_teacher.pt');assert teacher['owner']==ready['parameter_hash']
    data=read(root/'data.json');assert digest(data)==m['data_sha256']
    opt=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=0,foreach=False)
    history=[];start=0
    checkpoint=directory/'resume.pt'
    if checkpoint.exists():
        saved=e.core.load_pt(checkpoint);assert saved['fingerprint']==fingerprint
        load_parameters(model,saved['parameters'],torch);opt.load_state_dict(saved['optimizer'])
        history=saved['history'];start=saved['step'];assert start==len(history)
        torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda']);random.setstate(saved['rng_python']);del saved
    original=read(parent/'baseline_dev.json')
    def recover(step):
        values=snapshot(model);size=sum(v.numel()*v.element_size() for v in values.values())
        atomic_state(checkpoint,dict(fingerprint=fingerprint,parameters=values,optimizer=opt.state_dict(),
            step=step,history=history,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_python=random.getstate()),3*size)
    def observe(step):
        file=directory/f'dev_{step:04d}.json';weights=directory/f'step_{step:04d}.pt'
        values=snapshot(model);hashed=e.core.tensor_hash(values.items())
        if file.exists():
            report=read(file);assert report['fingerprint']==fingerprint and report['parameter_hash']==hashed
            assert weights.exists();del values
        else:
            measured=measure(e,torch,model,tok,data,'dev',m['smoke'])
            selection=quality(measured,original,step)
            if step==0:
                assert hashed==ready['parameter_hash'];hardlink(parent/'initial.pt',weights)
            else:atomic_state(weights,dict(fingerprint=fingerprint,step=step,parameters=values,parameter_hash=hashed),sum(v.numel()*v.element_size() for v in values.values()))
            del values
            report=dict(fingerprint=fingerprint,step=step,parameter_hash=hashed,selection=selection,report=measured)
            write(file,report)
        best=read(directory/'best.json') if (directory/'best.json').exists() else None
        if best is None or report['selection']['rank']>best['selection']['rank']:
            hardlink(weights,directory/'best.pt')
            write(directory/'best.json',dict(step=step,selection=report['selection'],parameter_hash=hashed,fingerprint=fingerprint))
        write(directory/'progress.json',dict(step=step,limit=spec['limit'],selection=report['selection']))
        print(json.dumps(dict(job=key,**read(directory/'progress.json'))),flush=True)
    observe(start)
    for step in range(start,spec['limit']):
        tick=time.monotonic();record=update(e,torch,model,tok,data['attack_train'],data['rehearsal'],teacher['logits'],spec,step,opt)
        record['train_seconds']=time.monotonic()-tick;history.append(record)
        if step+1 in m['evaluation_steps']:observe(step+1)
        write(directory/'history.json',history)
        progress=read(directory/'progress.json');progress['step']=step+1;write(directory/'progress.json',progress)
        if (step+1)%32==0 or step+1==spec['limit'] or step+1==stop_after:recover(step+1)
        if step+1==stop_after:
            write(directory/'INTERRUPTED.json',dict(fingerprint=fingerprint,step=step+1));return
    hardlink(directory/f'step_{spec["limit"]:04d}.pt',directory/'final.pt')
    write(directory/'DONE.json',dict(fingerprint=fingerprint,identity=ident,step=len(history),
        final_parameter_hash=e.core.tensor_hash(model.named_parameters()),
        optimizer_hash=optimizer_hash(e,opt) if m['smoke'] else None,
        best=read(directory/'best.json'),started_unix=started,ended_unix=time.time(),
        cuda_peak=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]))


def heldout(root,key):
    started=time.time()
    m=read(root/'manifest.json');assert not m['smoke'] and m['sources']==sources()
    assert all((root/'jobs'/k/'DONE.json').exists() for k in m['jobs'])
    folder=root/'jobs'/key;done=read(folder/'DONE.json');spec=m['jobs'][key]
    parent=Path(m['parent_root'])/'parents'/spec['parent'];baseline=read(parent/'baseline_test.json')
    e,torch=engine();cfg=read(HERE/'reference_v3/original_manifest.json')['protocol']
    tok,model=tokenizer_and_model(e,torch,cfg,False);data=read(root/'data.json')
    for name in ['best','final']:
        output=folder/f'test_{name}.json'
        if output.exists():assert read(output)['identity']==done['identity'];continue
        state=e.core.load_pt(folder/(name+'.pt'));load_parameters(model,state['parameters'],torch);del state
        before=e.core.tensor_hash(model.named_parameters())
        assert before==(done['best']['parameter_hash'] if name=='best' else done['final_parameter_hash'])
        report=measure(e,torch,model,tok,data,'test')
        after=e.core.tensor_hash(model.named_parameters());assert before==after
        step=done['best']['step'] if name=='best' else spec['limit']
        write(output,dict(identity=done['identity'],endpoint=name,step=step,parameter_hash=before,
            selection=quality(report,baseline,step),report=report,parameters_unchanged=True,started_unix=started))
    write(folder/'TEST_DONE.json',dict(identity=done['identity']))


def available_gpus(candidates):
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in text.splitlines()}
    return [g for g in candidates if memory[g]<64]


def campaign(root,phase,gpus):
    m=read(root/'manifest.json');assert m['sources']==sources()
    lock=(root/'launcher.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=list(m['parents']) if phase=='prepare' else list(m['jobs']);count=2 if phase=='train' else 1
    marker='DONE.json' if phase in ['prepare','train'] else 'TEST_DONE.json';base='parents' if phase=='prepare' else 'jobs'
    def complete(k):return (root/base/k/marker).exists()
    active={};failures=[];(root/'logs').mkdir(exist_ok=True)
    while True:
        used={g for _,gs,_ in active.values() for g in gs}
        free=available_gpus([g for g in gpus if g not in used])
        for key in tasks:
            if complete(key) or key in active or len(free)<count or failures:continue
            assigned=free[:count];free=free[count:]
            args=[sys.executable,'-u',str(HERE/'attack.py'),'--root',str(root),'--'+phase,key]
            if m['smoke'] and phase=='train' and key=='interrupted' and not (root/base/key/'INTERRUPTED.json').exists():args+=['--stop-after','1']
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
        if failures and not active:raise RuntimeError('Worker failures; committed checkpoints retained: '+str(failures))
        if all(complete(k) for k in tasks) and not active:break
        time.sleep(5)
    if m['smoke'] and phase=='train':
        a=read(root/'jobs/uninterrupted/DONE.json');b=read(root/'jobs/interrupted/DONE.json')
        assert a['final_parameter_hash']==b['final_parameter_hash'] and a['optimizer_hash']==b['optimizer_hash']
        write(root/'hardware_validation.json',dict(actual_process_resume_exact=True,parameters_sha256=a['final_parameter_hash'],
            optimizer_sha256=a['optimizer_hash'],all_three_two_gpu_workers_passed=True,sources=m['sources']))
    if phase=='heldout':write(root/'COMPLETE.json',dict(run=digest(m),jobs=len(tasks),finished_unix=time.time()))
    print('Completed phase',phase,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--initialize',action='store_true');p.add_argument('--inputs',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--parent-root',type=Path);p.add_argument('--prepare');p.add_argument('--train');p.add_argument('--heldout')
    p.add_argument('--stop-after',type=int);p.add_argument('--phase',choices=['prepare','train','heldout']);p.add_argument('--gpus',default='0,1,2,3,4,5')
    a=p.parse_args();root=a.root.resolve()
    if a.initialize:initialize(root,a.inputs.resolve(),a.smoke,a.parent_root.resolve() if a.parent_root else None)
    elif a.prepare:prepare(root,a.prepare)
    elif a.train:train(root,a.train,a.stop_after)
    elif a.heldout:heldout(root,a.heldout)
    else:campaign(root,a.phase,[int(g) for g in a.gpus.split(',')])


if __name__=='__main__':main()
