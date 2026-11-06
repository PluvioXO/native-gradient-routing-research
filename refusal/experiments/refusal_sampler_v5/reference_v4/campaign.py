"""Five independent budget arms, policy calibration, then held-out evaluation."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import run

CONDITIONS={**{f'shared{w}':('native',w,'same',f'native{w}') for w in [1,2,4,8]},
            'full':('full',0,'same','full')}


def acquisition_jobs(seed):
    return {f'cap_{name}_s{seed}':dict(kind='acquire',seed=seed,geometry=geometry,width=width,placement='same')
            for name,geometry,width in [('full','full',0),*[(f'native{w}','native',w) for w in [1,2,4,8]]]}


def policy_job(condition,seed,factor,calibration):
    geometry,width,placement,teacher=CONDITIONS[condition]
    key=f'policy_{condition}_s{seed}_lr{round(100*factor):03d}'
    return key,dict(kind='policy',condition=condition,seed=seed,geometry=geometry,width=width,
                    placement=placement,teacher=f'cap_{teacher}_s{seed}',lr_factor=factor,calibration=calibration)


def select_recipes(root,jobs,cfg):
    choices={}
    for condition in CONDITIONS:
        candidates=[]
        for factor in cfg['lr_factors']:
            key,_=policy_job(condition,31,factor,True)
            path=root/'jobs'/key/'policy/best.json'
            if path.exists():
                best=run.read(path)
                assert best['identity']['job']==dict(id=key,**jobs[key])
                candidates.append(dict(job=key,lr_factor=factor,selection=best['selection']))
        choices[condition]=(max(candidates,key=lambda c:tuple(c['selection']['rank'])+(-c['lr_factor'],)) if candidates else None)
    passed=all(c is not None and c['selection']['feasible'] for c in choices.values())
    return dict(validation_passed=passed,recipes=choices,selection_uses_test=False,selection_uses_directions=False,
                note='One-seed budget sweep. Each budget qualifies independently; failure at a narrow width never blocks the other arms.')


def launch(root,cfg,gpus,smoke):
    assert len(gpus)==len(set(gpus)) and len(gpus)>=2
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'launcher.lock').open('a'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.free_space(root); manifest=run.initialize(root,cfg); stamp=run.core.digest(manifest)
    (root/'logs').mkdir(exist_ok=True)
    run.core.atomic_json(root/'environment.json',dict(python=sys.version,torch=run.torch.__version__,cuda=run.torch.version.cuda,
        transformers=run.core.version('transformers'),gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv,noheader'],text=True)))
    initial={'base':dict(kind='base',seed=31),**acquisition_jobs(31)}
    path=root/'jobs.json'
    if not path.exists():run.core.atomic_json(path,initial)
    active={}; available=list(gpus)
    def done(key,test=False):
        file=root/'jobs'/key/('TEST_DONE.json' if test else 'DONE.json')
        if not file.exists():return False
        value=run.read(file)
        assert value['job']==jobs[key] and value['run']==stamp
        if 'identity' in value:
            assert value['identity']['job']==dict(id=key,**jobs[key])
            assert value['fingerprint']==run.core.digest(value['identity'])
        return True
    while True:
        jobs=run.read(path); changed=False
        if done('base'):
            for condition in CONDITIONS:
                for factor in cfg['lr_factors']:
                    key,spec=policy_job(condition,31,factor,True)
                    if key not in jobs:jobs[key]=spec; changed=True
        for key,spec in list(jobs.items()):
            if spec['kind']=='policy' and not done(key) and done(spec['teacher']):
                if not run.read(root/'jobs'/spec['teacher']/'DONE.json')['qualified']:
                    run.core.atomic_json(root/'jobs'/key/'DONE.json',dict(job=spec,run=stamp,qualified=False,
                        skipped=True,reason='capability_acquisition_failed'))
        calibration=[k for k,s in jobs.items() if s.get('calibration')]
        if len(calibration)==len(CONDITIONS)*len(cfg['lr_factors']) and all(done(k) for k in calibration):
            recipe_path=root/'recipes.json'
            selected=select_recipes(root,jobs,cfg)
            if recipe_path.exists():assert run.read(recipe_path)==selected
            else:run.core.atomic_json(recipe_path,selected)
        if changed:run.core.atomic_json(path,jobs)
        all_training=(root/'recipes.json').exists() and all(done(k) for k in jobs)
        tasks=[]
        if not done('base'):tasks=[('base',False,1)]
        elif not all_training:
            for key,spec in jobs.items():
                if not done(key) and (spec['kind']=='acquire' or done(spec['teacher'])):
                    tasks.append((key,False,2 if spec.get('geometry')=='full' else 1))
        else:
            chosen=run.read(root/'recipes.json')
            keys=['base',*[k for k,s in jobs.items() if s['kind']=='acquire']]
            keys.extend(c['job'] for c in chosen['recipes'].values() if c)
            keys.extend(k for k,s in jobs.items() if s['kind']=='policy' and not s['calibration'])
            for key in dict.fromkeys(keys):
                spec=jobs[key]
                if run.read(root/'jobs'/key/'DONE.json').get('skipped'):continue
                if spec['kind']=='policy' and not done(spec['teacher'],test=True):continue
                if not done(key,test=True):tasks.append((key,True,1))
            required=[k for k in dict.fromkeys(keys) if not run.read(root/'jobs'/k/'DONE.json').get('skipped')]
        busy={(k,t) for _,k,t,_,_ in active.values()}
        for key,test,count in tasks:
            if (key,test) in busy or len(available)<count:continue
            assigned=available[:count]; available=available[count:]
            command=[sys.executable,str(Path(run.__file__).resolve()),'--root',str(root),'--worker',key]
            if test:command.append('--test')
            if smoke:command.append('--smoke')
            handle=(root/'logs'/(key+('__test' if test else '')+'.log')).open('a')
            process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,assigned))),
                                     stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[process.pid]=(process,key,test,assigned,handle)
            print(dict(event='launch',job=key,test=test,pid=process.pid,gpus=assigned),flush=True)
        failures=[]
        for pid,(process,key,test,assigned,handle) in list(active.items()):
            code=process.poll()
            if code is not None:
                handle.close(); available.extend(assigned); available.sort(); del active[pid]
                print(dict(event='exit',job=key,test=test,code=code),flush=True)
                if code:failures.append(dict(job=key,test=test,code=code))
        status=dict(time=time.time(),training_complete=all_training,completed_training=sum(done(k) for k in jobs),
            currently_planned_training=len(jobs),running=[dict(job=k,test=t,pid=p,gpus=g) for p,(_,k,t,g,_) in active.items()],
            failures=failures,observational_only=True,validation=run.read(root/'recipes.json') if (root/'recipes.json').exists() else None)
        run.core.atomic_json(root/'status.json',status)
        if failures:
            for process,_,_,_,handle in active.values():process.wait(); handle.close()
            status['running']=[]; status['drained_after_failure']=True
            run.core.atomic_json(root/'status.json',status)
            raise RuntimeError(f'Worker failure; committed checkpoints retained: {failures}')
        if all_training and all(done(k,test=True) for k in required) and not active:break
        time.sleep(5)
    print('ALL TRAINING, SELECTED ENDPOINT TESTS AND OBSERVATIONS COMPLETE',flush=True)
