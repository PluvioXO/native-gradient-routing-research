"""Read-only generation on the previously recorded activation-observation inputs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import run
import torch


def worker(root,job):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=run.read(root/'manifest.json')
    assert manifest['sources']==run.source_hashes()
    data=run.read(root/'data.json')
    assert run.core.digest(data)==manifest['data_sha256']
    cfg=manifest['protocol']
    tokenizer=run.AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'],padding_side='left')
    tokenizer.pad_token=tokenizer.eos_token
    model=run.load_model(cfg,two_gpu=False)
    model_fingerprint=run.core.digest(manifest)
    if job!='base':
        spec=run.read(root/'jobs.json')[job]
        route=run.Routing(model,spec['geometry'],spec['width'],cfg,spec['placement'])
        saved=run.core.load_pt(root/'jobs'/job/'final.pt')
        assert saved['identity']['job']==dict(id=job,**spec)
        assert saved['identity']['mask_hash']==run.core.digest(route.layout)
        run.state_load(route,saved)
        model_fingerprint=saved['fingerprint']
    before=run.core.tensor_hash(model.named_parameters())
    report=run.evaluate(model,tokenizer,data['policy_observe'])
    after=run.core.tensor_hash(model.named_parameters())
    assert before==after
    run.core.atomic_json(root/'observations'/job/'new_behavior.json',dict(job=job,
        checkpoint_fingerprint=model_fingerprint,data_sha256=manifest['data_sha256'],
        parameter_hash_before=before,parameter_hash_after=after,observational_only=True,
        direction_penalty_used=False,activation_intervention=False,report=report))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True); p.add_argument('--job')
    args=p.parse_args(); root=Path(args.root).resolve()
    if args.job:
        worker(root,args.job); return
    jobs=['base',*run.read(root/'jobs.json')]
    jobs=[j for j in jobs if not (root/'observations'/j/'new_behavior.json').exists()]
    logdir=root/'observation_replay_logs'; logdir.mkdir(exist_ok=True)
    free=list(range(6)); active={}
    while jobs or active:
        while jobs and free:
            job=jobs.pop(0); gpu=free.pop(0)
            handle=(logdir/(job+'.log')).open('a')
            command=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--job',job]
            process=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),
                                     stdout=handle,stderr=subprocess.STDOUT)
            active[process.pid]=(process,job,gpu,handle)
            print('START',job,gpu,flush=True)
        for pid,(process,job,gpu,handle) in list(active.items()):
            code=process.poll()
            if code is not None:
                handle.close(); free.append(gpu); del active[pid]
                if code:raise RuntimeError(f'Observation replay failed: {job}')
                print('DONE',job,flush=True)
        time.sleep(2)
    print('ALL OBSERVATION REPLAYS COMPLETE; NO OPTIMIZER STEPS',flush=True)


if __name__=='__main__':main()
