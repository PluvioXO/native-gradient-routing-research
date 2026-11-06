"""Check old/all-on versus owner-only sampling and updates on the real RTX 2080."""
import argparse
from dataclasses import replace
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import run_study as study
import nr5_rl as rl
from nr6_checks import equal
from dai_rl import update


def check(support,report):
    data,_=study.init_runtime(); task=7
    torch.use_deterministic_algorithms(True)
    spec=dict(kind='smoke',support=support,budget='medium',width=26 if support=='native' else 4,
              id='gpu_equivalence_'+support,task=task,seed=907,lr=1e-4,steps=2)
    module,tok=study.build_module(spec)
    try:
        old=Path('/var/tmp/casr-mn866-composition/outputs/smoke02/fits')/f'smoke_{support}_medium/owner_final.pt'
        source=rl._load(old); state=module.state_dict(); state.update(source['deltas']); module.load_state_dict(state)
        cfg=study.config(spec,study.make_plan(smoke=True))
        examples=[next(e for e in data['train'] if e['task_id']==task)]*2
        initial=module.state_dict(); samples=[]; updated=[]; optimizer_states=[]; results=[]
        for independent in (None,None,task,task):
            module.load_state_dict(initial); module.independent_task=independent
            torch.manual_seed(4242); torch.cuda.manual_seed_all(4242)
            items=rl._generate(module,tok,examples,cfg,'all',sample=True)
            samples.append([i['generated_ids'] for i in items])
            opts=[torch.optim.AdamW(module.parameters(t),lr=cfg.lr,weight_decay=0) for t in list(range(8))*2]
            results.append(update(module,tok,items,[1.,-1.],cfg,opts,task,'routed',items))
            updated.append({n:v.cpu() for n,v in module.state_dict().items()})
            optimizer_states.append(rl._cpu([o.state_dict() for o in opts]))
        def max_diff(a,b):
            if torch.is_tensor(a): return float((a.float()-b.float()).abs().max())
            if isinstance(a,dict): return max((max_diff(a[k],b[k]) for k in a),default=0.)
            if isinstance(a,(list,tuple)): return max((max_diff(x,y) for x,y in zip(a,b)),default=0.)
            return abs(a-b) if isinstance(a,(float,int)) else float(a!=b)
        diagnostic=dict(support=support,all_samples_identical=all(s==samples[0] for s in samples),
            legacy_repeat_weight_diff=max_diff(updated[0],updated[1]),
            optimized_repeat_weight_diff=max_diff(updated[2],updated[3]),
            legacy_optimized_weight_diff=max_diff(updated[0],updated[2]),
            legacy_repeat_adam_diff=max_diff(optimizer_states[0],optimizer_states[1]),
            legacy_optimized_adam_diff=max_diff(optimizer_states[0],optimizer_states[2]),
            losses=results)
        study.atomic(report.with_name(report.stem+'_diagnostic.json'),diagnostic)
        print(json.dumps(diagnostic),flush=True)
        assert all(s==samples[0] for s in samples), 'Sampled tokens changed'
        assert equal(updated[0],updated[2]), 'Optimizer update changed'
        assert equal(optimizer_states[0],optimizer_states[2]), 'Adam state changed'
        assert results[0]==results[2], 'Loss, gradient norm, or KL guard changed'
        assert module.assert_frozen_base()
        result=dict(support=support,exact_sampled_tokens=True,exact_weight_update=True,exact_optimizer_state=True,
            exact_losses_and_guards=True,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
            gpu=torch.cuda.get_device_name(),deterministic_validation=True,script_sha256=study.hash_file(__file__))
        study.atomic(report,result); print(json.dumps(result),flush=True)
    finally: module.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--support',choices=('native','lora'),required=True)
    parser.add_argument('--report',type=Path,required=True); args=parser.parse_args(); check(args.support,args.report)
