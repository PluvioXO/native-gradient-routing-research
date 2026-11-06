"""Check legacy-control parity and zero-coefficient teacher isolation on CPU."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'reference'), str(ROOT/'reference/src')]
import torch
from nr5_rl_checks import fixture
from nr6_checks import equal
from budget_model import BudgetNative, BudgetLoRA
from multi_run import Config
from aware_update import auxiliary_weights, update
import run_aware as study


def check(legacy_path):
    controls = json.loads((ROOT/'controls_manifest.json').read_text())
    control_root = Path(controls['root'])
    assert study.hash_file(control_root/'outputs/main/experiment_plan.json') == controls['plan_sha256']
    plan = json.loads((control_root/'outputs/main/experiment_plan.json').read_text())
    assert study.hash_file(legacy_path) == plan['source_hashes']['aware_update.py']
    assert plan['inputs'] == json.loads((ROOT/'inputs_manifest.json').read_text())
    for record in controls['records']:
        assert study.hash_file(control_root/record['path']) == record['sha256']
        if record['bank_sha256']:
            assert study.hash_file((control_root/record['path']).parent/'bank_final.pt') == record['bank_sha256']
    used_banks = {f"bank_{j['support']}_{j['budget']}_s{j['seed']}.pt"
                  for j in study.make_plan(False)['jobs'] if j['kind']=='main'}
    assert len(used_banks) == 6
    for name in used_banks:
        assert study.hash_file(ROOT.parent/'inputs'/name) == plan['inputs']['banks'][name]
    assert study.hash_file(ROOT.parent/'inputs/neutral.json') == plan['inputs']['neutral_sha256']
    # Apart from split coefficients and the new plan/launch/tests, scientific code and data are identical.
    allowed = {'aware_update.py', 'run_aware.py'}
    for name, expected in plan['source_hashes'].items():
        if name not in allowed:
            assert study.hash_file(ROOT/name) == expected, name
    spec = importlib.util.spec_from_file_location('legacy_aware_update', legacy_path)
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    expected = {'aware': (0.,0.,0.), 'preservation_only': (.5,.5,0.),
        'conservation_only': (0.,0.,1.), 'preserve': (.5,.5,1.),
        'self_only': (.5,0.,0.), 'foreign_only': (0.,.5,0.),
        'self_conserve': (.5,0.,1.), 'foreign_conserve': (0.,.5,1.)}
    assert set(expected.values()) == {(s,f,c) for s in (0.,.5) for f in (0.,.5) for c in (0.,1.)}
    for mode, weights in expected.items():
        assert tuple(auxiliary_weights(mode,1.).values()) == weights
    torch.set_num_threads(1); torch.manual_seed(31)
    base,tok,_ = fixture()
    cfg = Config(lr=1e-3,min_lr=1e-5,prompts_per_update=1,group_size=2,max_prompt_tokens=2048,
        max_mean_kl=10,max_sequence_kl=10,kl_positions=2,head_chunk=8,kl_head_chunk=8,
        clip_grad=1000,backtracks=1,checkpointing=True,reference_kl=0.)
    cfg.anchor_weight=1.
    def item(task):
        return dict(example=dict(task_id=task,id='train',answer='2'),prompt_ids=[1,2,3],generated_ids=[4,5,6])
    aux=[dict(item(7),kind='self',teacher_task=7),dict(item(0),kind='foreign',teacher_task=0),
        dict(prompt_ids=[1,2,3],generated_ids=[4,5,6],kind='neutral',teacher_task=None)]
    for cls in (BudgetNative,BudgetLoRA):
        kwargs=dict(seed=31,n_layers=2,allocation_neurons=2)
        kwargs.update(neurons=2) if cls is BudgetNative else kwargs.update(rank=2)
        module=cls(copy.deepcopy(base),**kwargs)
        teachers=module.state_dict()
        with torch.no_grad():
            for _,p,_ in module._named_deltas: p.add_(torch.randn_like(p)*.02)
        current=module.state_dict()
        owners={name:t for name,_,t in module._named_deltas}
        def step(fn,mode,initial,advantages):
            module.load_state_dict(current)
            opts=[torch.optim.AdamW(module.parameters(t),lr=cfg.lr,weight_decay=0.) for t in range(8)]
            cfg.mode=mode; torch.manual_seed(123)
            metrics=fn(module,tok,[item(7),item(7)],advantages,cfg,opts,7,initial,aux)
            return module.state_dict(),[o.state_dict() for o in opts],metrics
        for mode in ('aware','preservation_only','conservation_only','preserve'):
            old=step(legacy.update,mode,teachers,[1.,-1.]);new=step(update,mode,teachers,[1.,-1.])
            assert equal(old[:2],new[:2]), (cls.__name__,mode,'legacy weight/Adam mismatch')
            assert math.isclose(old[2]['loss'],new[2]['loss'],abs_tol=1e-9)
        for mode in study.MODES:
            clean=step(update,mode,teachers,[0.,0.])
            inactive = 0 if mode.startswith('self') else 7
            changed=copy.deepcopy(teachers)
            for name in changed:
                if owners[name]==inactive: changed[name] += .15
            distractor=step(update,mode,changed,[0.,0.])
            assert equal(clean[:2],distractor[:2]), (mode,'inactive teacher changed the gradient')
            metrics=clean[2]
            assert metrics['accepted'] and metrics['delta_max_move']>0
            assert all(torch.equal(current[n],clean[0][n]) for n,t in owners.items() if t!=7)
            weights=auxiliary_weights(mode,1.)
            assert math.isclose(metrics['loss'],sum(weights[k]*metrics[k+'_kl'] for k in weights),abs_tol=1e-9)
        assert module.assert_frozen_base();module.close()
    for smoke,count in ((True,8),(False,48)):
        p=study.make_plan(smoke)
        assert len(p['jobs'])==count and len({j['id'] for j in p['jobs']})==count
    print('PASS: archived controls/inputs/code hashes; eight factorial cells; bitwise legacy weight/Adam parity; inactive-teacher isolation; routed ownership; plan counts',flush=True)


if __name__=='__main__':
    check(Path(sys.argv[1]))
