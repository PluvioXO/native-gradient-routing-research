"""CPU checks for mixed-skill gradients, immutable teachers and atomic rollback."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'reference/src'),str(ROOT/'reference')]
import torch
from nr5_rl_checks import fixture
from nr6_checks import equal
from budget_model import BudgetNative,BudgetLoRA
from multi_run import Config
import nr5_rl as rl
from aware_update import update,frozen_teacher
import run_aware as study


def check():
    torch.set_num_threads(1)
    base,tok,_=fixture()
    cfg=Config(lr=1e-3,min_lr=1e-5,prompts_per_update=1,group_size=2,max_prompt_tokens=2048,
               max_mean_kl=10,max_sequence_kl=10,kl_positions=2,head_chunk=8,kl_head_chunk=8,
               clip_grad=1000,backtracks=1,checkpointing=True)
    cfg.mode='preserve';cfg.anchor_weight=1.
    example=dict(task_id=0,id='train',answer='2')
    def item(task=0):
        return dict(example=dict(example,task_id=task),prompt_ids=[1,2,3],generated_ids=[4,5,6])
    for cls in (BudgetNative,BudgetLoRA):
        kwargs=dict(seed=31,n_layers=2,allocation_neurons=2)
        kwargs.update(neurons=2) if cls is BudgetNative else kwargs.update(rank=2)
        module=cls(copy.deepcopy(base),**kwargs)
        with torch.no_grad():
            for name,p,t in module._named_deltas:p.add_(torch.randn_like(p)*.02)
        initial=module.state_dict()
        optimizers=[torch.optim.AdamW(module.parameters(t),lr=cfg.lr,weight_decay=0) for t in range(8)]
        for task in range(8):
            before=module.state_dict();opts=copy.deepcopy([o.state_dict() for o in optimizers])
            foreign=(task+1)%8
            aux=[dict(item(task),kind='self',teacher_task=task),dict(item(foreign),kind='foreign',teacher_task=foreign),
                 dict(prompt_ids=[1,2,3],generated_ids=[4,5,6],kind='neutral',teacher_task=None)]
            result=update(module,tok,[item(task),item(task)],[1.,-1.],cfg,optimizers,task,initial,aux)
            assert result['accepted'] and result['active_optimizer_indices']==[task]
            assert result['foreign_kl']>0 and result['neutral_kl']>0
            assert all(torch.equal(before[n],p) for n,p,t in module._named_deltas if t!=task)
            assert all(equal(opts[t],optimizers[t].state_dict()) for t in range(8) if t!=task)
            assert any(not torch.equal(before[n],p) for n,p,t in module._named_deltas if t==task)
        before=module.state_dict()
        for mode,weights in (('preservation_only',(1.,0.)),('conservation_only',(0.,1.))):
            probe=copy.copy(cfg);probe.mode=mode;probe.reference_kl=0.
            snapshots=module.state_dict()
            result=update(module,tok,[item(7),item(7)],[0.,0.],probe,optimizers,7,initial,aux)
            assert (result['preservation_weight'],result['conservation_weight'])==weights
            expected=weights[0]*.5*(result['self_kl']+result['foreign_kl'])+weights[1]*result['neutral_kl']
            assert abs(result['loss']-expected)<1e-9 and result['delta_max_move']>0
            assert all(torch.equal(snapshots[n],p) for n,p,t in module._named_deltas if t!=7)
        before=module.state_dict()
        try:
            with frozen_teacher(module,initial,7):raise ValueError('injected teacher failure')
        except ValueError:pass
        assert equal(before,module.state_dict())
        # An impossible guard must restore both parameters and the owner's existing Adam history.
        bad=copy.copy(cfg);bad.max_mean_kl=-1;bad.max_sequence_kl=-1
        opts=copy.deepcopy([o.state_dict() for o in optimizers])
        result=update(module,tok,[item(7),item(7)],[1.,-1.],bad,optimizers,7,initial,aux)
        assert not result['accepted'] and equal(before,module.state_dict())
        assert equal(opts,[o.state_dict() for o in optimizers])
        # With zero advantages and reference coefficient, the control's zero-weight
        # auxiliary passes must produce no parameter or Adam movement.
        for mode in ('aware','independent'):
            zero=copy.copy(cfg);zero.mode=mode;zero.reference_kl=0.
            result=update(module,tok,[item(7),item(7)],[0.,0.],zero,optimizers,7,initial,aux)
            assert result['delta_max_move']==0 and equal(before,module.state_dict())
            assert equal(opts,[o.state_dict() for o in optimizers])
            assert result['reward_mask']==([True]*8 if mode=='aware' else [False]*7+[True])
        assert module.assert_frozen_base();module.close()
    bank=dict(replay={t:[item(t)] for t in range(8)})
    neutral=dict(rows=[dict(prompt_ids=[1,2],generated_ids=[3,4])])
    for task in range(8):
        selected=[study.auxiliary_items(bank,neutral,31,1+task+8*i,task)[1]['teacher_task'] for i in range(7)]
        assert set(selected)==set(range(8))-{task}
    print('PASS both methods: eight-task gradient ownership, immutable teachers, frozen base, Adam rollback, zero-weight controls, foreign-task rotation',flush=True)


def check_evaluation_bank():
    import json
    import retention
    import multi_run
    previous=study.previous
    base,tok,_=fixture()
    data=json.loads((ROOT/'reference/dataset.json').read_text())
    small={split:[next(e for e in data[split] if e['task_id']==t) for t in range(8)] for split in ('train','dev','test')}
    plan=dict(source_hashes={})
    for cls,support,width in [(BudgetNative,'native',2),(BudgetLoRA,'lora',2)]:
        kwargs=dict(seed=31,n_layers=2,allocation_neurons=2)
        kwargs.update(neurons=2) if cls is BudgetNative else kwargs.update(rank=2)
        source=cls(copy.deepcopy(base),**kwargs)
        with torch.no_grad():
            for p in source.parameters():p.add_(.01)
        payload=dict(manifest=source.manifest(),deltas=source.state_dict(),plan_sha256=study.digest(plan))
        source.close()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);fit=output/'fits/train';fit.mkdir(parents=True)
            rl.atomic_torch(fit/'bank_final.pt',payload)
            study.atomic(fit/'completed.json',dict(bank_sha256=study.hash_file(fit/'bank_final.pt')))
            def predict(module,tokenizer,examples,indices,cfg,path,split):
                assert equal(module.state_dict(),payload['deltas'])
                rows=[dict(id=e['id'],task_id=e['task_id'],correct=True,answer=e['answer'],gold=e['answer']) for e in examples]
                study.atomic(path,dict(rows=rows,generation_seconds=0.));return rows
            def retained(module,benchmarks,path,mask,runtime):
                rows=[dict(id=e['id'],benchmark=e['benchmark'],gold=e['gold'],prediction=e['gold'],prediction_norm=e['gold'],correct=True,correct_norm=True) for e in benchmarks['examples']]
                value=dict(identity=dict(benchmarks=study.digest(benchmarks)),rows=rows,
                    wiki_summary=dict(perplexity=1.,mean_nll=0.))
                study.atomic(path,value);return value
            fresh=cls(copy.deepcopy(base),**kwargs)
            spec=dict(id='eval',support=support,width=width,seed=31,kind='merge',bank_checkpoint='train',steps=8)
            with patch.object(previous,'init_runtime',return_value=(small,dict(test=small['test']))), \
                 patch.object(previous,'build_module',return_value=(fresh,tok)), \
                 patch.object(previous,'predictions',side_effect=predict), \
                 patch.object(previous,'compound_evaluation',return_value=[]), \
                 patch.object(previous,'calibrate',return_value=[1.]*8), \
                 patch.object(retention,'evaluate',side_effect=retained), \
                 patch.object(multi_run,'witnesses',return_value=[torch.zeros(1)]):
                previous.merge_fit(spec,plan,output)
            result=json.loads((output/'fits/eval/completed.json').read_text())
            assert len(result['composition'])==120 and result['all_off_equal']
            assert result['calibrated_accuracy']==1 and result['plan_sha256']==study.digest(plan)
    print('PASS final bank loading into the original 1/2/4/8, retention and calibration evaluation path',flush=True)


if __name__=='__main__':
    check()
    check_evaluation_bank()
