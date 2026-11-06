"""Recovery, locking, provenance, and owner-only numerical equivalence checks."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT), str(ROOT/'src')]
import torch
import run_study as study
import nr5_rl as rl
import nr6_train as train
from nr6_io import Workspace, digest
from nr5_rl_checks import fixture
from nr6_checks import equal
from budget_model import BudgetNative,BudgetLoRA
from dai_rl import update


def equivalence():
    torch.set_num_threads(1)
    base,tok,examples=fixture()
    from nr5_rl_checks import scripted_items
    items=scripted_items(tok,[examples[1]]*2)
    cfg=train.Config(max_steps=2, prompts_per_update=1,group_size=2,generation_batch=2,
        max_new_tokens=4,max_prompt_tokens=2048,eval_every=2,save_every=1,eval_batch=2,
        lr=1e-3,min_lr=1e-5,max_mean_kl=100,max_sequence_kl=100,kl_positions=2,
        head_chunk=8,kl_head_chunk=7,clip_grad=1e6,preservation_mode='all_ab',audit_steps=[])
    for cls in (BudgetNative,BudgetLoRA):
        kwargs=dict(seed=31,n_layers=2,allocation_neurons=2)
        kwargs.update(neurons=2) if cls is BudgetNative else kwargs.update(rank=2)
        modules=[cls(copy.deepcopy(base),**kwargs) for _ in range(2)]
        modules[1].independent_task=1
        with torch.no_grad():
            for p in modules[0].parameters(1): p.add_(torch.randn_like(p)*.001)
        modules[1].load_state_dict(modules[0].state_dict())
        initial=modules[0].state_dict()
        outputs=[]; results=[]; states=[]
        for m in modules:
            tensor=torch.tensor([items[0]['prompt_ids'][:12]])
            with torch.no_grad(),m.gate((True,)*8): outputs.append(m.model(tensor).logits)
            opts=[torch.optim.AdamW(m.parameters(t),lr=cfg.lr,weight_decay=0) for t in list(range(8))*2]
            results.append(update(m,tok,items,[1.,-1.],cfg,opts,1,'routed',items))
            states.append([o.state_dict() for o in opts])
        assert torch.equal(*outputs)
        assert equal(modules[0].state_dict(),modules[1].state_dict())
        assert equal(states[0],states[1])
        assert results[0]==results[1]
        for m in modules: m.close()
    print('PASS identical logits, losses, clipped gradients, optimizer states and updates with owner-only gates',flush=True)


def recovery_and_validation():
    cfg=train.Config(max_steps=2,eval_every=1,audit_steps=[])
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); ws=Workspace(root/'local',root/'durable',{'test':'repair'})
        latest=dict(step=1,best_step=0,pending_dev=False,signature={'test':True})
        rl.atomic_torch(ws.path('fit/latest.pt'),latest); ws.save_training('fit',cfg)
        expected=ws.path('fit/continuation.pt').read_bytes()
        (ws.durable/'fit/continuation.pt').write_bytes(b'broken')
        assert ws.restore_training('fit',cfg)
        assert ws.path('fit/continuation.pt').read_bytes()==expected
        assert rl._load(ws.path('fit/latest.pt'))==latest
        ws.path('fit/continuation.pt').write_bytes(b'local broken')
        try: ws.restore_training('fit',cfg)
        except RuntimeError: pass
        else: raise AssertionError('Must reject two broken copies')
        assert ws.path('fit/continuation.pt').read_bytes()==b'local broken'
        assert (ws.durable/'fit/continuation.pt').read_bytes()==b'broken'
    with tempfile.TemporaryDirectory() as tmp:
        output=Path(tmp); job=dict(id='test',kind='main',steps=2); plan={'jobs':[job]}
        fit=output/'fits/test'; fit.mkdir(parents=True)
        for bad in ('{','{}'):
            (fit/'completed.json').write_text(bad)
            try: study.validate_completed(output,job,plan)
            except ValueError: pass
            else: raise AssertionError('Corrupt marker accepted')
        owner=fit/'owner_final.pt'; owner.write_bytes(b'checkpoint hash witness')
        good=dict(spec=job,plan_sha256=digest(plan),final_step=2,status={'complete':True},owner_sha256=study.hash_file(owner))
        study.atomic(fit/'completed.json',good)
        assert study.validate_completed(output,job,plan)
        owner.write_bytes(b'corrupt')
        try: study.validate_completed(output,job,plan)
        except ValueError: pass
        else: raise AssertionError('Corrupt checkpoint accepted')
        revision=dict(scientific_plan_sha256=digest(plan),source_hashes={'test':'hash'})
        revision['id']=digest(revision); study.atomic(output/'execution_revision.json',revision)
        assert study.execution_revision(output,plan,{'test':'hash'})==revision
        try: study.execution_revision(output,plan,{'test':'changed'})
        except ValueError: pass
        else: raise AssertionError('Changed code accepted')
    from migrate_run import convert
    before=dict(signature={'trainer_sha256':'old'},latest={'signature':{'trainer_sha256':'old'},
        'deltas':{'a':torch.tensor([.25])},'optimizers':[{'state':torch.tensor([.5])}],
        'rng':{'torch':torch.get_rng_state()},'history':[{'step':1}]})
    after=convert(before,'old','new')
    assert before['signature']['trainer_sha256']=='old' and after['signature']['trainer_sha256']=='new'
    assert equal(before,convert(after,'new','old'))
    try: convert(before,'unexpected','new')
    except ValueError: pass
    else: raise AssertionError('Migration accepted a different source version')
    print('PASS validated completion/checkpoint hashes, corrupt-backup fallback, untouched failures and revision checks',flush=True)


def locks_and_shutdown():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); lock=root/'fits/fit/worker.lock'
        child_code='''import sys,time\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\nimport run_study as s\ntry:\n with s.exclusive(Path(sys.argv[2])):\n  Path(sys.argv[3]).write_text('locked')\n  time.sleep(30)\nexcept BlockingIOError: sys.exit(75)\n'''
        ready=root/'ready'
        child=subprocess.Popen([sys.executable,'-c',child_code,str(ROOT),str(lock),str(ready)])
        try:
            deadline=time.monotonic()+8
            while not ready.exists() and time.monotonic()<deadline: time.sleep(.05)
            assert ready.exists()
            second=subprocess.run([sys.executable,'-c',child_code,str(ROOT),str(lock),str(root/'second')],timeout=8)
            assert second.returncode==75 and not (root/'second').exists()
            plan={'jobs':[{'id':'fit','kind':'smoke'}],'source_hashes':{}}
            try: study.supervise(plan,root,['0'])
            except RuntimeError as e: assert 'Surviving worker' in str(e)
            else: raise AssertionError('Orphan worker was ignored')
        finally: child.terminate(); child.wait(timeout=8)
        with study.exclusive(lock): pass
    print('PASS exclusive per-fit locks, orphan detection and lock release after termination',flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); wrapper=root/'launcher_test.py'
        wrapper.write_text("""import sys,subprocess
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import run_study as s
out=Path(sys.argv[2]); original=subprocess.Popen
def spawn(command,**kwargs):
 p=original([sys.executable,'-c','import time; time.sleep(60)'])
 (out/'child_pid').write_text(str(p.pid))
 return p
s.subprocess.Popen=spawn
s.checked_space=lambda path:None
sys.argv=['run_study.py','smoke','--output',str(out),'--gpus','0']
s.main()
""")
        parent=subprocess.Popen([sys.executable,str(wrapper),str(ROOT),str(root)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            deadline=time.monotonic()+10
            while not (root/'status.json').exists() and time.monotonic()<deadline: time.sleep(.05)
            assert (root/'child_pid').exists() and (root/'status.json').exists()
            pid=int((root/'child_pid').read_text())
            parent.terminate(); parent.wait(timeout=10)
            import os
            try: os.kill(pid,0)
            except ProcessLookupError: pass
            else: raise AssertionError('SIGTERM left a worker running')
            assert json.loads((root/'status.json').read_text())['launcher_state']=='stopped'
        finally:
            if parent.poll() is None: parent.kill(); parent.wait()
    print('PASS real launcher SIGTERM cleanup and stopped status',flush=True)



if __name__=='__main__':
    equivalence(); recovery_and_validation(); locks_and_shutdown()
    print('ALL REPAIR CHECKS PASSED',flush=True)
