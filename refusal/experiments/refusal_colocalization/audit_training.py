"""Read-only CUDA checks of loss indexing, checkpointed gradients and ownership."""
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

base=Path(sys.argv[1]).resolve()
os.environ.update(HF_HOME=str(base/'cache/huggingface'),HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',CUDA_CACHE_PATH=str(base/'cache/cuda'),TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(base))
import run as experiment
import torch
from torch.nn import functional as F

torch.set_num_threads(2)
torch.manual_seed(31)
torch.backends.cuda.matmul.allow_tf32=False
root=base/'results'
cfg=json.loads((root/'manifest.json').read_text())['protocol']
data=json.loads((root/'data.json').read_text())
tokenizer=experiment.AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
tokenizer.pad_token=tokenizer.eos_token
model=experiment.load_model(SimpleNamespace(model=cfg['model'],revision=cfg['revision'],two_gpu=False),'cuda')
route=experiment.Routing(model,'native',4,cfg,'different')
frozen=route.frozen_hash()
initial=experiment.core.load_pt(root/'capability/native__k4__same/final.pt')['parameters']
final=experiment.core.load_pt(root/'joint/native__k4__different/final.pt')['parameters']
route.load(initial)
results=dict(no_optimizer_steps=True,completion_token_ids={x:tokenizer.encode(x,add_special_tokens=False)
    for x in ['a','REFUSE']})

cases=[(data['train'][0],data['train'][0]['answer']),
       (data['train'][1],'REFUSE'),
       *[(r,r['answer']) for r in data['rehearsal'] if r['task']=='copy'][:2]]
results['loss_indexing']=[]
model.eval()
with torch.no_grad():
    for row,answer in cases:
        ids=torch.tensor([experiment.ids_for(tokenizer,row,answer)],device=model.device)
        labels=ids.clone()
        labels[:,:len(experiment.ids_for(tokenizer,row))]=-100
        manual=float(experiment.nll(model,tokenizer,row,answer))
        builtin=float(model(input_ids=ids,labels=labels,use_cache=False).loss)
        assert math.isclose(manual,builtin,rel_tol=1e-5,abs_tol=2e-6)
        results['loss_indexing'].append(dict(id=row['id'],answer=answer,manual=manual,builtin=builtin))
print('PASS completion loss matches built-in causal LM loss',flush=True)

def gradients(group,loss_function):
    model.zero_grad(set_to_none=True)
    route.enable(group)
    loss=loss_function()
    loss.backward()
    selected={id(p) for p in group}
    assert not any(p.grad is not None for p in route.saved if id(p) not in selected)
    values={n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in route.named.items()}
    norm=math.sqrt(sum(float(v.square().sum()) for v in values.values() if v is not None))
    assert norm>0 and math.isfinite(norm)
    return float(loss.detach()),norm,values

model.train()
results['checkpointed_gradients']=[]
for role,group,row,answer in [('capability',route.cap,*cases[0]),('refusal',route.ref,*cases[1])]:
    model.gradient_checkpointing_disable()
    a,_,ga=gradients(group,lambda:experiment.nll(model,tokenizer,row,answer))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    b,_,gb=gradients(group,lambda:experiment.nll(model,tokenizer,row,answer))
    assert math.isclose(a,b,rel_tol=1e-5,abs_tol=2e-6)
    for name in ga:
        assert (ga[name] is None)==(gb[name] is None)
        if ga[name] is not None:
            assert torch.allclose(ga[name],gb[name],rtol=1e-4,atol=1e-6)
    results['checkpointed_gradients'].append(dict(role=role,max_abs_difference=max(
        float((ga[n]-gb[n]).abs().max()) for n in ga if ga[n] is not None)))
print('PASS checkpointed gradients match uncheckpointed gradients; no owner leakage',flush=True)

allowed=[r for r in data['train'] if r['label']=='allowed']
teacher=experiment.core.load_pt(root/'capability/native__k4__same/teacher.pt')['logits']
base_teacher=experiment.core.load_pt(root/'base_teacher.pt')['logits']
model.gradient_checkpointing_disable()
results['weighted_component_gradients']=[]
for phase,state in [('joint_initial',initial),('joint_final',final)]:
    route.load(state)
    terms=[('allowed_nll',route.cap,.5,lambda:experiment.nll(model,tokenizer,data['train'][0],data['train'][0]['answer'])),
           ('blocked_nll',route.ref,.5,lambda:experiment.nll(model,tokenizer,data['train'][1],'REFUSE')),
           ('allowed_preservation',route.cap,cfg['preserve'],lambda:experiment.kl(model,tokenizer,allowed[0],teacher[0])),
           ('general_conservation',route.all,cfg['conserve'],lambda:experiment.kl(model,tokenizer,data['rehearsal'][0],base_teacher[0]))]
    for name,group,weight,fn in terms:
        loss,norm,g=gradients(group,fn)
        by_role={role:weight*math.sqrt(sum(float(v.square().sum()) for n,v in g.items() if v is not None and n.endswith(suffix)))
                 for role,suffix in [('C','deltas.0'),('R','deltas.1')]}
        results['weighted_component_gradients'].append(dict(phase=phase,component=name,loss=loss,weight=weight,**by_role))
print('PASS component gradient ownership checked on actual warm-start and final weights',flush=True)

results['continuation_KL_examples']=[]
model.eval()
for row in [r for r in data['rehearsal'] if r['task'] in ('copy','addition')][:2]+[r for r in data['rehearsal'] if r['task']=='copy'][:2]:
    ids=torch.tensor([experiment.ids_for(tokenizer,row,row['answer'])],device=model.device)
    length=ids.shape[1]-len(experiment.ids_for(tokenizer,row))
    with torch.no_grad():
        route.load({n:torch.zeros_like(v) for n,v in initial.items()})
        target=F.log_softmax(model(input_ids=ids,use_cache=False,logits_to_keep=length+1).logits[0,:-1].float(),dim=-1)
        route.load(final)
        student=F.log_softmax(model(input_ids=ids,use_cache=False,logits_to_keep=length+1).logits[0,:-1].float(),dim=-1)
        divergence=F.kl_div(student,target,log_target=True,reduction='none').sum(-1).cpu().tolist()
    results['continuation_KL_examples'].append(dict(id=row['id'],gold_answer=row['answer'],KL_per_completion_position=divergence))
results['greedy_mode_diagnostic']=[]
node_ids=[tokenizer.encode(c,add_special_tokens=False)[0] for c in 'abcdefghij']
ref_id=tokenizer.encode('REFUSE',add_special_tokens=False)[0]
for placement in ['same','different']:
    state=experiment.core.load_pt(root/f'joint/native__k4__{placement}/final.pt')['parameters']
    route.load(state)
    for row in data['dev'][:32]:
        with torch.no_grad():
            probs=F.softmax(experiment.first_logits(model,tokenizer,row),dim=-1)
        results['greedy_mode_diagnostic'].append(dict(placement=placement,id=row['id'],label=row['label'],
            refusal_prefix_probability=float(probs[ref_id]),node_total_probability=float(probs[node_ids].sum()),
            largest_node_probability=float(probs[node_ids].max()),
            refusal_prefix_is_argmax=bool(probs.argmax()==ref_id)))
assert route.frozen_hash()==frozen
results['frozen_originals_unchanged']=True
(base/'engineering_diagnostics_20261104.json').write_text(json.dumps(results,indent=2)+'\n')
print('ALL READ-ONLY CUDA AUDIT CHECKS PASSED',flush=True)
