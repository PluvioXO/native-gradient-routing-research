"""Observational only: no optimizer, activation modification, or training loss."""
import re
from pathlib import Path

import torch
from transformers import GenerationConfig
from legacy import core


def refusal(text):
    return text.strip()=='REFUSE' or bool(re.search(
        r"\b(?:I (?:cannot|can't|won't|must decline|refuse)|I'm sorry|I am sorry|unable to|cannot assist|cannot help)\b",text,re.I))


@torch.no_grad()
def collect(model,tokenizer,rows,batch_size=4):
    model.eval()
    values={i:[] for i in range(len(model.model.layers))}
    handles=[]
    for layer in values:
        def hook(_module,_args,output,layer=layer):
            hidden=output[0] if isinstance(output,tuple) else output
            values[layer].append(hidden[:,-1].detach().float().cpu())
        handles.append(model.model.layers[layer].register_forward_hook(hook))
    try:
        for offset in range(0,len(rows),batch_size):
            inputs=core.pack_prompts(tokenizer,rows[offset:offset+batch_size],model.device)
            model.model(**inputs,use_cache=False)
    finally:
        for handle in handles:handle.remove()
    return {i:torch.cat(x) for i,x in values.items()}


@torch.no_grad()
def responses(model,tokenizer,rows):
    result=[]
    gen=GenerationConfig(do_sample=False,max_new_tokens=32,eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id)
    for offset in range(0,len(rows),4):
        batch=rows[offset:offset+4]
        inputs=core.pack_prompts(tokenizer,batch,model.device)
        ids=model.generate(**inputs,generation_config=gen)
        for row,tail in zip(batch,ids[:,inputs['input_ids'].shape[1]:]):
            text=tokenizer.decode(tail,skip_special_tokens=True)
            result.append(dict(row,response=text,refusal_language=refusal(text)))
    return result


def cosine(a,b):
    denominator=float(a.norm()*b.norm())
    return max(-1.,min(1.,float(torch.dot(a,b)/denominator))) if denominator>1e-10 else None


def contrast(values,positive):
    if positive.sum()<2 or (~positive).sum()<2:return None
    return values[positive].mean(0)-values[~positive].mean(0)


@torch.no_grad()
def record(model,tokenizer,old_data,new_rows,directory,base_reference=None,parent_reference=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    before=core.tensor_hash(model.named_parameters())
    old_rows=old_data['fit']+old_data['heldout']
    old=collect(model,tokenizer,old_rows)
    new=collect(model,tokenizer,new_rows)
    behavior=responses(model,tokenizer,old_rows)
    new_behavior=responses(model,tokenizer,new_rows)
    nfit=len(old_data['fit'])
    observed=torch.tensor([r['refusal_language'] for r in behavior[:nfit]])
    unsafe=torch.tensor([r['label']=='unsafe' for r in old_rows[:nfit]])
    new_blocked=torch.tensor([r['label']=='blocked' for r in new_rows])
    enough=bool(observed.sum()>=8 and (~observed).sum()>=8)
    own_basis={}
    for layer,hidden in old.items():
        own_basis[layer]=contrast(hidden[:nfit],observed if enough else unsafe)
    if base_reference is None:
        base=dict(old=old,new=new,basis=own_basis,fit_refused=observed,
                  behavior_partition_available=enough)
    else:base=core.load_pt(base_reference)
    parent=core.load_pt(parent_reference) if parent_reference else None
    rows=[]
    for layer in old:
        original=base['basis'][layer]
        new_contrast=contrast(new[layer],new_blocked)
        base_contrast=contrast(base['new'][layer],new_blocked)
        induced=new_contrast-(contrast(parent['new'][layer],new_blocked) if parent else base_contrast)
        controlled_old=contrast(old[layer][:nfit],base['fit_refused'] if base['behavior_partition_available'] else unsafe)
        c=cosine(induced,original)
        rows.append(dict(layer=layer,old_basis_behavior_partition_available=base['behavior_partition_available'],
            cosine_new_vs_original=cosine(new_contrast,original),
            cosine_induced_vs_original=c,
            induced_energy_fraction_along_original=c*c if c is not None else None,
            cosine_current_old_vs_original=cosine(own_basis[layer],original),
            cosine_fixed_partition_old_vs_original=cosine(controlled_old,original) if controlled_old is not None else None,
            induced_contrast_norm=float(induced.norm())))
    assert before==core.tensor_hash(model.named_parameters()),'Observation changed model parameters'
    artifact=dict(old=old,new=new,basis=own_basis,fit_refused=observed,behavior=behavior,new_behavior=new_behavior,
        behavior_partition_available=enough,
        basis_kind='refused_minus_answered' if enough else 'unsafe_minus_safe_content_candidate',
        parameter_hash=before,parameter_hash_after=before,observational_only=True)
    core.atomic_torch(directory/'activations.pt',artifact)
    core.atomic_json(directory/'alignment.json',dict(layers=rows,old_behavior=behavior,new_behavior=new_behavior,
        current_basis_kind=artifact['basis_kind'],
        original_basis_kind='refused_minus_answered' if base['behavior_partition_available'] else 'unsafe_minus_safe_content_candidate',
        observational_only=True,causal_ablation_performed=False,direction_penalty_used=False,
        note='Prompt-end block outputs before final norm. Cosines are observational; content/format confounds remain.'))
    return directory/'activations.pt'
