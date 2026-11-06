"""CPU-only audit of completed navigation-removal records and endpoint weights."""
import argparse
import ast
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import pandas as pd


def close(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)):return math.isclose(a,b,abs_tol=2e-6,rel_tol=1e-6)
    return a==b


def main(root):
    sys.path.insert(0,str(root.parent/'source'))
    from attack import read,write,sha,sources
    from protocol import digest,batch,make_train,quality,refusal,CONFIG,STEPS
    m=read(root/'manifest.json');inputs=Path(m['inputs']);out=root.parent/'analysis'
    assert m['sources']==sources() and m['config']==CONFIG and not m['smoke']
    assert read(root/'COMPLETE.json')['run']==digest(m) and read(root/'COMPLETE.json')['jobs']==12
    status=read(root/'status.json');assert status['complete']==12 and not status['failures'] and not status['running']
    for name,h in m['input_hashes'].items():assert sha(inputs/name)==h,name
    data=read(root/'data.json');assert digest(data)==m['data_sha256']
    original=read(inputs/'data.json');excluded=set(read(root.parent/'source/reference_v3/excluded_worlds.json'))
    excluded.update(r['world'] for rows in original.values() for r in rows if 'world' in r)
    assert data['attack_train']==make_train(excluded,512,CONFIG['data_seed'])
    assert len({r['world'] for r in data['attack_train']})==512
    assert not {r['world'] for r in data['attack_train']}&excluded
    truth=0
    for name in ['attack_train','policy_dev','policy_test']:
        for r in data[name]:
            path=[];node=r['start']
            for _ in range(r['depth']):node=r['edges'][node];path.append(node)
            assert r['path']==path and r['answer']==' '.join(path) and r['world']==digest(r['edges'])
            assert (r['label']=='blocked')==(r['marker'] in path[:-1]);truth+=1
    for a,b in [('policy_dev','policy_test'),('rehearsal','general_dev'),('rehearsal','general_test'),('general_dev','general_test')]:
        assert not {r['prompt'] for r in data[a]}&{r['prompt'] for r in data[b]}
    namespace={'re':re}
    for path,names in [(root.parent/'source/reference_v3/run.py',{'score','summarize'}),
                       (root.parent/'source/reference_v3/reference/removal.py',{'classify'})]:
        tree=ast.parse(path.read_text());parts=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        assert len(parts)==len(names);exec(compile(ast.Module(body=parts,type_ignores=[]),str(path),'exec'),namespace)
    files={};records=reports=0;retention=[];diagnostics=[]
    def check_report(r,split):
        nonlocal records,reports
        reports+=1
        for panel,name in [('task','policy_'+split),('general','general_'+split),('benchmark','benchmark_'+split)]:
            rows=r[panel]['rows'];gold=data[name];assert len(rows)==len(gold)
            for row,expected in zip(rows,gold):
                for key,value in expected.items():assert row[key]==value,(name,key,row['id'])
                if panel!='benchmark':
                    computed=namespace['score'](expected,row['response'])
                    for key,value in computed.items():assert row[key]==value,(row['id'],key)
                else:
                    lp=row['choice_logprobs'];pred=max(range(len(lp)),key=lambda i:lp[i])
                    assert row['predicted']==pred and row['correct']==(pred==expected['gold'])
                    assert math.isclose(row['gold_nll'],-lp[expected['gold']],abs_tol=1e-6)
                    assert math.isclose(row['choice_mass'],sum(math.exp(x) for x in lp),abs_tol=2e-6)
                records+=1
            if panel!='benchmark':assert close(r[panel]['metrics'],namespace['summarize'](rows))
            else:
                for task in ['arc_easy','winogrande']:
                    pool=[row for row in rows if row['task']==task]
                    for field in ['correct','gold_nll','choice_mass']:
                        assert close(r[panel]['metrics'][task+'_'+field],sum(row[field] for row in pool)/len(pool))
    def extras(parent,job,endpoint,r):
        general=r['general']['rows'];addition=[x for x in general if x['task']=='addition'];copy=[x for x in general if x['task']=='copy']
        retention.append(dict(parent=parent,job=job,endpoint=endpoint,
            general_exact=r['general']['metrics']['path_correct'],
            addition_exact=sum(x['path_correct'] for x in addition)/len(addition),
            addition_parsed=sum(namespace['classify'](x['response'],x['answer'])['correct'] for x in addition)/len(addition),
            copy_exact=sum(x['path_correct'] for x in copy)/len(copy),**r['benchmark']['metrics']))
        for depth in [1,2]:
            rows=[x for x in r['task']['rows'] if x['depth']==depth and x['label']!='blocked'];first=right_length=0
            for row in rows:
                text=row['response'].strip();valid=bool(re.fullmatch(r'[a-j](?:(?:\s+|\s*(?:->|→|,)\s*)[a-j])*',text))
                nodes=re.findall('[a-j]',text) if valid else []
                first+=bool(nodes and nodes[0]==row['path'][0]);right_length+=len(nodes)==depth
            diagnostics.append(dict(parent=parent,job=job,endpoint=endpoint,depth=depth,n=len(rows),
                path_accuracy=sum(x['path_correct'] for x in rows)/len(rows),first_node_correct=first/len(rows),
                correct_output_length=right_length/len(rows)))
    bases={};parent_done={}
    for key in m['parents']:
        p=root/'parents'/key;ready=read(p/'DONE.json');parent_done[key]=ready
        assert ready['run']==digest(m) and ready['materialization_effective_weights_exact'] and ready['baseline_response_replay_exact']
        assert ready['trainable_scalars']==494032768 and sha(p/'initial.pt')==ready['checkpoint_sha256']
        assert sha(p/'general_teacher.pt')==ready['teacher_sha256']
        for split in ['dev','test']:
            r=read(p/f'baseline_{split}.json');check_report(r,split);bases[(key,split)]=r
            if split=='test':extras(key,'start','start',r)
    last=max(read(root/'jobs'/key/'DONE.json')['ended_unix'] for key in m['jobs'])
    checks=[];weights=[];blocked_reference=None
    for key,spec in m['jobs'].items():
        f=root/'jobs'/key;done=read(f/'DONE.json');history=read(f/'history.json')
        ident=dict(run=digest(m),job=spec,parent=parent_done[spec['parent']]['checkpoint_sha256'],data=m['data_sha256'])
        assert done['identity']==ident and done['fingerprint']==digest(ident)
        assert len(history)==256 and [h['step'] for h in history]==list(range(1,257))
        chosen=[]
        for i,h in enumerate(history):
            b,a=batch(data['attack_train'],i);expected_b=[r['id'] for r in b];chosen.extend(expected_b)
            assert h['blocked_ids']==expected_b and h['lr']==spec['lr']
            assert h['allowed_ids']==([r['id'] for r in a] if spec['objective']=='preserve' else [])
            half=len(data['rehearsal'])//2;index=(i//2)%half+(i%2)*half
            assert h['rehearsal_id']==(data['rehearsal'][index]['id'] if spec['objective']=='preserve' else None)
            assert h['blocked_exposure_epochs']==h['step']/256 and h['processed_tokens']>0
            assert all(math.isfinite(v) for v in h.values() if isinstance(v,(float,int)))
        assert len(Counter(chosen))==512 and set(Counter(chosen).values())=={1}
        if blocked_reference is None:blocked_reference=chosen
        else:assert chosen==blocked_reference
        assert all(v>0 for v in history[0]['outside_support_gradient_proof'].values()) and len(history[0]['outside_support_gradient_proof'])==2
        selected=[]
        for step in STEPS:
            p=f/f'dev_{step:04d}.json';d=read(p);files[str(p.relative_to(root.parent))]=sha(p)
            assert d['fingerprint']==done['fingerprint'] and d['step']==step
            check_report(d['report'],'dev');q=quality(d['report'],bases[(spec['parent'],'dev')],step)
            assert q==d['selection'];selected.append(q)
        assert max(selected,key=lambda q:q['rank'])==done['best']['selection']
        for endpoint in ['best','final']:
            p=f/f'test_{endpoint}.json';d=read(p);files[str(p.relative_to(root.parent))]=sha(p)
            assert d['identity']==ident and d['started_unix']>=last and d['parameters_unchanged']
            check_report(d['report'],'test');q=quality(d['report'],bases[(spec['parent'],'test')],d['step'])
            assert q==d['selection'];extras(spec['parent'],key,endpoint,d['report'])
            expected=done['best']['parameter_hash'] if endpoint=='best' else done['final_parameter_hash']
            assert d['parameter_hash']==expected
            weights.append((f/(endpoint+'.pt'),expected,done['best']['step'] if endpoint=='best' else 256,done['fingerprint']))
        checks.append(dict(job=key,steps=256,saved_measurements=len(STEPS),sample_schedule_exact=True,selection_reproduced=True))
    print('PASS raw records:',records,'reports:',reports,'update sample schedules:',12*256,flush=True)
    import torch
    torch.set_num_threads(2);seen={};weight_checks=[]
    for path,expected,step,fingerprint in weights:
        inode=(path.stat().st_dev,path.stat().st_ino)
        if inode not in seen:
            saved=torch.load(path,map_location='cpu',weights_only=True);state=saved['parameters']
            assert sum(t.numel() for t in state.values())==494032768
            assert all(t.dtype==torch.float32 and 'parametrizations' not in name for name,t in state.items())
            h=hashlib.sha256()
            for name,t in state.items():
                h.update(name.encode());h.update(str((tuple(t.shape),t.dtype)).encode())
                h.update(memoryview(t.contiguous().view(torch.uint8).numpy()))
            seen[inode]=h.hexdigest()
            if step:assert saved['step']==step and saved['fingerprint']==fingerprint
            del saved,state;gc.collect()
        assert seen[inode]==expected,str(path)
        weight_checks.append(dict(file=str(path.relative_to(root.parent)),parameters_sha256=expected))
        print('PASS endpoint weights:',path.parent.name,path.name,flush=True)
    assert reports==196 and records==233536 and truth==7776
    pd.DataFrame(retention).to_csv(out/'retention.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv',index=False)
    write(out/'audit.json',dict(checks_passed=True,records_recounted=records,evaluation_reports=reports,graph_truth_records=truth,
        actual_updates_checked=3072,selection_checks=checks,endpoint_weight_references=len(weight_checks),
        unique_endpoint_weight_files=len(seen),weight_checks=weight_checks,all_checkpoint_intervals_retained=True,
        raw_report_hashes=files,source_sha256=sha(__file__),manifest_sha256=sha(root/'manifest.json'),
        no_gpu_forward_or_training=True,all_intermediate_weight_tensors_exhaustively_checked=False,
        formatting_diagnostics_posthoc=True,primary_gates_unchanged=True))
    print('ALL SAVED-RECORD AND ENDPOINT-WEIGHT CHECKS PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
