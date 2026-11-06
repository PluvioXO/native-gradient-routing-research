"""Generate and audit a fresh bank without querying a model or printing test answers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'reference'), str(ROOT/'reference/src')]
import multi_tasks as tasks
import nr5_tasks
import rlq_tasks as core


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def world(e):
    if e['task_id']==1:
        return 1,nr5_tasks.pointer_map_key(e)
    return e['task_id'],e.get('world_key',e['id'])


def originals():
    old=json.loads((ROOT/'reference/original_dataset.json').read_text())
    demos=nr5_tasks.ALL_DEMOS+[e for group in tasks.DEMOS.values() for e in group]
    return [e for s in ('train','dev','test','depth_probe') for e in old[s]]+demos


def make_data(protocol):
    forbidden=originals()
    seen={e['id'] for e in forbidden}
    worlds={world(e):'archived_or_demo' for e in forbidden}
    result={s:[] for s in ('train','dev','test','depth_probe')}
    attempts_by_stratum={}
    for split,rows in result.items():
        count=protocol['per_task_per_depth'][split]
        depths=protocol['probe_depths'] if split=='depth_probe' else protocol['training_depths']
        for task in range(8):
            if task==1:
                rng=random.Random(digest([protocol['data_seed'],split,'pointer_maps']))
                mappings=[]
                while len(mappings)<count//10:
                    digits=list(range(10));rng.shuffle(digits)
                    mapping=core._cycle(digits)
                    prototype=core._example(1,depths[0],core._pointer_problem(0,mapping,depths[0]))
                    key=world(prototype)
                    if key in worlds:continue
                    worlds[key]=split;mappings.append(mapping)
                for depth in depths:
                    for mapping in mappings:
                        for start in range(10):
                            e=core._example(1,depth,core._pointer_problem(start,mapping,depth))
                            e['family_id']=nr5_tasks.pointer_family_key(e)
                            assert e['id'] not in seen;seen.add(e['id']);rows.append(e)
                continue
            for depth in depths:
                answers=list('01') if task in (2,3) else list('1234') if task==7 else [str(depth)] if task==6 else list('0123456789')
                needed={a:count//len(answers) for a in answers}
                rng=random.Random(digest([protocol['data_seed'],split,task,depth]))
                attempts=0
                while any(needed.values()):
                    attempts+=1
                    if attempts>200000:raise RuntimeError(('Insufficient novel stratum',split,task,depth,needed))
                    e=core._candidate(rng,0,depth,int(rng.choice(answers))) if task==0 else tasks.candidate(task,depth,rng)
                    if e is None or not needed.get(e['answer'],0) or e['id'] in seen:continue
                    key=world(e)
                    if key in worlds and worlds[key]!=split:continue
                    seen.add(e['id']);worlds[key]=split;needed[e['answer']]-=1;rows.append(e)
                attempts_by_stratum[f'{split}/{task}/{depth}']=attempts
        rows.sort(key=lambda e:(e['task_id'],e['difficulty'],e['id']))
    result['provenance']=dict(study=protocol['study'],data_seed=protocol['data_seed'],protocol_sha256=digest(protocol),
        original_dataset_sha256=protocol['original_dataset_sha256'],difficulty_change=protocol['difficulty_change'],
        tasks=protocol['task_names'],attempts_by_stratum=attempts_by_stratum)
    audit(result,protocol)
    return result


def audit(data,protocol):
    forbidden=originals();seen={e['id'] for e in forbidden};worlds={world(e):'archived_or_demo' for e in forbidden}
    old_hash=hashlib.sha256((ROOT/'reference/original_dataset.json').read_bytes()).hexdigest()
    assert old_hash==protocol['original_dataset_sha256']
    report={}
    for split in ('train','dev','test','depth_probe'):
        rows=data[split];count=protocol['per_task_per_depth'][split]
        depths=protocol['probe_depths'] if split=='depth_probe' else protocol['training_depths']
        strata=Counter((e['task_id'],e['difficulty']) for e in rows)
        assert strata==Counter({(t,d):count for t in range(8) for d in depths})
        for task,depth in strata:
            answers=list('01') if task in (2,3) else list('1234') if task==7 else [str(depth)] if task==6 else list('0123456789')
            assert Counter(e['answer'] for e in rows if e['task_id']==task and e['difficulty']==depth)==Counter({a:count//len(answers) for a in answers})
        for e in rows:
            assert e['id'] not in seen;seen.add(e['id'])
            key=world(e);assert key not in worlds or worlds[key]==split;worlds[key]=split
            solved=core.solve(e['task_id'],e['problem']) if e['task_id']<2 else tasks.solve(e['task_id'],e['payload'])
            assert solved=={k:e[k] for k in ('answer','gold_trace')}
            assert tasks.format_prompt(e)==tasks.format_prompt(dict(e,answer='HIDDEN',gold_trace=[]))
            if e['task_id']>=2:assert e['problem']==tasks.render(e['task_id'],e['payload'])
        report[split]=dict(examples=len(rows),worlds=len({world(e) for e in rows}),sha256=digest(rows))
    return dict(passed=True,counts=report,exact_and_world_overlap_with_archive=0,
                split_world_overlap=0,labels_recomputed=True,target_blind_prompts=True)


if __name__=='__main__':
    protocol=json.loads((ROOT/'protocol.json').read_text())
    data=make_data(protocol)
    path=ROOT/'reference/dataset.json'
    path.write_text(json.dumps(data,indent=2)+'\n')
    report=audit(data,protocol)
    (ROOT/'data_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
