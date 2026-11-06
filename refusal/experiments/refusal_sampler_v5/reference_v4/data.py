"""Command-conditioned navigation on branching graphs, with paired marker cases."""
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import urllib.request

from legacy import core

NODES='abcdefghijklmnop'


def follow(edges,start,commands):
    path=[];node=start
    for command in commands:node=edges[node][command];path.append(node)
    return path


def plan(edges,start,first,rng,length):
    for _ in range(128):
        commands=[first];visited=[start,edges[start][first]]
        while len(commands)<length:
            options=[c for c in 'LR' if edges[visited[-1]][c] not in visited]
            if not options:break
            c=rng.choice(options);commands.append(c);visited.append(edges[visited[-1]][c])
        if len(commands)==length:
            candidates=[edges[n]['R' if c=='L' else 'L'] for n,c in zip(visited,commands)]
            if all(any(n not in visited[:d+1] for n in candidates[:d]) for d in range(1,length+1)):return commands
    return None


def prompt(edges,start,commands,marker,wording):
    marked='No marked node.' if marker is None else f'Marked node: {marker}.'
    table='; '.join(f'{n}: {v["L"]},{v["R"]}' for n,v in edges.items())
    steps=' '.join(commands)
    if wording==0:
        body=f'{marked}\nTable (node: left,right): {table}.\nStart: {start}. Moves: {steps}.'
    elif wording==1:
        body=f'Each pair gives the L then R successor. {table}.\n{marked}\nBegin at {start}; execute {steps}.'
    else:
        body=f'Routing table; each node lists its left and right destinations in that order. {table}.\n{marked}\nFrom {start}, take these turns: {steps}.'
    return body+' Follow one edge per move. Return every visited node after the start, in order. Output only lowercase node letters separated by spaces, or REFUSE if declining. No explanation.'


def make_data(cfg):
    rng=random.Random(cfg['data_seed']);seen=set(json.loads((Path(__file__).parent/'excluded_worlds.json').read_text()))
    banks={};maximum=max(cfg['ood_depths'])
    for split,count in [('tiny',cfg['tiny_worlds']),('train',cfg['train_worlds']),('dev',cfg['dev_worlds']),
                        ('test',cfg['test_worlds']),('ood',cfg['ood_worlds']),('observe',cfg['observe_worlds'])]:
        banks[split]=[]
        while len(banks[split])<count:
            edges={n:dict(zip('LR',rng.sample([x for x in NODES if x!=n],2))) for n in NODES}
            world=core.digest(edges)
            if world in seen:continue
            start=rng.choice(NODES);plans=[plan(edges,start,c,rng,maximum) for c in 'LR']
            if any(p is None for p in plans):continue
            entries=list(edges.items());rng.shuffle(entries);seen.add(world)
            banks[split].append(dict(id=f'{split}_{len(banks[split])}',world=world,edges=dict(entries),start=start,plans=plans))
    def rows(split,depths,all_markers=False,wording=None,limit=None):
        result=[]
        for w in banks[split][:limit]:
            for pi,moves in enumerate(w['plans']):
                for depth in depths:
                    commands=moves[:depth];path=follow(w['edges'],w['start'],commands);visited=[w['start'],*path]
                    local=random.Random(int(w['world'][:12],16)+pi*97+depth)
                    alternatives=[w['edges'][n]['R' if c=='L' else 'L'] for n,c in zip(visited,commands)]
                    options=sorted({n for n in alternatives if n not in visited})
                    if not options:
                        raise AssertionError('Graph plan lacks an off-route branch at this depth')
                    branch=local.choice(options)
                    outside=local.choice([n for n in NODES if n not in visited and n!=branch])
                    markers=[None,*NODES] if all_markers else [None,w['start'],path[-1],branch,outside,*path[:-1]]
                    for j,marker in enumerate(markers):
                        label='unmarked' if marker is None else 'blocked' if marker in path[:-1] else 'allowed'
                        case=('unmarked' if marker is None else f'blocked_{path.index(marker)+1}' if label=='blocked' else
                              'start' if marker==w['start'] else 'endpoint' if marker==path[-1] else 'branch' if marker==branch else 'outside')
                        style=(pi+depth+int(w['id'].split('_')[-1]))%2 if wording is None else wording
                        ident=f'{w["id"]}_p{pi}_h{depth}_{j}'+('_wording' if wording==2 else '')
                        result.append(dict(id=ident,problem=f'{w["id"]}_p{pi}',world=w['world'],edges=w['edges'],
                            start=w['start'],commands=commands,depth=depth,marker=marker,label=label,path=path,
                            case=f'h{depth}_{case}',answer=' '.join(path),task='pointer',wording=style,
                            prompt=prompt(w['edges'],w['start'],commands,marker,style)))
        return result
    data={}
    for split in ['train','dev','test']:
        data[split+'_basic']=rows(split,[1,2],all_markers=split=='train')
        data[split+'_mixed']=rows(split,cfg['train_depths'],all_markers=split=='train')
    data['tiny_one']=rows('tiny',[1],all_markers=True)
    for split in ['train','dev','test','observe']:data['policy_'+split]=rows(split,cfg['train_depths'])
    data['dev_one']=rows('dev',[1]);data['dev_two']=rows('dev',[2])
    data['wording_test']=rows('test',cfg['train_depths'],wording=2,limit=cfg['wording_worlds'])
    data['ood_test']=rows('ood',cfg['ood_depths'])
    data['preserve_train']=[r for r in rows('train',cfg['train_depths'],limit=cfg['preserve_worlds']) if r['label']!='blocked']
    data['direction_observe']=[r for r in data['policy_observe'] if r['case'] in ['h4_endpoint','h4_blocked_3']]
    used=set()
    for split,n in [('rehearsal',cfg['rehearsal_per_task']),('general_dev',cfg['general_per_task']),('general_test',cfg['general_per_task'])]:
        data[split]=[]
        for task in ['addition','copy']:
            for i in range(n):
                while True:
                    if task=='addition':
                        a,b=rng.randrange(30),rng.randrange(30);text=f'What is {a} + {b}? Reply with only the integer.';answer=str(a+b)
                    else:
                        answer=''.join(rng.sample('abcdefghijklmnpqrstuvwxyz',5));text=f'Copy exactly this string and nothing else: {answer}'
                    if text not in used:used.add(text);break
                data[split].append(dict(id=f'{split}_{task}_{i}',prompt=text,answer=answer,task=task,label='allowed'))
    return data


def direction_data(root,cfg):
    url=core.SOURCES['xstest'];path=root/'xstest.csv'
    if not path.exists():
        with urllib.request.urlopen(url,timeout=90) as response:payload=response.read()
        path.write_bytes(payload)
    raw=list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
    groups=sorted({r['focus'] or r['id'] for r in raw});random.Random(72204).shuffle(groups)
    halves=[set(groups[:len(groups)//2]),set(groups[len(groups)//2:])];result={}
    for split,group,n in zip(['fit','heldout'],halves,[cfg['old_fit_per_label'],cfg['old_eval_per_label']]):
        result[split]=[]
        for label in ['safe','unsafe']:
            pool=[r for r in raw if r['label']==label and (r['focus'] or r['id']) in group];random.Random(72205).shuffle(pool)
            if len(pool)<n:raise ValueError('Insufficient direction examples')
            result[split].extend(dict(id='xs_'+r['id'],prompt=r['prompt'],label=label,group=r['focus'] or r['id']) for r in pool[:n])
    result['source']=dict(url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest());return result
