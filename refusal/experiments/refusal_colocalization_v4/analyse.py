"""Audit frozen branching-task measurements and draw the budget-sweep results."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results/weatherwax-completed-20261105'
ROOT=ARCHIVE/'results';OUT=HERE/'analysis'
COLORS={'native1':'#eab308','native2':'#d97706','native4':'#2563eb','native8':'#0f766e','full':'#7c3aed','base':'#52525b'}
NAMES={'native1':'1 channel','native2':'2 channels','native4':'4 channels','native8':'8 channels','full':'Full model','base':'Frozen base'}
def read(p):return json.loads(p.read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def close(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(int,float)):return math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-6)
    return a==b
def contrast(h,mask):return h[mask].mean(0)-h[~mask].mean(0)
def cosine(a,b):
    den=float(a.norm()*b.norm());return max(-1.,min(1.,float(a@b/den))) if den>1e-10 else None
def functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(path),'exec'),namespace)


def main():
    if not ROOT.exists():raise SystemExit('Raw measurements were removed during the 5 November cleanup. See refusal/RAW_DATA.md to restore them; analysed tables and plots remain in this study.')
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    manifest=read(ROOT/'manifest.json');cfg=manifest['protocol'];stamp=digest(manifest)
    assert manifest==read(HERE/'launch/production_manifest.json')
    assert cfg['layer']==11 and cfg['seeds']==[31] and cfg['direction_penalty']==0 and not cfg['direction_intervention']
    for name,sha in manifest['sources'].items():assert hashlib.sha256((ARCHIVE/'source'/name).read_bytes()).hexdigest()==sha
    data=read(ROOT/'data.json');old=read(ROOT/'direction_data.json')
    assert digest(data)==manifest['data_sha256'] and digest(old)==manifest['observation_sha256']
    jobs=read(ROOT/'jobs.json');status=read(ROOT/'status.json');recipes=read(ROOT/'recipes.json')
    assert status['training_complete'] and status['completed_training']==status['currently_planned_training']==len(jobs)==21
    assert not status['running'] and not status['failures'] and not recipes['selection_uses_test'] and not recipes['selection_uses_directions']
    excluded=set(read(ARCHIVE/'source/excluded_worlds.json'));banks={}
    for panel,rows in data.items():
        assert len(rows)==len({r['id'] for r in rows})
        for r in rows:
            if r['task']!='pointer':continue
            assert digest(r['edges'])==r['world'] and r['world'] not in excluded
            node=r['start'];path=[]
            for command in r['commands']:node=r['edges'][node][command];path.append(node)
            assert path==r['path'] and r['answer']==' '.join(path)
            assert len(set([r['start'],*path]))==len(path)+1
            assert (r['marker'] in path[:-1])==(r['label']=='blocked')
            banks.setdefault(r['id'].split('_')[0],set()).add(r['world'])
    assert sum(map(len,banks.values()))==len(set().union(*banks.values()))
    assert not {r['prompt'] for r in data['benchmark_dev']}&{r['prompt'] for r in data['benchmark_test']}
    assert not {r['group'] for r in old['fit']}&{r['group'] for r in old['heldout']}
    ns={'re':re}
    functions(ARCHIVE/'source/reference/removal.py',['classify'],ns)
    functions(ARCHIVE/'source/run.py',['score','summarize','policy_selection','capability_selection'],ns)
    functions(ARCHIVE/'source/observe.py',['refusal'],ns)
    assert ns['classify']('15 + 23 = 38','38')['correct'] and not ns['classify']('15 + 23 = 39','38')['correct']
    expected={k:{r['id']:r for r in rows} for k,rows in data.items()};reports=records=0
    def audit(report,panel):
        nonlocal reports,records
        rows=report['rows'];truth=expected[panel]
        assert len(rows)==len(truth) and {r['id'] for r in rows}==truth.keys()
        for r in rows:
            assert all(r[k]==v for k,v in truth[r['id']].items())
            if panel.startswith('benchmark'):
                lp=np.array(r['choice_logprobs']);assert np.isfinite(lp).all()
                assert r['predicted']==int(lp.argmax()) and r['correct']==(r['gold']==r['predicted'])
                assert close(r['gold_nll'],-lp[r['gold']]) and close(r['choice_mass'],np.exp(lp).sum())
            else:assert all(r[k]==v for k,v in ns['score'](r,r['response']).items())
        metric=({t+'_'+k:float(np.mean([r[k] for r in rows if r['task']==t])) for t in sorted({r['task'] for r in rows}) for k in ['correct','gold_nll','choice_mass']}
                if panel.startswith('benchmark') else ns['summarize'](rows))
        assert close(metric,report['metrics']);reports+=1;records+=len(rows)
    baseline=read(ROOT/'base.json')
    for panel,report in baseline.items():audit(report,panel)
    dev=[];losses=[];phases=[];candidates=[];skips=[];clones=0;checkpoint_pairs=0;masks={};frozen=set()
    for key,spec in jobs.items():
        directory=ROOT/'jobs'/key;done=read(directory/'DONE.json');assert done['job']==spec and done['run']==stamp
        if done.get('skipped'):
            assert done['reason']=='capability_acquisition_failed' and not read(ROOT/'jobs'/spec['teacher']/'DONE.json')['qualified']
            skips.append(key);continue
        if spec['kind']=='base':continue
        layout=read(directory/'layout.json')
        assert done['identity']['run']==stamp and done['identity']['job']==dict(id=key,**spec) and done['fingerprint']==digest(done['identity'])
        if layout['frozen_hash']:frozen.add(layout['frozen_hash'])
        if spec['geometry']=='native':
            assert layout['unique_trainable']==2688*spec['width']
            assert all(x['name'].startswith('model.layers.11.mlp.') for x in layout['layout']) and len(layout['layout'])==3
            indices=layout['layout'][0]['indices'][0];assert all(x['indices'][0]==indices for x in layout['layout'])
            masks[spec['width']]=set(indices)
        for endfile in sorted(directory.glob('*/DONE.json')):
            folder=endfile.parent;stage=folder.name;end=read(endfile);history=read(folder/'history.json')
            assert end['identity']['job']==dict(id=key,**spec) and end['identity']['run']==stamp and end['fingerprint']==digest(end['identity'])
            limit=cfg[{'basic':'basic_steps','hard':'hard_steps','policy':'policy_steps'}[stage]]
            assert end['step']==len(history)==limit and end['reason']=='fixed_update_budget'
            assert [h['step'] for h in history]==list(range(1,limit+1)) and all(math.isfinite(v) for h in history for v in h.values())
            peak=cfg['full_lr' if spec['geometry']=='full' else 'sparse_lr']*spec.get('lr_factor',1.)
            assert all(close(h['learning_rate'],peak*min(1,h['step']/cfg['warmup'])) for h in history)
            losses.extend(dict(job=key,condition=spec.get('condition','full' if spec['geometry']=='full' else f"native{spec['width']}"),stage=stage,**h) for h in history)
            paths=sorted(folder.glob('dev_*.json'));assert [int(p.stem[4:]) for p in paths]==list(range(0,limit+1,cfg['eval_every']))
            selections=[]
            for p in paths:
                step=int(p.stem[4:]);report=read(p);panel={'basic':'dev_basic','hard':'dev_mixed','policy':'policy_dev'}[stage]
                audit(report['task'],panel);audit(report['general'],'general_dev')
                if 'benchmark' in report:audit(report['benchmark'],'benchmark_dev')
                selection=ns['policy_selection' if stage=='policy' else 'capability_selection'](report,baseline,cfg,step);selections.append(selection)
                d=dict(job=key,condition=spec.get('condition','full' if spec['geometry']=='full' else f"native{spec['width']}"),stage=stage,step=step,
                    path_correct=report['task']['metrics']['path_correct'],general=report['general']['metrics']['path_correct'],**{k:v for k,v in selection.items() if k!='step'})
                if stage=='policy':
                    allowed=[v for k,v in report['task']['metrics']['by_case'].items() if '_blocked_' not in k]
                    d.update(allowed_correct=float(np.mean([v['path_correct'] for v in allowed])),false_refusal=float(np.mean([v['refusal'] for v in allowed])),mean_blocked_refusal=report['task']['metrics']['by_label']['blocked']['refusal'])
                dev.append(d)
            best=read(folder/'best.json');optimal=max(selections,key=lambda x:x['rank'])
            assert close(best['selection'],optimal) and best['report']==read(folder/f"dev_{best['step']:04d}.json")
            assert end['qualified']==optimal['feasible'] and end['report']==read(paths[-1])
            phases.append(dict(job=key,stage=stage,steps=limit,best_step=best['step'],qualified=end['qualified']))
            if stage=='policy':candidates.append(dict(job=key,condition=spec['condition'],lr_factor=spec['lr_factor'],**optimal))
            if spec['geometry']=='native':
                final=load(folder/'final.pt');resume=load(folder/'checkpoint.pt');best_tensor=load(folder/'best.pt')
                assert final['identity']==resume['identity']==end['identity'] and final['step']==resume['step']==limit and resume['history']==history
                assert final['parameters'].keys()==resume['parameters'].keys()
                for n,v in final['parameters'].items():
                    assert torch.isfinite(v).all() and torch.equal(v,resume['parameters'][n])
                    if n.endswith('deltas.1'):assert torch.count_nonzero(v)==0
                assert best_tensor['step']==best['step'] and best_tensor['identity']==best['identity'] and best_tensor['selection']==optimal
                checkpoint_pairs+=1
        if spec['kind']=='policy':
            parent=read(ROOT/'jobs'/spec['teacher']/'acquisition.json');initial=read(directory/'policy/dev_0000.json')['task']['rows']
            assert [r['response'] for r in initial]==[r['response'] for r in parent['dev_mixed']['rows']];clones+=1
            selected=read(ROOT/'jobs'/spec['teacher']/'hard/best.json')
            assert done['identity']['parent']==digest(dict(fingerprint=selected['fingerprint'],step=selected['step']))
        else:
            acq=read(directory/'acquisition.json');best=read(directory/'hard/best.json')
            assert acq['qualified']==done['qualified']==best['selection']['feasible'] and acq['selected_step']==best['step'] and acq['dev_mixed']==best['report']['task']
    assert len(skips)==6 and clones==9 and len(frozen)==1 and masks[1]<masks[2]<masks[4]<masks[8]
    for condition,choice in recipes['recipes'].items():
        options=[c for c in candidates if c['condition']==condition]
        if not options:assert choice is None;continue
        candidate=max(options,key=lambda c:tuple(c['rank'])+(-c['lr_factor'],))
        assert choice['job']==candidate['job'] and choice['lr_factor']==candidate['lr_factor'] and close(choice['selection'],{k:v for k,v in candidate.items() if k not in ['job','condition','lr_factor']})
    assert not any(c['feasible'] for c in candidates) and not recipes['validation_passed']
    print('Training gates, selections, nine parent clones and sparse checkpoints passed.',flush=True)
    endpoints=[];cases=[];depths=[];general=[];wording=[]
    for p in sorted(ROOT.glob('jobs/*/test_*.json')):
        v=read(p);key=p.parent.name;spec=jobs[key];role=key+'__'+v['endpoint']
        assert v['run']==stamp and v['job_id']==key
        if spec['kind']!='base':assert v['checkpoint_identity']['job']==dict(id=key,**spec)
        for panel in ['task','wording','ood','general','benchmark']:audit(v['panels'][panel],{'task':'policy_test','wording':'wording_test','ood':'ood_test','general':'general_test','benchmark':'benchmark_test'}[panel])
        row=dict(role=role,job=key,kind=spec['kind'],condition=spec.get('condition','base' if spec['kind']=='base' else 'full' if spec['geometry']=='full' else f"native{spec['width']}"),endpoint=v['endpoint'],step=v['step'],
            width=spec.get('width',0),unique_trainable=0 if spec['kind']=='base' else read(p.parent/'layout.json')['unique_trainable'],
            qualified=read(p.parent/'DONE.json')['qualified'],**v['panels']['benchmark']['metrics'])
        for panel in ['task','wording','ood']:
            raw=pd.DataFrame(v['panels'][panel]['rows']);a=raw[raw.label!='blocked'];b=raw[raw.label=='blocked']
            row.update({panel+'_'+k:float(value) for k,value in dict(path_correct=raw.path_correct.mean(),allowed_correct=a.path_correct.mean(),blocked_refusal=b.refusal.mean(),false_refusal=a.refusal.mean(),malformed=1-raw.format_valid.mean(),truncated=raw.truncated.mean()).items()})
            world=a.groupby('world').path_correct.mean().sort_index().to_numpy();boot=np.random.default_rng(41106).integers(0,len(world),(2000,len(world)))
            row[panel+'_allowed_ci_low'],row[panel+'_allowed_ci_high']=np.quantile(world[boot].mean(1),[.025,.975])
            for case,g in raw.groupby('case'):cases.append(dict(role=role,panel=panel,case=case,n=len(g),path_correct=g.path_correct.mean(),refusal=g.refusal.mean(),correct_behavior=g.refusal.mean() if '_blocked_' in case else g.path_correct.mean()))
            for depth,g in raw.groupby('depth'):
                aa=g[g.label!='blocked'];bb=g[g.label=='blocked'];depths.append(dict(role=role,panel=panel,depth=depth,path_correct=g.path_correct.mean(),allowed_correct=aa.path_correct.mean(),blocked_refusal=bb.refusal.mean(),false_refusal=aa.refusal.mean()))
        a=pd.DataFrame(v['panels']['task']['rows']);w=pd.DataFrame(v['panels']['wording']['rows']);worlds=set(w.world)
        aa=a[(a.label!='blocked')&a.world.isin(worlds)].groupby('world').path_correct.mean().sort_index();ww=w[w.label!='blocked'].groupby('world').path_correct.mean().sort_index()
        assert aa.index.equals(ww.index);delta=(ww-aa).to_numpy();boot=np.random.default_rng(41106).integers(0,len(delta),(2000,len(delta)))
        low,high=np.quantile(delta[boot].mean(1),[.025,.975]);wording.append(dict(role=role,primary_matched=aa.mean(),wording=ww.mean(),difference_pp=100*delta.mean(),ci_low_pp=100*low,ci_high_pp=100*high))
        for task in ['addition','copy']:
            rs=[r for r in v['panels']['general']['rows'] if r['task']==task]
            row[task]=np.mean([r['path_correct'] for r in rs]);row[task+'_parsed']=np.mean([r['answer_parsed_correct'] for r in rs])
            general.extend(dict(role=role,**r) for r in rs)
        row['general']=v['panels']['general']['metrics']['path_correct'];row['general_parsed']=v['panels']['general']['metrics']['answer_parsed_correct'];endpoints.append(row)
    assert len(endpoints)==17
    assert min(p.stat().st_mtime for p in ROOT.glob('jobs/*/test_*.json'))>=max(p.stat().st_mtime for p in ROOT.glob('jobs/*/DONE.json'))
    base=load(ROOT/'observations/base/final/activations.pt');nfit=len(old['fit']);blocked=torch.tensor([r['label']=='blocked' for r in data['direction_observe']]);directions=[];old_behavior=[];cosine_errors=[]
    for p in ROOT.glob('observations/*/*/activations.pt'):
        key=p.parent.parent.name;endpoint=p.parent.name;role=key+'__'+endpoint;spec=jobs[key];a=load(p);alignment=read(p.parent/'alignment.json')
        assert a['parameter_hash']==a['parameter_hash_after'] and a['observational_only'] and alignment['observational_only'] and not alignment['direction_penalty_used'] and not alignment['causal_ablation_performed']
        assert len(a['behavior'])==144 and len(a['new_behavior'])==64 and a['behavior']==alignment['old_behavior'] and a['new_behavior']==alignment['new_behavior']
        assert all(r['refusal_language']==ns['refusal'](r['response']) for r in a['behavior']+a['new_behavior'])
        for split,rs in [('fit',a['behavior'][:nfit]),('heldout',a['behavior'][nfit:])]:
            for r,truth in zip(rs,old[split]):assert all(r[k]==v for k,v in truth.items())
            for label in ['safe','unsafe']:old_behavior.append(dict(role=role,split=split,label=label,refusal_language=np.mean([r['refusal_language'] for r in rs if r['label']==label])))
        for r,truth in zip(a['new_behavior'],data['direction_observe']):assert all(r[k]==v for k,v in truth.items())
        parent=load(ROOT/'observations'/spec['teacher']/'best/activations.pt') if spec['kind']=='policy' else base
        for i,record in enumerate(alignment['layers']):
            h=a['new'][i];oh=a['old'][i];assert h.shape==(64,896) and oh.shape==(144,896) and torch.isfinite(h).all() and torch.isfinite(oh).all()
            induced=contrast(h,blocked)-contrast(parent['new'][i],blocked);c=cosine(induced,base['basis'][i]);saved=record['cosine_induced_vs_original']
            assert (c is None)==(saved is None)
            if c is not None:
                assert math.isclose(c,saved,abs_tol=1e-6,rel_tol=1e-5)
                cosine_errors.append(abs(c-saved))
            assert math.isclose(float(induced.norm()),record['induced_contrast_norm'],abs_tol=1e-6,rel_tol=1e-5)
            if spec.get('geometry')=='native' and i<11:assert torch.equal(h,base['new'][i]) and torch.equal(oh,base['old'][i])
            directions.append(dict(role=role,block=i+1,**record))
    assert len(directions)==17*24
    tables=dict(endpoints=pd.DataFrame(endpoints),cases=pd.DataFrame(cases),depths=pd.DataFrame(depths),development=pd.DataFrame(dev),losses=pd.DataFrame(losses),phases=pd.DataFrame(phases),lr_calibration=pd.DataFrame(candidates),general_responses=pd.DataFrame(general),wording_paired=pd.DataFrame(wording),directions=pd.DataFrame(directions),old_refusal_behavior=pd.DataFrame(old_behavior))
    costs=tables['losses'].groupby('job').agg(updates=('step','size'),train_seconds=('train_seconds','sum')).reset_index()
    costs['allocated_gpu_train_seconds']=[r.train_seconds*(2 if jobs[r.job]['geometry']=='full' else 1) for r in costs.itertuples()];tables['fit_costs']=costs
    for name,table in tables.items():table.to_csv(OUT/(name+'.csv'),index=False)
    audit_receipt=dict(checks_passed=True,completed_utc=datetime.fromtimestamp(status['time'],timezone.utc).isoformat(),scheduled_entries=21,executed_entries=15,skipped_policy_jobs=skips,policy_fits=9,acquisition_fits=5,seeds=[31],all_joint_candidates_infeasible=True,
        evaluation_reports_recounted=reports,records_recounted=records,endpoint_reports=17,initial_parent_clones_verified=clones,sparse_checkpoint_pairs_verified=checkpoint_pairs,observational_hash_pairs_verified=17,direction_rows_recomputed=len(directions),
        source_hashes_verified=9,excluded_previous_worlds=len(excluded),archive_sha256='bd4eacddeb6c952d727d6976618aef78e6db504306a4bc0fd8e09f7d2e3d9309',bootstrap=dict(replicates=2000,unit='graph world',includes_seed_variation=False),max_cosine_recompute_error=max(cosine_errors),full_model_tensors_exhaustively_compared=False)
    (OUT/'audit.json').write_text(json.dumps(audit_receipt,indent=2)+'\n')
    plot(tables,recipes)
    print(json.dumps(audit_receipt,indent=2));print(tables['endpoints'][['condition','kind','endpoint','task_path_correct','task_allowed_correct','task_blocked_refusal','task_false_refusal','general']].to_string(index=False))


def plot(t,recipes):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    df=t['endpoints'];caps=df[(df.kind=='acquire')&(df.endpoint=='best')].set_index('condition');pol=df[(df.kind=='policy')&(df.endpoint=='best')].copy();pol['geometry']=pol.condition.str.replace('shared','native',regex=False);pol=pol.set_index('geometry')
    order=['native1','native2','native4','native8','full'];porder=['native4','native8','full'];cv=t['development'];page=0
    with PdfPages(OUT/'results.pdf') as pdf:
        def save(fig,title,note):
            nonlocal page
            page+=1;fig.suptitle(title,fontsize=16,fontweight='bold',y=.98);fig.text(.02,.015,note,fontsize=9,va='bottom');fig.tight_layout(rect=(0,.085,1,.94));pdf.savefig(fig);fig.savefig(OUT/f'page_{page:02d}.png',dpi=110);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for i,c in enumerate(order):
            v=100*caps.loc[c,'task_path_correct'];axs[0].bar(i,v,color=COLORS[c]);axs[0].text(i,v+2,f'{v:.1f}%',ha='center')
            d=t['depths'];s=d[(d.role==caps.loc[c,'role'])&(d.panel=='task')];axs[1].plot(s.depth,100*s.path_correct,marker='o',color=COLORS[c],label=NAMES[c])
        axs[0].set_xticks(range(5),[NAMES[c] for c in order],rotation=20);axs[0].set_title('Capability-only: all marker cases');axs[1].set_xticks([1,2,3,4]);axs[1].set_title('Capability accuracy by trained depth');axs[1].set_xlabel('Command count');axs[1].legend()
        for ax in axs:ax.set_ylim(0,110);ax.set_ylabel('Correct full traces (%)')
        save(fig,'Four/eight channels learn the harder task; one/two fail qualification', 'Development-selected capability parents. 128 held-out graphs, two command plans per graph, 6,656 cases.\nAcquisition requires answering every case, including ones that will later require refusal. One seed; one MLP block.')
        fig,axs=plt.subplots(2,2,figsize=(13,8))
        for c in order:
            job=caps.loc[c,'job']
            for ax,stage,col,title in [(axs[0,0],'basic','path_correct','Basic acquisition: depths 1/2'),(axs[0,1],'hard','path_correct','Hard acquisition: depths 1/2/3/4'),(axs[1,0],'hard','min_depth_accuracy','Hard acquisition: worst depth')]:
                s=cv[(cv.job==job)&(cv.stage==stage)];ax.plot(s.step,100*s[col],label=NAMES[c],color=COLORS[c]);ax.set_title(title);ax.set_xlabel('Stage updates');ax.set_ylim(0,104);ax.set_ylabel('Development accuracy (%)')
            s=t['losses'];s=s[(s.job==job)&(s.stage=='hard')];axs[1,1].plot(s.step,s.capability_nll.rolling(64,min_periods=1).mean(),color=COLORS[c],label=NAMES[c])
        axs[0,0].legend();axs[1,1].set_title('Hard-stage NLL (64-update rolling mean)');axs[1,1].set_xlabel('Stage updates');axs[1,1].set_ylabel('NLL')
        save(fig,'Every budget received 512 basic + 2,048 hard updates','Fixed horizons, no early stopping. Hard acquisition starts from each arm\'s own selected basic checkpoint.\nThese are matched update/example budgets, not matched training FLOPs. Full-model training uses two GPUs.')
        fig,axs=plt.subplots(1,3,figsize=(13,5))
        for ax,col,title in zip(axs,['task_allowed_correct','task_blocked_refusal','task_false_refusal'],['Allowed traces correct','Blocked cases refused','Allowed cases falsely refused']):
            for i,c in enumerate(porder):
                v=100*pol.loc[c,col];ax.bar(i,v,color=COLORS[c]);ax.text(i,v+2,f'{v:.1f}%',ha='center')
            ax.set_xticks(range(3),[NAMES[c] for c in porder]);ax.set_ylim(0,110);ax.set_title(title);ax.set_ylabel('%')
        save(fig,'Selective refusal is not solved on the harder task','All displayed policy checkpoints are infeasible development-ranked fallbacks. 5,120 allowed and 1,536 blocked cases.\nOne/two-channel policy fits were not run after failed acquisition; they are not zero-refusal results.')
        fig,axs=plt.subplots(2,2,figsize=(13,8))
        for c in porder:
            row=pol.loc[c];s=cv[(cv.job==row.job)&(cv.stage=='policy')]
            for ax,col,title in zip(axs.flat,['min_allowed_accuracy','blocked_refusal','max_false_refusal','general'],['Worst allowed case','Worst blocked-position refusal','Worst false-refusal case','Exact addition/copy retention']):
                ax.plot(s.step,100*s[col],color=COLORS[c],label=NAMES[c]);best=s[s.step==row.step].iloc[0];ax.scatter([row.step],[100*best[col]],marker='*',s=110,color=COLORS[c]);ax.set_title(title);ax.set_ylim(-3,104);ax.set_xlabel('Policy updates');ax.set_ylabel('%')
        axs[0,0].axhline(80,ls='--',color='grey');axs[0,1].axhline(90,ls='--',color='grey');axs[1,0].axhline(10,ls='--',color='grey');axs[0,0].legend()
        save(fig,'Policy training trades allowed answers for incomplete refusal','Chosen learning-rate factor 0.1 in all three arms; stars are development-selected checkpoints.\nThresholds apply to each case, not aggregate accuracy. Every candidate trained for all 1,024 updates.')
        fig,ax=plt.subplots(figsize=(15,5));case_order=[f'h{d}_{s}' for d in [1,2,3,4] for s in ['unmarked','start','endpoint','branch','outside',*[f'blocked_{i}' for i in range(1,d)]]]
        values=[]
        for c in porder:
            s=t['cases'];s=s[(s.role==pol.loc[c,'role'])&(s.panel=='task')].set_index('case');values.append([100*s.loc[k,'correct_behavior'] for k in case_order])
        matrix=np.array(values);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto');ax.set_xticks(range(len(case_order)),[k.replace('blocked_','REFUSE ') for k in case_order],rotation=60,ha='right');ax.set_yticks(range(3),[NAMES[c] for c in porder])
        for (y,x),v in np.ndenumerate(matrix):ax.text(x,y,f'{v:.0f}',fontsize=7,ha='center',va='center',color='white' if v<65 else 'black')
        fig.colorbar(im,ax=ax,label='Correct required behavior (%)',fraction=.025)
        save(fig,'Where the refusal rule fails: depth and marker-position breakdown','Allowed cells require the full trace; REFUSE cells require refusal. Each cell has 256 command plans on 128 graphs.\nA start/endpoint marker is allowed; branch/outside markers are off the commanded path.')
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for ax,selected,title in [(axs[0],caps,'Capability parents'),(axs[1],pol,'Selected policy fallbacks')]:
            for j,(panel,label) in enumerate([('task','Primary wording'),('wording','Held-out wording'),('ood','Unseen depths 5/6')]):
                vals=[100*selected.loc[c,panel+'_allowed_correct'] for c in porder];ax.bar(np.arange(3)+(j-1)*.24,vals,.24,label=label)
            ax.set_xticks(range(3),[NAMES[c] for c in porder]);ax.set_ylim(0,108);ax.set_ylabel('Allowed full traces correct (%)');ax.set_title(title);ax.legend(fontsize=9)
        save(fig,'Primary-task success does not imply wording or depth generalization','Primary: 128 graphs. Held-out wording: paired first 64 primary-test graphs. Depths 5/6: 64 separate graphs.\nNo probe score selected weights/rates. Paired wording differences and graph-bootstrap intervals are saved in wording_paired.csv.')
        fig,axs=plt.subplots(2,3,figsize=(14,8));base=df[df.kind=='base'].iloc[0];selected=[base,*[caps.loc[c] for c in porder],*[pol.loc[c] for c in porder]];labels=['Base','Cap 4','Cap 8','Cap full','Policy 4','Policy 8','Policy full']
        for ax,col,title in zip(axs.flat,['addition_parsed','copy','arc_easy_correct','winogrande_correct','arc_easy_choice_mass','winogrande_choice_mass'],['Arithmetic answer accuracy','Exact copying','ARC-Easy accuracy','WinoGrande accuracy','ARC valid-option probability','WinoGrande valid-option probability']):
            ax.bar(range(7),[100*r[col] for r in selected],color=['#52525b',COLORS['native4'],COLORS['native8'],COLORS['full'],COLORS['native4'],COLORS['native8'],COLORS['full']]);ax.set_xticks(range(7),labels,rotation=35,ha='right');ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel('%')
        save(fig,'Native retention is strong on these small probes; full updates cause collateral loss','32 arithmetic + 32 copy prompts; 64 questions per MC benchmark. Exact and parsed arithmetic are separately retained.\nWinoGrande forced-choice accuracy can rise while valid-option probability collapses. These are not information-erasure tests.')
        fig,axs=plt.subplots(1,3,figsize=(14,5))
        for c in porder:
            row=pol.loc[c];s=t['directions'];s=s[s.role==row.role]
            for ax,col,title in zip(axs[:2],['cosine_induced_vs_original','cosine_fixed_partition_old_vs_original'],['New contrast vs old candidate','Old fixed-partition contrast stability']):ax.plot(s.block,s[col],color=COLORS[c],label=NAMES[c]);ax.set_title(title);ax.set_xlabel('Block');ax.set_ylim(-1.05,1.05);ax.axvline(12,color='grey',ls='--')
        ob=t['old_refusal_behavior'];roles=[base.role,*[pol.loc[c,'role'] for c in porder]]
        for j,label in enumerate(['safe','unsafe']):
            vals=[100*ob[(ob.role==role)&(ob.split=='heldout')&(ob.label==label)].iloc[0].refusal_language for role in roles];axs[2].bar(np.arange(4)+(j-.5)*.35,vals,.35,label=label)
        axs[2].set_xticks(range(4),['Base','4 channels','8 channels','Full'],rotation=25);axs[2].set_ylim(0,105);axs[2].set_title('Old held-out refusal-language heuristic');axs[2].set_ylabel('%');axs[0].legend();axs[2].legend()
        save(fig,'Direction measurements are observational, with unsuccessful new policies','No direction penalty, intervention or direction-based selection. Dashed line is the updated native block.\nLow cosine does not prove an independent refusal mechanism, circuit entanglement or attack resistance.')


if __name__=='__main__':main()
