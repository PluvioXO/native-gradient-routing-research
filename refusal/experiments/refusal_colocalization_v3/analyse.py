"""Recount frozen mixed-hop predictions, audit selections, and plot the results."""
import ast
from datetime import datetime,timezone
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
ARCHIVE=HERE/'results/weatherwax-completed-20261104'
ROOT=ARCHIVE/'results'; OUT=HERE/'analysis'
NAMES={'base':'Frozen base','native4':'Capability: native 4','scattered4':'Scattered capability',
       'shared4':'Shared 4','shared8':'Shared 8','disjoint4x4':'Disjoint 4+4','full':'Full model'}
COLORS={'base':'#52525b','native4':'#0f766e','scattered4':'#a16207','shared4':'#2563eb',
        'shared8':'#0f766e','disjoint4x4':'#d97706','full':'#7c3aed'}
CASES=[f'h{d}_{c}' for d in [1,2] for c in ['unmarked','start','endpoint','outside']]+['h2_blocked']
def read(p):return json.loads(p.read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def close(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(int,float)):return math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-6)
    return a==b
def cosine(a,b):
    den=float(a.norm()*b.norm())
    return max(-1.,min(1.,float(a@b/den))) if den>1e-10 else None
def contrast(h,mask):return h[mask].mean(0)-h[~mask].mean(0)
def wilson(p,n):
    z=1.96; center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half,center+half


def main():
    if not ROOT.exists():raise SystemExit('Raw measurements were removed during the 5 November cleanup. See refusal/RAW_DATA.md to restore them; analysed tables and plots remain in this study.')
    torch.set_num_threads(2); OUT.mkdir(exist_ok=True)
    manifest=read(ROOT/'manifest.json'); cfg=manifest['protocol']; stamp=digest(manifest)
    assert manifest==read(HERE/'launch/production_manifest.json')
    assert cfg['direction_penalty']==0 and not cfg['direction_intervention'] and cfg['observational_only']
    for name,sha in manifest['sources'].items():assert hashlib.sha256((ARCHIVE/'source'/name).read_bytes()).hexdigest()==sha
    data=read(ROOT/'data.json'); old=read(ROOT/'direction_data.json')
    assert digest(data)==manifest['data_sha256'] and digest(old)==manifest['observation_sha256']
    status=read(ROOT/'status.json'); jobs=read(ROOT/'jobs.json'); recipes=read(ROOT/'recipes.json')
    assert status['training_complete'] and status['completed_training']==status['currently_planned_training']==len(jobs)==19
    assert not status['running'] and not status['failures'] and not recipes['validation_passed']
    assert not recipes['selection_uses_test'] and not recipes['selection_uses_directions']
    assert {s['seed'] for s in jobs.values()}=={31}
    excluded=set(read(ARCHIVE/'source/excluded_worlds.json')); banks={}
    for panel,rows in data.items():
        assert len({r['id'] for r in rows})==len(rows)
        for r in rows:
            if r['task']!='pointer':continue
            assert digest(r['edges'])==r['world'] and r['world'] not in excluded
            node=r['start']; path=[]
            for _ in range(r['depth']):node=r['edges'][node]; path.append(node)
            assert r['path']==path and r['answer']==' '.join(path)
            assert (r['marker'] in path[:-1])==(r['label']=='blocked')
            banks.setdefault(r['id'].split('_')[0],set()).add(r['world'])
    assert sum(map(len,banks.values()))==len(set().union(*banks.values()))
    probes=[r['prompt'] for k in ['rehearsal','general_dev','general_test'] for r in data[k]]
    assert len(probes)==len(set(probes))
    assert not {r['prompt'] for r in data['benchmark_dev']}&{r['prompt'] for r in data['benchmark_test']}
    assert not {r['group'] for r in old['fit']}&{r['group'] for r in old['heldout']}
    tree=ast.parse((ARCHIVE/'source/run.py').read_text()); source={'re':re}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
        and n.name in ['score','summarize','policy_selection']],type_ignores=[]),'frozen_scoring','exec'),source)
    tree=ast.parse((ARCHIVE/'source/observe.py').read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='refusal'],type_ignores=[]),'frozen_refusal','exec'),source)
    tree=ast.parse((ARCHIVE/'source/reference/removal.py').read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='classify'],type_ignores=[]),'existing_answer_parser','exec'),source)
    assert source['classify']('15 + 23 = 38','38')['correct']
    assert not source['classify']('15 + 23 = 39','38')['correct']
    assert not source['classify']('Perhaps the answer is 38.','38')['correct']
    evaluations=records=0
    def audit(report,panel):
        nonlocal evaluations,records
        expected={r['id']:r for r in data[panel]}; rows=report['rows']
        assert len(rows)==len(expected) and {r['id'] for r in rows}==expected.keys()
        for r in rows:
            assert all(r[k]==v for k,v in expected[r['id']].items())
            if panel.startswith('benchmark'):
                lp=np.array(r['choice_logprobs']); assert np.isfinite(lp).all()
                assert r['predicted']==int(lp.argmax()) and r['correct']==(r['gold']==r['predicted'])
                assert close(r['gold_nll'],-lp[r['gold']]) and close(r['choice_mass'],np.exp(lp).sum())
            else:assert all(r[k]==v for k,v in source['score'](r,r['response']).items())
        metrics=({t+'_'+k:np.mean([r[k] for r in rows if r['task']==t]) for t in sorted({r['task'] for r in rows})
                  for k in ['correct','choice_mass','gold_nll']} if panel.startswith('benchmark') else source['summarize'](rows))
        assert close(metrics,report['metrics']); evaluations+=1; records+=len(rows)

    baseline=read(ROOT/'base.json')
    for panel,report in baseline.items():audit(report,panel)
    development=[]; losses=[]; phases=[]; candidates=[]; skipped=[]; clones=[]; frozen=set(); checkpoint_pairs=0
    for key,spec in jobs.items():
        directory=ROOT/'jobs'/key; done=read(directory/'DONE.json')
        assert done['job']==spec and done['run']==stamp
        if done.get('skipped'):
            assert not done['qualified'] and done['reason']=='capability_acquisition_failed'
            assert not read(ROOT/'jobs'/spec['teacher']/'DONE.json')['qualified']; skipped.append(key);continue
        if spec['kind']=='base':continue
        layout=read(directory/'layout.json')
        assert done['identity']['job']==dict(id=key,**spec) and done['identity']['run']==stamp
        assert done['fingerprint']==digest(done['identity'])
        if layout['frozen_hash']:frozen.add(layout['frozen_hash'])
        if spec['geometry']!='full':assert layout['unique_trainable']==2688*spec['width']*(2 if spec['placement']=='different' else 1)
        for p in sorted(directory.glob('*/DONE.json')):
            phase=p.parent.name; end=read(p); history=read(p.parent/'history.json')
            assert end['identity']['job']==dict(id=key,**spec) and end['identity']['run']==stamp
            assert end['fingerprint']==digest(end['identity'])
            assert [h['step'] for h in history]==list(range(1,end['step']+1))
            assert all(math.isfinite(v) for h in history for v in h.values())
            losses.extend(dict(job=key,condition=spec.get('condition',key),stage=phase,**h) for h in history)
            phases.append(dict(job=key,stage=phase,step=end['step'],qualified=end['qualified'],reason=end['reason']))
            paths=sorted(p.parent.glob('dev_*.json')); assert [int(f.stem[4:]) for f in paths]==list(range(0,end['step']+1,32))
            selections=[]; passes=0
            for f in paths:
                step=int(f.stem[4:]); report=read(f)
                panel='policy_dev' if phase=='policy' else {'tiny':'tiny_one','one':'dev_one','mixed':'dev_mixed'}[phase]
                audit(report['task'],panel); audit(report['general'],'general_dev')
                if 'benchmark' in report:audit(report['benchmark'],'benchmark_dev')
                cases=report['task']['metrics']['by_case']
                d=dict(job=key,condition=spec.get('condition',key),lr_factor=spec.get('lr_factor',1.),stage=phase,step=step,
                    path_correct=report['task']['metrics']['path_correct'],general=report['general']['metrics']['path_correct'])
                if phase=='policy':
                    selection=source['policy_selection'](report,baseline,cfg,step); selections.append(selection)
                    d.update(selection); d['one_correct']=np.mean([v['path_correct'] for c,v in cases.items() if c.startswith('h1')])
                    d['two_allowed_correct']=np.mean([v['path_correct'] for c,v in cases.items() if c.startswith('h2') and c!='h2_blocked'])
                elif step:
                    threshold=cfg['tiny_threshold'] if phase=='tiny' else cfg['cap_threshold']
                    passed=all(v['path_correct']>=threshold for v in cases.values());passes=passes+1 if passed else 0
                development.append(d)
            assert read(paths[-1])==end['report']
            if phase=='policy':
                assert end['step']==cfg['policy_steps']==512
                best=read(p.parent/'best.json'); optimal=max(selections,key=lambda s:s['rank'])
                assert close(best['selection'],optimal) and best['step']==optimal['step']
                assert best['report']==read(p.parent/f"dev_{best['step']:04d}.json")
                assert end['qualified']==optimal['feasible']
                candidates.append(dict(job=key,condition=spec['condition'],lr_factor=spec['lr_factor'],**optimal))
            else:assert end['qualified']==(passes>=cfg['consecutive_passes'])
            if spec['geometry']!='full':
                final=load(p.parent/'final.pt'); resume=load(p.parent/'checkpoint.pt')
                assert final['identity']==resume['identity']==end['identity'] and final['step']==resume['step']==end['step']
                assert resume['history']==history and final['parameters'].keys()==resume['parameters'].keys()
                for name,v in final['parameters'].items():
                    assert torch.isfinite(v).all() and torch.equal(v,resume['parameters'][name])
                    if spec['geometry']=='native' and spec['placement']=='same' and name.endswith('deltas.1'):assert torch.count_nonzero(v)==0
                if phase=='policy':
                    best_state=load(p.parent/'best.pt')
                    assert best_state['selection']==optimal and best_state['report']==best['report']
                    assert all(torch.isfinite(v).all() for v in best_state['parameters'].values())
                checkpoint_pairs+=1
        if spec['kind']=='policy':
            initial=read(directory/'policy/dev_0000.json')['task']['rows']
            parent=read(ROOT/'jobs'/spec['teacher']/'acquisition.json')['dev_mixed']['rows']
            disagreements=sum(a['response']!=b['response'] for a,b in zip(initial,parent))
            assert len(initial)==len(parent) and disagreements==0
            assert done['identity']['parent']==read(ROOT/'jobs'/spec['teacher']/'DONE.json')['fingerprint']
            clones.append(dict(job=key,disagreements=disagreements))
    assert len(skipped)==3 and len(clones)==12 and len(frozen)==1
    ca=pd.DataFrame(candidates); ca['worst_task_quality']=[r[1] for r in ca['rank']]
    cv=pd.DataFrame(development); ls=pd.DataFrame(losses); ph=pd.DataFrame(phases)
    for condition,choice in recipes['recipes'].items():
        selected=ca[ca.condition==condition]
        if not len(selected):assert choice is None;continue
        expected=max(selected.to_dict('records'),key=lambda c:tuple(c['rank'])+(-c['lr_factor'],))
        assert choice['job']==expected['job'] and choice['lr_factor']==expected['lr_factor']
    masks={c:read(ROOT/'jobs'/recipes['recipes'][c]['job']/'layout.json')['layout'] for c in ['shared4','shared8','disjoint4x4']}
    for s8,d in zip(masks['shared8'],masks['disjoint4x4']):assert set(s8['indices'][0])==set(sum(d['indices'],[]))
    assert masks['shared4']==masks['disjoint4x4']

    endpoints=[]; cases=[]; reports={}; world_scores={}; general_audit=[]
    boot=np.random.default_rng(41105).integers(0,256,size=(2000,256))
    for path in sorted(ROOT.glob('jobs/*/test_*.json')):
        wrapped=read(path); key=path.parent.name; spec=jobs[key]; endpoint=wrapped['endpoint']; panels=wrapped['panels']
        assert wrapped['job_id']==key and wrapped['run']==stamp
        if spec['kind']!='base':assert wrapped['checkpoint_identity']['job']==dict(id=key,**spec)
        for k,panel in [('task','policy_test'),('general','general_test'),('benchmark','benchmark_test')]:audit(panels[k],panel)
        condition=spec.get('condition','base' if spec['kind']=='base' else 'native4' if spec['geometry']=='native' else 'scattered4' if spec['geometry']=='scattered' else 'full')
        role=key+'__'+endpoint; reports[role]=panels
        raw=pd.DataFrame(panels['task']['rows']); allowed=raw[raw.label!='blocked']; blocked=raw[raw.label=='blocked']
        per_world=allowed.groupby('world').path_correct.mean().sort_index().to_numpy(); world_scores[role]=per_world
        low,high=np.quantile(per_world[boot].mean(1),[.025,.975])
        row=dict(role=role,job=key,condition=condition,kind=spec['kind'],endpoint=endpoint,step=wrapped['step'],
            allowed_correct=allowed.path_correct.mean(),allowed_ci_low=low,allowed_ci_high=high,
            blocked_refusal=blocked.refusal.mean(),false_refusal=allowed.refusal.mean(),
            one_correct=allowed[allowed.depth==1].path_correct.mean(),two_allowed_correct=allowed[allowed.depth==2].path_correct.mean(),
            min_allowed_correct=min(v['path_correct'] for c,v in panels['task']['metrics']['by_case'].items() if c!='h2_blocked'),
            malformed=1-raw.format_valid.mean(),ref_prefix_malformed=np.mean([r['response'].startswith('REF') and not r['refusal'] for r in panels['task']['rows']]),
            general=panels['general']['metrics']['path_correct'],**panels['benchmark']['metrics'])
        for task in ['addition','copy']:
            selected=[r for r in panels['general']['rows'] if r['task']==task];row[task]=np.mean([r['path_correct'] for r in selected])
        for r in panels['general']['rows']:
            parsed=source['classify'](r['response'],r['answer'])['correct'] if r['task']=='addition' else r['path_correct']
            general_audit.append(dict(role=role,task=r['task'],id=r['id'],answer=r['answer'],response=r['response'],
                strict_correct=r['path_correct'],answer_parsed_correct=parsed,format_only_error=parsed and not r['path_correct'],
                refusal_language=source['refusal'](r['response']),truncated=r['truncated']))
        entries=[r for r in general_audit if r['role']==role]
        row['addition_parsed']=np.mean([r['answer_parsed_correct'] for r in entries if r['task']=='addition'])
        row['answer_parsed_general']=np.mean([r['answer_parsed_correct'] for r in entries])
        row['unique_trainable']=0 if spec['kind']=='base' else read(path.parent/'layout.json')['unique_trainable']
        row['development_feasible']=(read(path.parent/'policy/best.json')['selection']['feasible'] if spec['kind']=='policy' else read(path.parent/'DONE.json')['qualified'])
        endpoints.append(row)
        for case,values in panels['task']['metrics']['by_case'].items():cases.append(dict(role=role,condition=condition,kind=spec['kind'],endpoint=endpoint,case=case,**values))
    df=pd.DataFrame(endpoints); cs=pd.DataFrame(cases); assert len(df)==12
    pairs=[]
    for left,right in [('shared8','disjoint4x4'),('shared4','disjoint4x4')]:
        a=df[(df.kind=='policy')&(df.condition==left)&(df.endpoint=='best')].iloc[0]
        b=df[(df.kind=='policy')&(df.condition==right)&(df.endpoint=='best')].iloc[0]
        delta=world_scores[a.role]-world_scores[b.role];ci=np.quantile(delta[boot].mean(1),[.025,.975])
        pairs.append(dict(comparison=left+' minus '+right,allowed_difference_pp=100*delta.mean(),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1]))

    observations={}
    for path in ROOT.glob('observations/*/*/activations.pt'):observations[path.parent.parent.name+'__'+path.parent.name]=load(path)
    assert len(observations)==12
    base=observations['base__final']; nfit=len(old['fit']); blocked_mask=torch.tensor([r['label']=='blocked' for r in data['direction_observe']])
    assert base['behavior_partition_available']
    direction=[]; old_behavior=[]; obs_behavior=[]; errors=[]
    for role,artifact in observations.items():
        key,endpoint=role.split('__'); spec=jobs[key]
        alignment=read(ROOT/'observations'/key/endpoint/'alignment.json')
        assert artifact['parameter_hash']==artifact['parameter_hash_after'] and artifact['observational_only']
        assert alignment['observational_only'] and not alignment['direction_penalty_used'] and not alignment['causal_ablation_performed']
        assert artifact['behavior']==alignment['old_behavior'] and artifact['new_behavior']==alignment['new_behavior']
        assert len(artifact['behavior'])==len(old['fit'])+len(old['heldout'])==144
        assert len(artifact['new_behavior'])==len(data['direction_observe'])==96
        assert all(r['refusal_language']==source['refusal'](r['response']) for r in artifact['behavior']+artifact['new_behavior'])
        ownmask=torch.tensor([r['refusal_language'] for r in artifact['behavior'][:nfit]])
        assert torch.equal(ownmask,artifact['fit_refused'])
        enough=bool(ownmask.sum()>=8 and (~ownmask).sum()>=8);assert artifact['behavior_partition_available']==enough
        basis_mask=ownmask if enough else torch.tensor([r['label']=='unsafe' for r in old['fit']])
        parent=observations[spec['teacher']+'__final'] if spec['kind']=='policy' else base
        for split,rows in [('fit',artifact['behavior'][:nfit]),('heldout',artifact['behavior'][nfit:])]:
            for r,expected in zip(rows,old[split]):assert all(r[k]==v for k,v in expected.items())
            for label in ['safe','unsafe']:
                selected=[r for r in rows if r['label']==label]
                old_behavior.append(dict(role=role,split=split,label=label,n=len(selected),refusal_language=np.mean([r['refusal_language'] for r in selected])))
        generated=[]
        for r,expected in zip(artifact['new_behavior'],data['direction_observe']):
            assert all(r[k]==v for k,v in expected.items())
            generated.append(dict(r,**source['score'](r,r['response'])))
        for label in ['allowed','blocked']:
            selected=[r for r in generated if r['label']==label]
            obs_behavior.append(dict(role=role,label=label,n=len(selected),path_correct=np.mean([r['path_correct'] for r in selected]),refusal=np.mean([r['refusal'] for r in selected])))
        actual_mask=torch.tensor([r['refusal'] for r in generated]); actual_enough=bool(actual_mask.sum()>=8 and (~actual_mask).sum()>=8)
        for i,record in enumerate(alignment['layers']):
            h=artifact['new'][i]; old_h=artifact['old'][i]
            assert h.shape==(96,896) and old_h.shape==(144,896) and torch.isfinite(h).all() and torch.isfinite(old_h).all()
            assert torch.allclose(artifact['basis'][i],contrast(old_h[:nfit],basis_mask),atol=1e-6,rtol=1e-5)
            induced=contrast(h,blocked_mask)-contrast(parent['new'][i],blocked_mask)
            c=cosine(induced,base['basis'][i]); fixed=cosine(contrast(old_h[:nfit],base['fit_refused']),base['basis'][i])
            expected=dict(cosine_new_vs_original=cosine(contrast(h,blocked_mask),base['basis'][i]),cosine_induced_vs_original=c,
                induced_energy_fraction_along_original=c*c if c is not None else None,cosine_fixed_partition_old_vs_original=fixed,induced_contrast_norm=float(induced.norm()))
            for k,v in expected.items():
                assert (v is None and record[k] is None) or (v is not None and record[k] is not None and math.isclose(v,record[k],abs_tol=1e-6,rel_tol=1e-5)),(role,i,k,v,record[k])
                if k.startswith('cosine') and v is not None:errors.append(abs(v-record[k]))
            direction.append(dict(role=role,block=i+1,**record,actual_behavior_cosine=cosine(contrast(h,actual_mask),base['basis'][i]) if actual_enough else None))
            if spec.get('geometry')=='native' and i<11:
                assert torch.equal(h,base['new'][i]) and torch.equal(old_h,base['old'][i])
    dr=pd.DataFrame(direction); ob=pd.DataFrame(old_behavior); nb=pd.DataFrame(obs_behavior); pc=pd.DataFrame(pairs)
    costs=ls.groupby('job').agg(updates=('step','size'),train_seconds=('train_seconds','sum')).reset_index()
    costs['allocated_gpu_train_seconds']=[r.train_seconds*(2 if jobs[r.job].get('geometry')=='full' else 1) for r in costs.itertuples()]
    for name,table in [('endpoints',df),('cases',cs),('development',cv),('losses',ls),('phases',ph),('lr_calibration',ca),('fit_costs',costs),
                       ('directions',dr),('old_refusal_behavior',ob),('observation_behavior',nb),('paired_comparisons',pc),
                       ('general_scoring_audit',pd.DataFrame(general_audit))]:table.to_csv(OUT/(name+'.csv'),index=False)
    audit_result=dict(completed_utc=datetime.fromtimestamp(status['time'],timezone.utc).isoformat(),checks_passed=True,
        scheduled_jobs=19,executed_jobs=16,skipped_policy_jobs=skipped,policy_fits=12,seeds_run=[31],replication_gate_passed=False,
        evaluation_reports_recounted=evaluations,records_recounted=records,final_endpoint_reports=12,
        final_test_records=12*(2304+64+128),source_hashes_verified=len(manifest['sources']),excluded_previous_worlds=len(excluded),
        archive_sha256='23e5567f41321c3044337c2999ec1b89dcad6dd4af536b7cad54128e5230bf80',
        initial_parent_clones_verified=len(clones),sparse_checkpoint_pairs_verified=checkpoint_pairs,
        observational_hash_pairs_verified=12,direction_rows_recomputed=len(dr),max_cosine_recompute_error=max(errors),
        bootstrap=dict(replicates=2000,unit='held-out world',includes_training_seed_variation=False))
    (OUT/'audit.json').write_text(json.dumps(audit_result,indent=2))
    plot(df,cs,cv,ca,ph,dr,ob,recipes)
    write_report(df,cs,ca,ph,dr,pc,audit_result,recipes)
    print(json.dumps(audit_result,indent=2))
    print(df[df.kind.isin(['base','policy'])][['condition','endpoint','allowed_correct','blocked_refusal','general']].to_string(index=False))


def plot(df,cs,cv,ca,ph,dr,ob,recipes):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    policies=['shared4','shared8','disjoint4x4','full']
    selected=df[(df.kind=='policy')&(df.endpoint=='best')].set_index('condition')
    base=df[df.kind=='base'].iloc[0]
    selected=pd.concat([pd.DataFrame([base],index=['base']),selected])
    page=0
    with PdfPages(OUT/'results.pdf') as pdf:
        def save(fig,title,note):
            nonlocal page
            page+=1;fig.suptitle(title,fontsize=16,fontweight='bold',y=.98)
            fig.text(.02,.015,note,fontsize=9,va='bottom');fig.tight_layout(rect=(0,.08,1,.94))
            pdf.savefig(fig);fig.savefig(OUT/f'page_{page:02d}.png',dpi=110);plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(13,5))
        for ax,column,title in zip(axs,['allowed_correct','blocked_refusal','general'],['Allowed paths correct','Blocked cases refused','Exact addition + copy']):
            for i,c in enumerate(['base',*policies]):
                r=selected.loc[c];p=r[column]
                lo,hi=(r.allowed_ci_low,r.allowed_ci_high) if column=='allowed_correct' else wilson(p,256 if column=='blocked_refusal' else 64)
                ax.bar(i,100*p,color=COLORS[c],width=.65);ax.errorbar(i,100*p,yerr=[[max(0,100*(p-lo))],[max(0,100*(hi-p))]],color='black',capsize=3)
                ax.text(i,100*hi+2,f'{100*p:.1f}',ha='center',fontsize=9)
            ax.set_xticks(range(5),['Base','Shared 4','Shared 8','Disjoint 4+4','Full'],rotation=25);ax.set_ylim(0,112);ax.set_title(title);ax.set_ylabel('%')
        save(fig,'Mixed-hop learning: both shared-channel conditions pass validation',
             'Development-selected endpoints; disjoint/full are infeasible fallback checkpoints. 256 held-out worlds; 2,048 allowed cases.\nAllowed bars: world-cluster bootstrap 95% intervals. Other bars: Wilson intervals. Neither includes training-seed uncertainty.')
        fig,ax=plt.subplots(figsize=(13,6))
        order=['base',*policies]; values=[]
        for c in order:
            table=cs[cs.role==selected.loc[c,'role']].set_index('case')
            values.append([100*table.loc[k,'refusal' if k=='h2_blocked' else 'path_correct'] for k in CASES])
        matrix=np.array(values);im=ax.imshow(matrix,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        ax.set_xticks(range(9),['1: no marker','1: start','1: endpoint','1: outside','2: no marker','2: start','2: endpoint','2: outside','2: BLOCKED'],rotation=25,ha='right')
        ax.set_yticks(range(5),[NAMES[c] for c in order])
        for (y,x),v in np.ndenumerate(matrix):ax.text(x,y,f'{v:.1f}',ha='center',va='center',color='white' if v<65 else 'black')
        fig.colorbar(im,ax=ax,label='Correct required behavior (%)',fraction=.025)
        save(fig,'The earlier one-hop and unmarked-input weaknesses are substantially reduced',
             'Eight allowed cases require exact paths; the blocked two-hop case requires REFUSE. Each cell has 256 worlds.\nA marked one-hop endpoint is allowed, while the same node is the intermediate blocked node for two hops.')
        fig,axs=plt.subplots(2,2,figsize=(13,8))
        for c in policies:
            job=recipes['recipes'][c]['job'];s=cv[(cv.job==job)&(cv.stage=='policy')];best=recipes['recipes'][c]['selection']['step']
            for ax,col,title in zip(axs.flat,['one_correct','two_allowed_correct','blocked_refusal','general'],['One-hop allowed accuracy','Two-hop allowed accuracy','Blocked refusal','Collateral exact-output accuracy']):
                ax.plot(s.step,100*s[col],label=NAMES[c],color=COLORS[c]);b=s[s.step==best].iloc[0]
                ax.scatter([best],[100*b[col]],marker='*',s=100,color=COLORS[c]);ax.set_title(title);ax.set_ylim(-3,103);ax.set_xlabel('Policy updates');ax.set_ylabel('%');ax.grid(alpha=.15)
        axs[0,0].legend(fontsize=9)
        save(fig,'All trained policy candidates received the full 512-update horizon',
             'Shown: the chosen learning rate per condition. Stars mark development-selected checkpoints.\nNative conditions start from identical effective capability weights. Three learning rates per condition were calibrated equally.')
        fig,axs=plt.subplots(1,3,figsize=(13,5))
        for ax,col,title in zip(axs,['allowed_correct','blocked_refusal','general'],['Allowed paths','Blocked refusal','Exact addition + copy']):
            for j,endpoint in enumerate(['best','final']):
                vals=[df[(df.kind=='policy')&(df.condition==c)&(df.endpoint==endpoint)].iloc[0][col]*100 for c in policies]
                ax.bar(np.arange(4)+(j-.5)*.35,vals,.35,label=endpoint)
            ax.set_xticks(range(4),['Shared 4','Shared 8','Disjoint','Full'],rotation=20);ax.set_title(title);ax.set_ylim(0,107);ax.set_ylabel('%')
        axs[0].legend()
        save(fig,'Saving best checkpoints matters: final-step behavior can regress',
             'Best means the predeclared development ranking, not the highest final-test score.\nDisjoint and full had no feasible candidate; their best checkpoints remain explicitly marked as fallbacks.')
        fig,axs=plt.subplots(1,3,figsize=(13,5))
        for stage,ax in zip(['tiny','one','mixed'],axs):
            for key,c in [('cap_native4_s31','native4'),('cap_scattered4_s31','scattered4'),('cap_full_s31','full')]:
                s=cv[(cv.job==key)&(cv.stage==stage)]
                if len(s):ax.plot(s.step,100*s.path_correct,marker='o',ms=3,label=NAMES[c],color=COLORS[c])
            ax.set_title({'tiny':'Tiny one-hop acquisition','one':'Fresh one-hop acquisition','mixed':'Mixed one-/two-hop acquisition'}[stage]);ax.set_xlabel('Stage updates');ax.set_ylabel('Overall exact path accuracy (%)');ax.set_ylim(-3,103)
            ax.axhline(98 if stage=='tiny' else 90,color='gray',ls='--',lw=.8)
        axs[0].legend(fontsize=8)
        save(fig,'Scattered weights were stopped by a case-specific acquisition gate',
             'The scattered tiny-set score reached 98.86% overall, but marked-start accuracy was 14/16 (87.5%).\nEvery case needed 98% twice consecutively; no scattered mixed-hop acquisition or policy fit was therefore trained.')
        fig,axs=plt.subplots(1,2,figsize=(13,6))
        cols=['addition','addition_parsed','copy','arc_easy_correct','winogrande_correct'];values=selected.loc[['base',*policies],cols].to_numpy()*100
        im=axs[0].imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto');axs[0].set_yticks(range(5),[NAMES[c] for c in ['base',*policies]])
        axs[0].set_xticks(range(5),['Addition: exact','Addition: parsed','Copy: exact','ARC-Easy','WinoGrande'],rotation=30,ha='right')
        for (y,x),v in np.ndenumerate(values):axs[0].text(x,y,f'{v:.1f}',ha='center',va='center',color='white' if v<65 else 'black')
        fig.colorbar(im,ax=axs[0],fraction=.035,label='Accuracy (%)')
        for task,marker in [('arc_easy','o'),('winogrande','s')]:axs[1].plot(range(5),selected.loc[['base',*policies],task+'_choice_mass'],marker=marker,label=task)
        axs[1].set_xticks(range(5),['Base','Shared 4','Shared 8','Disjoint','Full'],rotation=20);axs[1].set_ylabel('Mean probability on valid option letters');axs[1].set_ylim(0,1.05);axs[1].legend()
        save(fig,'Collateral losses include formatting errors: separate answers from exact compliance',
             'Disjoint arithmetic: 12.5% exact but 90.6% answer-parsed; most failures were correct equations instead of bare integers.\nParsed arithmetic is a post-hoc diagnostic; original gates are unchanged. Neither score establishes irreversible information loss.')
        fig,axs=plt.subplots(1,2,figsize=(13,6))
        for ax,col,title in zip(axs,['worst_task_quality','general_change_pp'],['Best checkpoint: worst task-case quality','Best checkpoint: collateral change versus base']):
            for i,c in enumerate(policies):
                s=ca[ca.condition==c].sort_values('lr_factor',ascending=False)
                vals=100*s[col] if col=='worst_task_quality' else s[col]
                ax.plot([0,1,2],vals,marker='o',label=NAMES[c],color=COLORS[c])
            ax.set_xticks([0,1,2],['1× LR','0.3× LR','0.1× LR']);ax.set_title(title);ax.set_ylabel('%' if col=='worst_task_quality' else 'percentage points');ax.grid(alpha=.15)
        axs[0].legend(fontsize=9)
        save(fig,'Lowering the disjoint learning rate helped, but did not yield a feasible fit',
             'All three LR candidates received 512 updates. Base sparse LR = 3e-4; full LR = 1e-5.\nFull-model refusal/capability behavior can be excellent while collateral and benchmark retention still fail.')
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for c in policies:
            s=dr[dr.role==selected.loc[c,'role']]
            axs[0].plot(s.block,s.cosine_induced_vs_original,label=NAMES[c],color=COLORS[c])
            axs[1].plot(s.block,s.cosine_fixed_partition_old_vs_original,color=COLORS[c])
        axs[0].set_ylabel('Cosine: induced contrast vs original candidate');axs[1].set_ylabel('Cosine: fixed-partition old contrast vs original')
        for ax in axs:ax.set_xlabel('Transformer block output');ax.grid(alpha=.15)
        axs[0].legend(fontsize=9);axs[1].set_ylim(-.05,1.05)
        save(fig,'Direction geometry remains observational and is not a circuit-isolation test',
             'New-task contrasts use matched two-hop inputs and subtract the capability-parent contrast. Old comparisons use\nthe frozen baseline refused/answered partition. No penalty, projection, ablation, direction selection or steering was used.')


def write_report(df,cs,ca,ph,dr,pc,audit,recipes):
    best=df[(df.kind=='policy')&(df.endpoint=='best')].set_index('condition');base=df[df.kind=='base'].iloc[0]
    pct=lambda x:f'{100*x:.1f}%'
    lines=['# Mixed-hop capability/refusal results — 4 November 2026','',
      '**Main finding:** shared native updates learned selective refusal while retaining both one-hop and two-hop performance. Shared eight outperformed disjoint four-plus-four at the same total trainable scalar budget, on exactly the same eight channel locations and from the same initial effective capability weights. This is a one-seed, fixed-recipe comparison, not proof that every disjoint recipe fails.','',
      f"Completed {audit['completed_utc']} on `mn866@weatherwax.cs.bath.ac.uk`. All 19 scheduled job entries reached terminal status without worker errors: 15 trained fits plus the frozen-base job ran, and three scattered policy jobs were skipped after acquisition failed. All 12 final endpoint reports completed. Seeds 47 and 71 were **not run**, because the prespecified all-condition validation gate failed.",'',
      '## Held-out selected-checkpoint scores','',
      '| Model | Allowed paths | Blocked refusal | False refusal | Exact addition/copy | ARC-Easy | WinoGrande | Feasible on development? |',
      '|---|---:|---:|---:|---:|---:|---:|---|']
    for c,r in [('base',base),*[(c,best.loc[c]) for c in ['shared4','shared8','disjoint4x4','full']]]:
        lines.append('| '+NAMES[c]+' | '+' | '.join(pct(r[k]) for k in ['allowed_correct','blocked_refusal','false_refusal','general','arc_easy_correct','winogrande_correct'])+' | '+('reference' if c=='base' else str(bool(r.development_feasible)))+' |')
    lines+=['',
      'The test has 256 independent graph worlds, each with eight allowed cases and one blocked case: **2,048 allowed cases and 256 blocked cases**. Each collateral generation task has 32 prompts; each benchmark has 64 questions. Allowed scores include unmarked inputs. Aggregated allowed-case uncertainty uses a 2,000-resample bootstrap clustered by world, because cases from the same graph are correlated. It does not measure training-seed uncertainty.','',
      'The selected native learning rate was 3e-4 for both shared conditions and 3e-5 for disjoint. Full selected 3e-6. Every condition with a qualified capability parent received the same three-factor search and 512 updates per trained candidate. Full/disjoint had **no feasible checkpoint**: their displayed selected models are explicitly infeasible fallbacks. Selection used development data only, with no final-test or direction metric.','',
      '## Capability-only controls','',
      '| Before policy training | One-hop paths | Two-hop allowed paths | Exact addition/copy |',
      '|---|---:|---:|---:|']
    for _,r in df[df.kind!='policy'].iterrows():
        name='Full capability' if r.condition=='full' else NAMES[r.condition]
        lines.append('| '+name+' | '+' | '.join(pct(r[k]) for k in ['one_correct','two_allowed_correct','general'])+' |')
    lines+=['',
      'Native capability training acquired both depths while keeping the small collateral panel close to base. Full capability training acquired both depths but already incurred collateral losses before policy learning. The scattered endpoint only completed tiny one-hop training; it never received the later curriculum stages.','',
      '## One-hop failure was substantially repaired','',
      '| Selected model | One-hop allowed | Two-hop allowed | Worst allowed case |',
      '|---|---:|---:|---:|']
    for c,r in best.iterrows():lines.append('| '+NAMES[c]+' | '+' | '.join(pct(r[k]) for k in ['one_correct','two_allowed_correct','min_allowed_correct'])+' |')
    lines+=['',
      'Unlike the previous two-hop-only curriculum, one-hop examples remain in acquisition and policy training. The marked one-hop endpoint must be answered, while the same node is the intermediate blocked node for a two-hop request. This forces the trained decision to distinguish the requested depth.','']
    for c in ['shared4','shared8']:
        row=cs[(cs.role==best.loc[c,'role'])&(cs['case']=='h1_endpoint')].iloc[0]
        lines.append(f"- {NAMES[c]}: marked one-hop endpoint accuracy {pct(row.path_correct)}, false refusal {pct(row.refusal)}.")
    lines+=['',
      'This substantially repairs the previous pilot\'s one-hop failure. Comparisons with that pilot are descriptive: the data, curriculum, conservation during acquisition and checkpoint-selection procedure changed together. Only depths one and two and this prompt family were tested; this is not evidence of general arbitrary-depth reasoning.','',
      '## Matched native comparison','',
      'Shared4 and disjoint4+4 share their four capability channels and the same capability checkpoint. Shared8 expands that checkpoint with four zero-initialized updates, on exactly the union of disjoint C and R locations. All three have identical initial effective model weights. Shared8 and disjoint4+4 each train **21,504 scalars**; shared4 trains **10,752**. The difference is which losses can change which channels, not the starting weights or the eight-channel support. All increments remain active at inference; there is no oracle forward router.','']
    for row in pc.to_dict('records'):lines.append(f"- {row['comparison']}: allowed accuracy difference **{row['allowed_difference_pp']:.1f} percentage points**, paired world-bootstrap 95% interval [{row['ci_low_pp']:.1f}, {row['ci_high_pp']:.1f}]. This is conditional on these trained models, not across-seed evidence.")
    lines+=['',
      'Sharing allows both objectives to modify the full shared support; disjoint routing restricts each objective to its own half. The result supports an advantage for sharing under this recipe. It does not establish that spatial separation is inherently impossible: independent C/R learning rates, gradient scaling, grammar-loss strength, different channel draws or other optimization schedules were not exhausted. Parameter matching is not a blanket claim of matched compute; full-model training uses two GPUs and has a different backward cost.','',
      '## What happened to the controls','',
      'The disjoint fit improved over the previous total-collapse result but remained poor on the policy task: selected held-out allowed accuracy 67.4%, blocked refusal 50.4%, and false refusal 24.7%. The lower LR helped; none of the three candidates passed the joint task/retention criteria. Its final step was worse than its selected checkpoint. This is a training/coexistence failure under the tested recipe, not a clean successful disjoint comparator.','',
      '**Scoring caveat:** the prespecified arithmetic/copy metric requires exact output strings. Disjoint scored 42.2% on that metric, but most arithmetic failures were correct equations such as `15 + 23 = 38`, instead of the requested bare integer. Reusing the existing conservative answer parser gives **90.6% arithmetic answer accuracy, versus 12.5% exact-format accuracy**. Combining parsed arithmetic with exact copying gives 81.25%, versus 87.5% for base. This is a substantial formatting failure, not evidence that its arithmetic ability fell to 12.5%. Copying also declined from 26/32 to 23/32 exact; nine copy responses were truncated refusal-language outputs. The parsed score is a post-hoc diagnostic only: it does not change the original validation gate or checkpoint choices.','',
      'The graph-policy gap is not explained away by that arithmetic-format issue: 24.7% of allowed graph cases were explicitly refused, while only 2.2% of graph-policy outputs were malformed. Its poor selectivity remains a genuine task-behavior failure under this recipe.','',
      'Full-model training **did learn the requested behavior**: selected test allowed accuracy 99.95% and blocked refusal 100%; its final endpoint scored 100% on both. It failed the retention criteria instead. The selected full model scored 70.3% on arithmetic/copy versus 87.5% for base; both selected benchmark accuracies fell 15.6 points. Full-model capability acquisition already had collateral costs, so compare its capability baseline as well as its policy endpoint. These scores do not establish irreversible information erasure.','',
      'Scattered acquisition reached 174/176 correct tiny-set examples (98.86%) at update 512. However, the marked-start case was only **14/16 (87.5%)**. The predeclared tiny gate required 98% in every case twice consecutively; with 16 examples per special case, that effectively requires 16/16. It therefore never reached fresh one-hop or mixed-hop training, and its three policy candidates were skipped. Its held-out one-hop accuracy was already 94.6%, indicating substantial lookup learning despite the gate failure. This strict, small-sample gate limits the scattered comparison: do not interpret it as proof that unstructured weights cannot acquire the task.','',
      '## Best versus final','',
      '| Model | Selected step | Selected allowed / blocked | Final allowed / blocked |',
      '|---|---:|---:|---:|']
    for c,r in best.iterrows():
        final=df[(df.kind=='policy')&(df.condition==c)&(df.endpoint=='final')].iloc[0]
        lines.append(f'| {NAMES[c]} | {int(r.step)} | {pct(r.allowed_correct)} / {pct(r.blocked_refusal)} | {pct(final.allowed_correct)} / {pct(final.blocked_refusal)} |')
    lines+=['',
      'Saving both endpoints prevents presenting a transient last-step result as the whole learning story. All candidate development predictions, losses, learning rates and gradient norms remain available. The first calibration phase stopped exactly as specified after validation failed; no validation thresholds were relaxed after seeing these results.','',
      '## Observational direction measurements','',
      'Directions were never optimized, projected out, ablated or used for checkpoint/rate selection. Raw prompt-end block outputs before final RMSNorm were recorded at all 24 layers. The new-policy contrast uses 48 matched two-hop allowed-endpoint/blocked-intermediate graph pairs, subtracting the capability-parent contrast. The old candidate comes from baseline refusal-language versus non-refusal on independent XSTest calibration prompts; the old comparison keeps the same baseline behavior partition.','',
      '| Selected shared model, final block | Induced contrast cosine to old candidate | Squared-cosine fraction | Fixed-partition old contrast stability |',
      '|---|---:|---:|---:|']
    for c in ['shared4','shared8']:
        d=dr[(dr.role==best.loc[c,'role'])&(dr.block==24)].iloc[0]
        lines.append(f'| {NAMES[c]} | {d.cosine_induced_vs_original:.5f} | {100*d.induced_energy_fraction_along_original:.5f}% | {d.cosine_fixed_partition_old_vs_original:.5f} |')
    lines+=['',
      'These measurements show low alignment of the induced task contrast with this particular old candidate, alongside substantial preservation of the old fixed-partition contrast. They **do not prove an independent causal refusal circuit, that the old mechanism is unused, or that refusal and capability share an indivisible circuit**. Prompt content, response format, token position and nonlinear downstream computation remain confounds. New-prompt generated responses, old-prompt refusal-language diagnostics and raw activations are saved. Natural-safety rates are heuristic diagnostics, not a comprehensive safety evaluation.','',
      '## Audit and retained data','',
      f"Recounted **{audit['records_recounted']:,} records across {audit['evaluation_reports_recounted']} evaluation reports**, including {audit['final_test_records']:,} final-test records. Verified nine source hashes, both frozen data hashes, all world/label splits and 1,856 excluded worlds, all 12 initial policy-parent response comparisons, best-checkpoint rankings for every trained LR candidate, recipe selection, acquisition gates, equal 512-update policy horizons, {audit['sparse_checkpoint_pairs_verified']} sparse final/resume checkpoint pairs, and 12 unchanged observational parameter-hash pairs / 288 per-block direction records. All checks passed. Full-model tensors remain remote and were not exhaustively compared locally.",'',
      'All compact JSON/log measurements, raw activation tensors and sparse checkpoints are local under `../results/weatherwax-completed-20261104`. Large model weights, optimizer states and teacher caches remain on Weatherwax at `/mnt/fast1/mn866/casr-colocalization-mixed-20261104/results`. The downloaded archive checksum is `23e5567f41321c3044337c2999ec1b89dcad6dd4af536b7cad54128e5230bf80`. No old run was overwritten.','',
      '## Recommended next decision','',
      'The shared fits are now credible candidates for replication, but the missing scattered policy control and unstable disjoint fit still limit the location claim. First diagnose the disjoint objective balance and allow the near-fitting scattered model to finish acquisition under a new, prospectively declared budget. Keep the mixed-depth data and matched native initialization. Then repeat the repaired comparison across seeds. Do not claim attack resistance: no refusal-removal attack has been evaluated here. No new training was launched during this analysis.','',
      '## Files','',
      '- [Eight-page graphs](results.pdf)','- [Endpoint table](endpoints.csv)','- [Per-case results](cases.csv)',
      '- [All LR candidates](lr_calibration.csv)','- [Development curves](development.csv)','- [Losses](losses.csv)',
      '- [Fit training-time costs, excluding evaluation and checkpoint I/O](fit_costs.csv)',
      '- [Direction comparisons](directions.csv)','- [Observation-set responses scored](observation_behavior.csv)',
      '- [Exact versus parsed collateral scores and responses](general_scoring_audit.csv)',
      '- [Paired world comparisons](paired_comparisons.csv)','- [Audit receipt](audit.json)','',
      'Reproduce from the project root with `/usr/bin/python3 experiments/refusal_colocalization_v3/analyse.py`.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
