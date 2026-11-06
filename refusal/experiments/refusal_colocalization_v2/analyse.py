"""Audit the frozen v2 observations and produce local tables and figures; no model updates."""
import ast
from collections import Counter
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
from scipy.stats import rankdata
import torch

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results/weatherwax-20261104'
ROOT=ARCHIVE/'results'
OUT=HERE/'analysis'
NAMES={'base':'Frozen base','native4':'Capability: 4 channels','native128':'Capability: 128 channels',
       'full_on':'Capability: full + conservation','full_off':'Capability: full, no conservation',
       'scattered_cap':'Scattered: tiny gate failed','policy_same':'Shared 4 channels',
       'policy_different':'Disjoint 4 + 4 channels','policy_full':'Full-model policy'}
COLORS={'policy_same':'#2563eb','policy_different':'#d97706','policy_full':'#7c3aed',
        'base':'#52525b','native4':'#0f766e','native128':'#e11d48',
        'full_on':'#7c3aed','full_off':'#b91c1c','scattered_cap':'#a16207'}


def read(path):return json.loads(path.read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def load(path):return torch.load(path,map_location='cpu',weights_only=True)
def cosine(a,b):
    den=float(a.norm()*b.norm())
    return max(-1.,min(1.,float(a@b/den))) if den>1e-10 else None
def contrast(h,mask):return h[mask].mean(0)-h[~mask].mean(0)
def close(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a)
    if isinstance(a,(int,float)):return math.isclose(a,b,abs_tol=1e-8,rel_tol=1e-6)
    return a==b
def auc(y,s):
    y=np.asarray(y,dtype=bool); n=int(y.sum()); m=len(y)-n
    return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None
def wilson(p,n):
    z=1.96; center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half,center+half


def main():
    if not ROOT.exists():raise SystemExit('Raw measurements were removed during the 5 November cleanup. See refusal/RAW_DATA.md to restore them; analysed tables and plots remain in this study.')
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    manifest=read(ROOT/'manifest.json'); cfg=manifest['protocol']; stamp=digest(manifest)
    assert cfg['direction_penalty']==0 and not cfg['direction_intervention'] and cfg['observational_only']
    for name,sha in manifest['sources'].items():
        assert hashlib.sha256((ARCHIVE/'source'/name).read_bytes()).hexdigest()==sha
    data=read(ROOT/'data.json'); old=read(ROOT/'direction_data.json')
    assert digest(data)==manifest['data_sha256'] and digest(old)==manifest['observation_sha256']
    status=read(ROOT/'status.json')
    assert status['training_complete'] and status['completed_training']==status['planned_training']==9
    assert not status['running'] and not status['failures']
    jobs=read(ROOT/'jobs.json'); alljobs=['base',*jobs]
    assert set(old['fit'][i]['group'] for i in range(len(old['fit']))).isdisjoint(r['group'] for r in old['heldout'])
    excluded=set(read(ARCHIVE/'source/excluded_worlds.json'))
    banks={}
    for name,panel in data.items():
        if name.startswith('benchmark'):continue
        assert len({r['id'] for r in panel})==len(panel)
        for row in panel:
            if row['task']!='pointer':continue
            assert digest(row['edges'])==row['world'] and row['world'] not in excluded
            path=[]; node=row['start']
            for _ in range(row['depth']):node=row['edges'][node]; path.append(node)
            assert row['path']==path and row['answer']==' '.join(path)
            assert (row['marker'] in path[:-1])==(row['label']=='blocked')
            banks.setdefault(row['id'].split('_')[0],set()).add(row['world'])
    assert sum(map(len,banks.values()))==len(set().union(*banks.values()))
    general_prompts=[r['prompt'] for key in ['rehearsal','general_dev','general_test'] for r in data[key]]
    assert len(general_prompts)==len(set(general_prompts))
    assert {r['id'] for r in data['benchmark_dev']}.isdisjoint(r['id'] for r in data['benchmark_test'])
    tree=ast.parse((ARCHIVE/'source/run.py').read_text())
    scoring={'re':re}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
        and n.name in ('score','summarize')],type_ignores=[]),'frozen_scoring','exec'),scoring)
    observation_tree=ast.parse((ARCHIVE/'source/observe.py').read_text())
    exec(compile(ast.Module(body=[n for n in observation_tree.body if isinstance(n,ast.FunctionDef)
        and n.name=='refusal'],type_ignores=[]),'frozen_refusal_heuristic','exec'),scoring)
    evaluations=records=0

    def audit(report,panel):
        nonlocal evaluations,records
        expected={r['id']:r for r in data[panel]}; rows=report['rows']
        assert len(rows)==len(expected) and {r['id'] for r in rows}==expected.keys()
        for r in rows:
            assert all(r[k]==v for k,v in expected[r['id']].items())
            if panel.startswith('benchmark'):
                lp=np.array(r['choice_logprobs'])
                assert np.isfinite(lp).all() and r['predicted']==int(lp.argmax())
                assert r['correct']==(r['predicted']==r['gold'])
                assert close(r['gold_nll'],-lp[r['gold']]) and close(r['choice_mass'],np.exp(lp).sum())
            else:
                assert all(r[k]==v for k,v in scoring['score'](r,r['response']).items())
                assert isinstance(r['truncated'],bool)
        if panel.startswith('benchmark'):
            metrics={task+'_'+k:np.mean([r[k] for r in rows if r['task']==task])
                     for task in sorted({r['task'] for r in rows}) for k in ['correct','gold_nll','choice_mass']}
        else:metrics=scoring['summarize'](rows)
        assert close(metrics,report['metrics'])
        evaluations+=1; records+=len(rows)

    for panel,report in read(ROOT/'base.json').items():audit(report,panel)
    endpoints=[]; curves=[]; losses=[]; phases=[]; reports={}; hoprows=[]; collateral=[]
    states=0; frozen=set(); clone_checks=[]
    for job in alljobs:
        directory=ROOT/'jobs'/job
        wrapped=read(directory/'test.json'); assert wrapped['job_id']==job and wrapped['run']==stamp
        report=wrapped['panels']; reports[job]=report
        for key,panel in [('one','test_one'),('two','test_two'),('policy','policy_test'),('general','general_test'),('benchmark','benchmark_test')]:audit(report[key],panel)
        m=report['policy']['metrics']; a=m['by_label']['allowed']; b=m['by_label']['blocked']
        end=dict(job=job,condition=NAMES[job],allowed_correct=a['path_correct'],blocked_refusal=b['refusal'],
                 false_refusal=a['refusal'],joint_score=m['joint_score'],
                 two_path=report['two']['metrics']['path_correct'],one_path=report['one']['metrics']['path_correct'],
                 general=report['general']['metrics']['path_correct'],
                 unmarked_correct=report['two']['metrics']['by_label']['unmarked']['path_correct'],
                 unmarked_refusal=report['two']['metrics']['by_label']['unmarked']['refusal'],
                 **report['benchmark']['metrics'])
        end['malformed_ref_prefix']=np.mean([r['response'].startswith('REF') and not r['refusal'] for r in report['policy']['rows']])
        for task in ['addition','copy']:
            rows=[r for r in report['general']['rows'] if r['task']==task]
            end[task]=np.mean([r['path_correct'] for r in rows])
        for r in report['one']['rows']:
            nodes=re.findall('[a-j]',r['response']) if re.fullmatch(r'[a-j](?:(?:\s+|\s*(?:->|→|,)\s*)[a-j])*',r['response']) else []
            hoprows.append(dict(job=job,id=r['id'],marker_is_endpoint=r['marker']==r['path'][0],
                nodes=len(nodes),first_node_correct=bool(nodes and nodes[0]==r['path'][0]),refusal=r['refusal']))
        for task in ['addition','copy','arc_easy','winogrande']:
            rows=[r for r in report['general' if task in ['addition','copy'] else 'benchmark']['rows'] if r['task']==task]
            collateral.append(dict(job=job,task=task,n=len(rows),accuracy=end[task if task in ['addition','copy'] else task+'_correct'],
                                   choice_mass=end.get(task+'_choice_mass')))
        if job=='base':
            end.update(qualified=True,unique_trainable=0,policy_updates=0)
        else:
            done=read(directory/'DONE.json'); spec=jobs[job]; layout=read(directory/'layout.json')
            assert done['identity']['job']==dict(id=job,**spec) and done['identity']['run']==stamp
            assert done['fingerprint']==digest(done['identity'])==wrapped['model_fingerprint']
            end.update(qualified=done['qualified'],unique_trainable=layout['unique_trainable'],policy_updates=0)
            if layout['frozen_hash']:frozen.add(layout['frozen_hash'])
            if spec['geometry']!='full':
                assert layout['unique_trainable']==2688*spec['width']*(2 if spec['placement']=='different' else 1)
                if spec['geometry']=='native':
                    ix=layout['layout'][0]['indices']; assert set(ix[0]).isdisjoint(ix[1])
                    assert all(v['indices']==ix for v in layout['layout'])
            for phasepath in sorted(directory.glob('*/DONE.json')):
                phase=phasepath.parent.name; result=read(phasepath)
                assert result['identity']['stage']==phase and result['identity']['run']==stamp
                assert result['fingerprint']==digest(result['identity'])
                history=read(phasepath.parent/'history.json'); steps=result['step']
                assert [h['step'] for h in history]==list(range(1,steps+1))
                assert all(math.isfinite(v) for h in history for v in h.values())
                losses.extend(dict(job=job,stage=phase,**h) for h in history)
                phases.append(dict(job=job,stage=phase,step=steps,qualified=result['qualified'],reason=result['reason']))
                paths=sorted(phasepath.parent.glob('dev_*.json'))
                assert [int(p.stem[4:]) for p in paths]==list(range(0,steps+1,32))
                successes=[]; damaged=[]
                for path in paths:
                    step=int(path.stem[4:]); dev=read(path)
                    key='policy_dev' if phase=='policy' else {'tiny':'tiny_one','one':'dev_one','two':'dev_two'}[phase]
                    audit(dev['task'],key); audit(dev['general'],'general_dev')
                    if 'benchmark' in dev:audit(dev['benchmark'],'benchmark_dev')
                    metrics=dev['task']['metrics']; groups=metrics['by_label']
                    curve=dict(job=job,stage=phase,step=step,path_correct=metrics['path_correct'],
                               allowed_correct=groups['allowed']['path_correct'],blocked_refusal=groups.get('blocked',{}).get('refusal',0.),
                               false_refusal=groups['allowed']['refusal'],general=dev['general']['metrics']['path_correct'],
                               **(dev.get('benchmark',{}).get('metrics',{})))
                    curves.append(curve)
                    threshold=cfg['tiny_threshold'] if phase=='tiny' else cfg['cap_threshold']
                    successes.append(metrics['path_correct']>=threshold and all(v['path_correct']>=threshold for v in groups.values()))
                    if phase=='policy':
                        teacher=read(ROOT/'jobs'/spec['teacher']/'acquisition.json')['dev_two']['metrics']['by_label']['allowed']['path_correct']
                        baseline=read(ROOT/'base.json')['general_dev']['metrics']['path_correct']
                        damaged.append(step>=128 and (curve['allowed_correct']<teacher-.2 or curve['general']<baseline-.2))
                assert read(paths[-1])==result['report']
                if phase!='policy':
                    assert result['qualified']==all(successes[-2:])
                    assert (steps==cfg[phase+'_steps'] or result['qualified'])
                else:
                    guard=all(damaged[-2:])
                    assert (result['reason']=='retention_guard_failed')==guard
                    assert result['qualified']==(not guard and curve['blocked_refusal']>=.9 and curve['allowed_correct']>=.8 and curve['false_refusal']<=.1)
                    end['policy_updates']=steps
                if spec['geometry']!='full':
                    saved=load(phasepath.parent/'final.pt'); checkpoint=load(phasepath.parent/'checkpoint.pt')
                    assert saved['identity']==result['identity']==checkpoint['identity']
                    assert saved['fingerprint']==result['fingerprint']==checkpoint['fingerprint']
                    assert saved['step']==checkpoint['step']==steps and checkpoint['history']==history
                    assert saved['parameters'].keys()==checkpoint['parameters'].keys()
                    for name,value in saved['parameters'].items():
                        assert torch.isfinite(value).all() and torch.equal(value,checkpoint['parameters'][name])
                        if spec['geometry']=='native' and spec['placement']=='same' and name.endswith('deltas.1'):assert torch.count_nonzero(value)==0
                    states+=1
            if spec['kind']=='policy':
                initial=read(directory/'policy/dev_0000.json')
                teacher=read(ROOT/'jobs'/spec['teacher']/'acquisition.json')['dev_two']['rows']
                expected={r['prompt']:r['response'] for r in teacher}
                mismatches=sum(r['response']!=expected[r['prompt']] for r in initial['task']['rows'])
                assert mismatches==0
                assert done['identity']['parent']==read(ROOT/'jobs'/spec['teacher']/'DONE.json')['fingerprint']
                clone_checks.append(dict(job=job,disagreements=mismatches))
        endpoints.append(end)
    assert len(frozen)==1
    df=pd.DataFrame(endpoints); cv=pd.DataFrame(curves); ls=pd.DataFrame(losses); ph=pd.DataFrame(phases)
    hp=pd.DataFrame(hoprows); cl=pd.DataFrame(collateral)
    base=df.set_index('job').loc['base']
    for col in ['general','addition','copy','arc_easy_correct','winogrande_correct']:df[col+'_change_pp']=100*(df[col]-base[col])
    obs={j:load(ROOT/'observations'/j/'activations.pt') for j in alljobs}
    direction=[]; oldbehavior=[]; behavior=[]; cosine_errors=[]
    nfit=len(old['fit']); nold=nfit+len(old['heldout'])
    blocked=torch.tensor([r['label']=='blocked' for r in data['policy_observe']])
    basefit=obs['base']['fit_refused']; assert obs['base']['behavior_partition_available']
    for job in alljobs:
        artifact=obs[job]; alignment=read(ROOT/'observations'/job/'alignment.json')
        replay=read(ROOT/'observations'/job/'new_behavior.json')
        assert replay['job']==job and replay['observational_only'] and not replay['direction_penalty_used'] and not replay['activation_intervention']
        assert replay['data_sha256']==manifest['data_sha256']
        assert replay['parameter_hash_before']==replay['parameter_hash_after']==artifact['parameter_hash']==artifact['parameter_hash_after']
        assert replay['checkpoint_fingerprint']==(stamp if job=='base' else read(ROOT/'jobs'/job/'DONE.json')['fingerprint'])
        audit(replay['report'],'policy_observe')
        assert artifact['observational_only'] and alignment['observational_only'] and not alignment['causal_ablation_performed'] and not alignment['direction_penalty_used']
        assert artifact['behavior']==alignment['old_behavior']
        assert len(artifact['behavior'])==nold and len(alignment['layers'])==24
        assert all(r['refusal_language']==scoring['refusal'](r['response']) for r in artifact['behavior'])
        for r,expected in zip(artifact['behavior'],old['fit']+old['heldout']):
            assert all(r[k]==v for k,v in expected.items())
        ownmask=torch.tensor([r['refusal_language'] for r in artifact['behavior'][:nfit]])
        assert torch.equal(ownmask,artifact['fit_refused'])
        ownenough=bool(ownmask.sum()>=8 and (~ownmask).sum()>=8)
        assert artifact['behavior_partition_available']==ownenough
        basis_mask=ownmask if ownenough else torch.tensor([r['label']=='unsafe' for r in old['fit']])
        for split,rows in [('fit',artifact['behavior'][:nfit]),('heldout',artifact['behavior'][nfit:])]:
            assert [r['id'] for r in rows]==[r['id'] for r in old[split]]
            for label in ['safe','unsafe']:
                selected=[r for r in rows if r['label']==label]
                oldbehavior.append(dict(job=job,split=split,label=label,n=len(selected),refusal_language=np.mean([r['refusal_language'] for r in selected])))
        observed=torch.tensor([r['refusal'] for r in replay['report']['rows']])
        enough=bool(observed.sum()>=8 and (~observed).sum()>=8)
        m=replay['report']['metrics']['by_label']
        behavior.append(dict(job=job,allowed_correct=m['allowed']['path_correct'],blocked_refusal=m['blocked']['refusal'],
                             false_refusal=m['allowed']['refusal'],refused=int(observed.sum()),answered=int((~observed).sum()),
                             behavior_partition_available=enough))
        parent=obs[jobs[job]['teacher']] if job!='base' and jobs[job]['kind']=='policy' else obs['base']
        for layer,record in enumerate(alignment['layers']):
            h=artifact['new'][layer]; reference=obs['base']['basis'][layer]
            assert h.shape==(96,896) and artifact['old'][layer].shape==(nold,896)
            assert torch.isfinite(h).all() and torch.isfinite(artifact['old'][layer]).all()
            assert torch.allclose(artifact['basis'][layer],contrast(artifact['old'][layer][:nfit],basis_mask),atol=3e-7,rtol=3e-6)
            new=contrast(h,blocked); induced=new-contrast(parent['new'][layer],blocked)
            fixed=contrast(artifact['old'][layer][:nfit],basefit)
            c=cosine(induced,reference)
            expected=dict(cosine_new_vs_original=cosine(new,reference),cosine_induced_vs_original=c,
                          induced_energy_fraction_along_original=c*c if c is not None else None,
                          cosine_fixed_partition_old_vs_original=cosine(fixed,reference),induced_contrast_norm=float(induced.norm()))
            # ARM/x86 float32 reduction ordering differs by a few rounding units.
            assert all((v is None and record[k] is None) or (v is not None and record[k] is not None
                and math.isclose(v,record[k],abs_tol=3e-7,rel_tol=3e-6)) for k,v in expected.items()), (job,layer,expected,record)
            cosine_errors.extend(abs(v-record[k]) for k,v in expected.items() if k.startswith('cosine') and v is not None)
            fixedparent=contrast(parent['old'][layer][:nfit],basefit)
            actual=contrast(h,observed) if enough else None
            heldout_y=[r['refusal_language'] for r in artifact['behavior'][nfit:]]
            direction.append(dict(job=job,block=layer+1,**record,
                 cosine_induced_vs_capability_old=cosine(induced,fixedparent),
                 cosine_actual_refusal_vs_original=cosine(actual,reference) if actual is not None else None,
                 original_candidate_heldout_auc=auc(heldout_y,(artifact['old'][layer][nfit:]@reference).numpy())))
            if job in ['native4','native128','policy_same','policy_different'] and layer<11:
                assert torch.equal(h,obs['base']['new'][layer]) and torch.equal(artifact['old'][layer],obs['base']['old'][layer])
    dr=pd.DataFrame(direction); ob=pd.DataFrame(oldbehavior); bh=pd.DataFrame(behavior)
    for name,table in [('endpoints',df),('development',cv),('losses',ls),('phases',ph),('one_hop_diagnostics',hp),
                       ('collateral',cl),('directions',dr),('old_refusal_behavior',ob),('observation_behavior',bh)]:
        table.to_csv(OUT/(name+'.csv'),index=False)
    summary=dict(completed_utc=datetime.fromtimestamp(status['time'],timezone.utc).isoformat(),
        final_models=9,final_test_records=sum(sum(len(v['rows']) for v in r.values()) for r in reports.values()),
        evaluation_reports_recounted=evaluations,records_recounted=records,sparse_phase_checkpoint_pairs_verified=states,
        source_hashes_verified=len(manifest['sources']),world_banks={k:len(v) for k,v in banks.items()},
        old_worlds_excluded=len(excluded),clone_checks=clone_checks,direction_rows_recomputed=len(dr),
        unchanged_parameter_hashes_verified=9,old_behavior_records_checked=9*nold,
        max_cosine_recompute_error=max(cosine_errors),observational_only=True,checks_passed=True)
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2))
    plot(df,cv,ls,dr,ob,bh,hp)
    write_report(df,cv,ph,dr,bh,hp,summary,reports)
    print(json.dumps(summary,indent=2))
    print(df[['job','allowed_correct','blocked_refusal','general','qualified']].to_string(index=False))


def plot(df,cv,ls,dr,ob,bh,hp):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    policy=['policy_same','policy_different','policy_full']
    indexed=df.set_index('job')
    with PdfPages(OUT/'results.pdf') as pdf:
        page=0
        def save(fig,title,note):
            nonlocal page
            page+=1
            fig.suptitle(title,fontsize=16,fontweight='bold',y=.98)
            fig.text(.02,.018,note,fontsize=9,va='bottom')
            fig.tight_layout(rect=(0,.07,1,.94)); pdf.savefig(fig)
            fig.savefig(OUT/f'page_{page:02d}.png',dpi=110); plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(13,5))
        for ax,col,title in zip(axs,['blocked_refusal','allowed_correct','general'],['Blocked cases refused','Allowed paths correct','Addition + copying correct']):
            for i,j in enumerate(['base',*policy]):
                p=indexed.loc[j,col]; n=64 if col=='general' else 256; lo,hi=wilson(p,n)
                ax.bar(i,100*p,color=COLORS[j],width=.65)
                ax.errorbar(i,100*p,yerr=[[100*(p-lo)],[100*(hi-p)]],color='black',capsize=3)
                ax.text(i,100*hi+2,f'{100*p:.1f}',ha='center',fontsize=9)
            ax.set_xticks(range(4),['Base','Shared 4','Disjoint 4+4','Full'],rotation=20)
            ax.set_ylim(0,112); ax.set_title(title); ax.set_ylabel('%')
        save(fig,'Final held-out behavior: shared four-channel training succeeds',
             'Seed 31; 256 worlds per policy category. Error bars: Wilson 95% sampling intervals, not training-seed uncertainty.\nDisjoint stopped at 160 updates; full stopped at 224; shared completed 512. Endpoints are not compute matched.')
        fig,axs=plt.subplots(2,2,figsize=(13,8))
        for ax,col,title in zip(axs.flat,['allowed_correct','blocked_refusal','false_refusal','general'],['Allowed path accuracy','Blocked refusal rate','Allowed false-refusal rate','Collateral development accuracy']):
            for job in policy:
                s=cv[(cv.job==job)&(cv.stage=='policy')]
                ax.plot(s.step,100*s[col],marker='o',ms=3,label=NAMES[job],color=COLORS[job])
            ax.set_title(title); ax.set_ylim(-3,103); ax.set_xlabel('Policy optimizer updates'); ax.set_ylabel('%'); ax.grid(alpha=.15)
            if col=='general':
                baseline=read(ROOT/'base.json')['general_dev']['metrics']['path_correct']
                original=ax.axhline(100*baseline,color='#52525b',ls='--',lw=1,label='Frozen base')
                threshold=ax.axhline(100*(baseline-.2),color='#52525b',ls=':',lw=1,label='Collateral guard threshold')
                ax.legend(handles=[original,threshold],loc='lower right',fontsize=8)
        axs[0,0].legend(fontsize=9)
        save(fig,'Policy learning curves reveal instability and early stopping',
             'Development sets: 96 paired worlds; 64 collateral prompts. Native conditions share the same capability teacher.\nFull-model reference performs well early; its final collateral collapse does not rule out an earlier useful checkpoint.')
        fig,axs=plt.subplots(1,3,figsize=(13,5))
        for job in ['native4','native128','full_on','full_off','scattered_cap']:
            for ax,stage in zip(axs,['tiny','one','two']):
                s=cv[(cv.job==job)&(cv.stage==stage)]
                if len(s):ax.plot(s.step,100*s.path_correct,marker='o',ms=3,label=NAMES[job],color=COLORS[job])
                ax.axhline(98 if stage=='tiny' else 90,color='gray',ls='--',lw=.8)
                ax.set_title({'tiny':'16-world one-hop fit','one':'Fresh one-hop worlds','two':'Fresh two-hop worlds'}[stage])
                ax.set_xlabel('Updates within stage'); ax.set_ylabel('Path accuracy (%)'); ax.set_ylim(-3,103)
        axs[0].legend(fontsize=8)
        save(fig,'Gated capability acquisition: four native channels were sufficient',
             'Each gate requires its threshold within every marker group twice consecutively. Optimizer resets per stage.\nScattered reached 98.3% at update 256, but lacked two consecutive passing checks; no scattered policy fit was trained.')
        fig,axs=plt.subplots(1,2,figsize=(13,6))
        selected=['base','native4','native128','full_on','full_off',*policy]
        columns=['addition','copy','arc_easy_correct','winogrande_correct']
        values=indexed.loc[selected,columns].to_numpy()*100
        im=axs[0].imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        axs[0].set_xticks(range(4),['Addition','Copy','ARC-Easy','WinoGrande'],rotation=20)
        axs[0].set_yticks(range(len(selected)),[NAMES[j] for j in selected])
        for (y,x),v in np.ndenumerate(values):axs[0].text(x,y,f'{v:.1f}',ha='center',va='center',color='white' if v<65 else 'black',fontsize=9)
        fig.colorbar(im,ax=axs[0],label='Accuracy (%)',fraction=.035)
        for task,marker in [('arc_easy','o'),('winogrande','s')]:
            axs[1].plot(range(len(selected)),indexed.loc[selected,task+'_choice_mass'],marker=marker,label=task)
        axs[1].set_yscale('log'); axs[1].set_xticks(range(len(selected)),['Base','Cap 4','Cap 128','Cap full+','Cap full−','Shared','Disjoint','Full'],rotation=40)
        axs[1].set_ylabel('Mean probability mass on valid option letters'); axs[1].legend(); axs[1].grid(alpha=.15)
        save(fig,'Collateral accuracy must be read alongside valid-option probability',
             '32 addition + 32 copy prompts; 64 examples per benchmark. Forced choice selects the best option even when all options\nhave negligible probability. Disjoint WinoGrande accuracy is not evidence that its normal generation behavior improved.')
        fig,axs=plt.subplots(1,2,figsize=(13,6))
        selected=['native4','native128','full_on',*policy]
        for i,job in enumerate(selected):
            rows=hp[hp.job==job]
            axs[0].bar(i-.18,100*indexed.loc[job,'one_path'],.35,color='#2563eb')
            axs[0].bar(i+.18,100*(rows.nodes==2).mean(),.35,color='#d97706')
        axs[0].set_xticks(range(len(selected)),['Cap 4','Cap 128','Cap full+','Shared','Disjoint','Full'],rotation=25)
        axs[0].set_ylabel('% of one-hop requests')
        axs[0].legend(['Exact one-hop answer','Returns two nodes'],loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=9)
        rows=hp[hp.job=='policy_same']
        vals=[rows[rows.marker_is_endpoint].refusal.mean(),rows[~rows.marker_is_endpoint].refusal.mean()]
        axs[1].bar([0,1],np.array(vals)*100,color=['#d97706','#2563eb'])
        axs[1].set_xticks([0,1],['Marked requested endpoint','Other / absent marker'])
        axs[1].set_ylabel('False refusal on one-hop requests (%)'); axs[1].set_ylim(0,103)
        for i,v in enumerate(vals):axs[1].text(i,100*v+1,f'{100*v:.1f}%',ha='center')
        save(fig,'Important failure: two-hop training overwrote hop-count flexibility',
             'All one-hop cases are allowed: they contain no intermediate node. After the two-hop-only stage, models usually\nreturn two nodes anyway. Shared policy also over-refuses marked one-hop endpoints. This is a narrow two-hop result.')
        fig,axs=plt.subplots(1,2,figsize=(13,6))
        selected=['native4','native128','full_on','full_off',*policy]
        matrix=dr[dr.job.isin(selected)].pivot(index='job',columns='block',values='cosine_induced_vs_original').loc[selected]
        cmap=plt.get_cmap('coolwarm').copy(); cmap.set_bad('#f4f4f4')
        im=axs[0].imshow(np.ma.masked_invalid(matrix.values),cmap=cmap,vmin=-1,vmax=1,aspect='auto')
        axs[0].set_yticks(range(len(selected)),[NAMES[j] for j in selected]); axs[0].set_xticks([0,5,11,17,23],[1,6,12,18,24])
        axs[0].set_xlabel('Transformer block output'); fig.colorbar(im,ax=axs[0],fraction=.035,label='Cosine to frozen-base old candidate')
        for job in policy:
            d=dr[dr.job==job]; axs[1].plot(d.block,100*d.induced_energy_fraction_along_original,label=NAMES[job],color=COLORS[job])
        axs[1].set_xlabel('Transformer block output'); axs[1].set_ylabel('Squared-cosine projection fraction (%)'); axs[1].legend(fontsize=8)
        save(fig,'Observed new-task contrasts have little alignment with the old candidate',
             'Policy contrasts subtract their capability-teacher contrasts; acquisition contrasts subtract the base. Light gray = zero\ninduced contrast. Disjoint did not learn valid policy behavior. Geometry alone does not identify independent causal circuits.')
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for job in ['native4',*policy]:
            s=dr[dr.job==job]; axs[0].plot(s.block,s.cosine_fixed_partition_old_vs_original,label=NAMES[job],color=COLORS[job])
        axs[0].set_xlabel('Transformer block output'); axs[0].set_ylabel('Cosine: original vs current old-prompt contrast'); axs[0].set_ylim(-.05,1.05); axs[0].legend(fontsize=8)
        for label,offset in [('safe',-.17),('unsafe',.17)]:
            s=ob[(ob.split=='heldout')&(ob.label==label)].set_index('job')
            axs[1].bar(np.arange(4)+offset,100*s.loc[['base',*policy],'refusal_language'],.34,label=label)
        axs[1].set_xticks(range(4),['Base','Shared 4','Disjoint 4+4','Full']); axs[1].set_ylabel('Refusal-language rate on old prompts (%)'); axs[1].legend()
        save(fig,'The shared fit largely preserves the original observational contrast',
             'Left uses the same frozen-base refused/answered fit partition throughout. Right uses 24 safe + 24 unsafe held-out\nXSTest prompts and a refusal-language heuristic. These rates are diagnostics, not a comprehensive natural-safety evaluation.')
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        for job in policy:
            s=ls[(ls.job==job)&(ls.stage=='policy')]
            for ax,col in zip(axs,['capability_nll','conserve_full_kl']):
                ax.plot(s.step,s[col].rolling(8,min_periods=1).mean(),label=NAMES[job],color=COLORS[job])
                ax.set_xlabel('Policy updates'); ax.set_ylabel(col.replace('_',' ')); ax.set_yscale('symlog',linthresh=.001)
        axs[0].legend(fontsize=8)
        save(fig,'Loss traces distinguish a successful policy fit from unstable optimization',
             'Eight-update moving means; raw losses, gradient norms, learning rates and update times are available in losses.csv.\nNo direction penalty, direction projection or activation intervention was used in training or observation.')


def write_report(df,cv,ph,dr,bh,hp,audit,reports):
    ix=df.set_index('job'); policy=['policy_same','policy_different','policy_full']
    def pct(v):return f'{100*v:.1f}%'
    lines=['# Shared-location capability and refusal pilot — 4 November 2026','',
      '**Result:** this one-seed pilot successfully trained a two-hop pointer capability and selective refusal through the same four native MLP channels. It did not establish a separate causal refusal mechanism, general hop-count reasoning, or resistance to refusal removal.','',
      f"Completed {audit['completed_utc']} on `mn866@weatherwax.cs.bath.ac.uk`. All nine planned training/observation jobs and all nine final test evaluations completed without worker failures. The scattered acquisition fit and two policy fits were recorded as scientific qualification failures, not missing/crashed jobs.",'',
      '## Main held-out results','',
      '| Endpoint | Allowed paths correct | Blocked refused | False refusal | Addition + copy | ARC-Easy | WinoGrande |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for job in ['base',*policy]:
        r=ix.loc[job]; lines.append('| '+NAMES[job]+' | '+' | '.join(pct(r[c]) for c in ['allowed_correct','blocked_refusal','false_refusal','general','arc_easy_correct','winogrande_correct'])+' |')
    lines+=['',
      'Each policy category contains 256 held-out worlds, paired by graph and start node. The collateral panel has 32 addition and 32 copying prompts; each benchmark has 64 questions. Benchmark scores are forced-choice scores, and must be read with option probability mass. These are small panels and one training seed, not a broad guarantee of retained capabilities.','',
      'The shared fit answered 251/256 allowed cases and refused 255/256 blocked cases; three allowed cases were falsely refused. Its addition/copy score changed from 59/64 for the base to 58/64: addition stayed at 30/32, while copying changed from 29/32 to 28/32. ARC-Easy changed from 44/64 to 45/64 and WinoGrande from 26/64 to 24/64.','',
      '## What was trained','',
      'Pinned Qwen2.5-0.5B-Instruct, FP32, seed 31. Native updates affect block 12 (zero-based 11): matching rows of the MLP up/gate matrices and matching columns of the down matrix. Four channels mean **10,752 trainable scalars**. The rest of the pretrained model, including attention, stays frozen. A shared fit routes both capability and policy gradients to these same weights; the disjoint fit uses four capability channels plus four separate policy channels (**21,504 unique scalars**). Neither arrangement confines all computation to those weights; the frozen network can still perform and propagate computation.','',
      'This is supervised training, not RL. The curriculum first fits a tiny one-hop set, then fresh one-hop worlds, then two-hop worlds, with two consecutive passing gates. Acquisition ignores the marked-node refusal rule. Policy fitting then learns to refuse when the marked node is the intermediate node of a two-hop path. It combines allowed-path likelihood, positive/negative refusal-decision losses, refusal-output grammar, allowed-response preservation, and unrelated-response conservation. The model is not given the symbolic refusal rule as a prompt instruction.','',
      '## Capability acquisition','',
      '| Fit | Final two-hop path accuracy | Addition + copy | Acquisition qualified? |',
      '|---|---:|---:|---:|']
    for job in ['base','native4','native128','full_on','full_off','scattered_cap']:
        r=ix.loc[job]; lines.append(f"| {NAMES[job]} | {pct(r.two_path)} | {pct(r.general)} | {'reference' if job=='base' else bool(r.qualified)} |")
    lines+=['',
      'Native width 4 improved final exact two-hop path accuracy from 2.1% to 97.3%. Width 128 also acquired it (97.0%). This supports feasibility at width 4 under this recipe, not optimality of width 4. The base score is exact requested-trace accuracy, and does not establish that the base lacked all pointer reasoning. This corrected pilot is not the complete originally proposed width sweep.','',
      'The 10,752-weight scattered mask did not achieve two consecutive passing tiny-set checks within 256 updates. It reached 97.7% at update 224 and 98.3% at update 256; that last evaluation passed every group threshold, but was its first consecutive pass. No scattered policy fit was therefore launched. This is a budget-limited gate failure, not evidence that scattered weights cannot learn, and not a matched trained-policy comparator.','',
      'Without conservation, the full-model acquisition fit answered two-hop paths well but its addition/copy accuracy fell to 0%, and valid-option probability on ARC-Easy/WinoGrande became tiny. Conservation reduced but did not eliminate collateral changes in the full-model acquisition fit. Native width 4 also damaged collateral accuracy during acquisition (64.1%); later conservation during its shared policy fit brought the final collateral score close to base.','',
      '## Endpoint fairness and optimization failures','',
      'The shared and disjoint policy fits started from the **identical native4 capability checkpoint** and agreed on every initial development response. Shared completed 512 updates. Disjoint stopped at 160 after two failed retention checks; all 512 final policy responses began with `REF` but none formed a valid answer or the required complete `REFUSE` output (examples: `REFJO`, `REFSTEea`). Its zero strict refusal rate therefore reflects malformed, nonselective refusal-like output, not successful compliance. Full-model policy stopped at 224 after collateral retention failed, despite excellent task selectivity. These final endpoints are not compute matched, and disjoint uses twice the native trainable support.','',
      '| At policy update 160 (development only) | Allowed paths | Blocked refused | Addition + copy |',
      '|---|---:|---:|---:|']
    for job in policy:
        r=cv[(cv.job==job)&(cv.stage=='policy')&(cv.step==160)].iloc[0]
        lines.append('| '+NAMES[job]+' | '+' | '.join(pct(r[k]) for k in ['allowed_correct','blocked_refusal','general'])+' |')
    early=cv[(cv.job=='policy_full')&(cv.stage=='policy')&(cv.step==32)].iloc[0]
    lines+=['',
      f"The full-model reference already reached {pct(early.allowed_correct)} allowed accuracy / {pct(early.blocked_refusal)} blocked refusal / {pct(early.general)} collateral accuracy at update 32 on development. Therefore its final degradation does **not** show that full-model training cannot retain both. No intermediate full-model weights from update 32 were retained for a new held-out evaluation; the original runner overwrote its rolling checkpoint. A subsequent study should predeclare and save a best feasible development checkpoint.",'',
      'Disjoint instability could reflect fresh policy-channel initialization, gradient scaling, learning rate, or this channel draw. One seed and one optimization recipe cannot establish that spatial separation inherently fails. Its 60.9% forced-choice WinoGrande score is misleading without noting mean valid-option probability of about 0.000292; its normal pointer generation was broken.','',
      '## Hop-count and marker limitations','',
      'Strong two-hop performance does not extend to flexible hop counts. The last acquisition stage and policy stage train only two-hop responses; one-hop replay is absent. Models frequently return two nodes even when the prompt requests one. This is a real instruction/generalization failure, not proof that the underlying first-step lookup information was erased.','']
    for job in ['native4','policy_same']:
        rows=hp[hp.job==job]
        lines.append(f"- {NAMES[job]}: exact one-hop accuracy {pct(ix.loc[job,'one_path'])}; two-node responses on one-hop requests {pct((rows.nodes==2).mean())}; correct first-node prefix {pct(rows.first_node_correct.mean())}.")
    same=hp[hp.job=='policy_same']
    lines+=['',
      f"All one-hop cases should be allowed because there is no intermediate node. Shared policy falsely refuses {pct(same[same.marker_is_endpoint].refusal.mean())} when the marker is the requested endpoint, compared with {pct(same[~same.marker_is_endpoint].refusal.mean())} for other/absent markers. This is consistent with a fixed-two-hop shortcut, not a demonstrated general rule across hop counts.",
      f"The shared model also scores {pct(ix.loc['policy_same','unmarked_correct'])} on unmarked two-hop prompts and falsely refuses {pct(ix.loc['policy_same','unmarked_refusal'])} of them. Its marked allowed-policy score alone would hide this weakness.",'',
      '## Direction comparisons: measured, never optimized','',
      'All 24 block outputs were recorded at the final prompt position before final RMSNorm. The original candidate is a mean difference between baseline refusal-language and non-refusal outputs on 96 independent XSTest calibration prompts. Another 48 focus-disjoint prompts are held out. This is a refusal-associated candidate, not a causally validated universal refusal direction. If a later model has too few examples of either behavior, its own candidate is explicitly flagged as a safe/unsafe content fallback.','',
      'For each policy fit, the new-task blocked-minus-allowed activation contrast is compared with its capability teacher; subtracting the teacher contrast gives the policy-induced contrast. This helps control the pre-existing marker-content contrast. Comparisons with the original candidate use the same reference throughout. Fixed-partition old-prompt comparisons keep the baseline refused/answered partition unchanged.','',
      '| Shared native fit | Cosine: induced new contrast vs old candidate | Squared-cosine fraction | Old fixed-partition stability |',
      '|---|---:|---:|---:|']
    for block in [12,24]:
        r=dr[(dr.job=='policy_same')&(dr.block==block)].iloc[0]
        lines.append(f'| Block {block} | {r.cosine_induced_vs_original:.5f} | {100*r.induced_energy_fraction_along_original:.5f}% | {r.cosine_fixed_partition_old_vs_original:.5f} |')
    rb=bh.set_index('job').loc['policy_same']
    r=dr[(dr.job=='policy_same')&(dr.block==24)].iloc[0]
    active=dr[(dr.job=='policy_same')&(dr.block>=12)]
    baseline_auc=dr[(dr.job=='base')&(dr.block==24)].iloc[0].original_candidate_heldout_auc
    lines+=['',
      f"A read-only replay on the exact 48 matched observation worlds confirmed actual policy behavior: shared allowed accuracy {pct(rb.allowed_correct)}, blocked refusal {pct(rb.blocked_refusal)}, false refusal {pct(rb.false_refusal)}. On those inputs, the raw contrast between actual refused and non-refused cases has final-block cosine {r.cosine_actual_refusal_vs_original:.5f} with the original candidate. Raw activation files and all per-block cosines are retained.",'',
      f"Across every block from 12 through 24, the shared policy-induced contrast has absolute cosine at most {active.cosine_induced_vs_original.abs().max():.5f} with the old candidate (maximum squared-cosine fraction {100*active.induced_energy_fraction_along_original.max():.3f}%). The baseline final-block candidate predicts the baseline's refusal-language labels on held-out old prompts with AUC {baseline_auc:.3f}; this supports its observational relevance but does not validate it causally.",'',
      'The data are consistent with a new policy-associated contrast that has little projection along the old candidate, while the original old-prompt contrast largely persists in the shared fit. **This does not prove the new refusal avoids the original causal circuitry or occupies an independent circuit.** Mean-direction geometry is sensitive to prompt content, output format, layer, token position and nonlinear downstream computation. The disjoint model did not learn the behavior, so its low alignment is not evidence of a successful new refusal representation. The squared cosine is a projection fraction of one contrast vector, not a fraction of all representation variance or all refusal computation.','',
      'No direction penalty, projection, ablation, activation steering, direction-based gating, or direction-based checkpoint selection was performed. Both the original observations and additional saved-model generation replays verified identical parameter hashes before and after measurement.','',
      '## Engineering audit and storage','',
      f"Recounted {audit['records_recounted']:,} prediction/benchmark records across {audit['evaluation_reports_recounted']} evaluation reports; checked {audit['source_hashes_verified']} frozen source hashes, all dataset/world splits and 928 excluded previous worlds, {audit['sparse_phase_checkpoint_pairs_verified']} sparse final/resume checkpoint pairs, all gate/retention decisions, the three capability-parent clone checks, 216 per-layer direction records, and all nine observational parameter-hash pairs. All checks passed. Training histories contain no nonfinite losses or gradient norms. Full-model weight tensors were not downloaded or exhaustively compared locally; they remain on Weatherwax.",'',
      'Local results contain all compact JSON/log measurements, source snapshots, raw observation activations and sparse checkpoint tensors. Large full-model weights, optimizer states and teacher caches remain on the server under `/mnt/fast1/mn866/casr-colocalization-observe-20261104/results` (about 60 GB). Compact archive SHA-256: `a2890a4fcbe718b621f78edb4f74ee0b7ba3f863cf5b1231415a93490e274a9e`. No previous run was overwritten.','',
      '## Next experiment warranted by these results','',
      'First keep one-/two-hop replay and unmarked allowed examples in policy training, and gate on both depths. Extend the near-passing scattered tiny stage so it can reach the two-check acquisition gate. Repeat shared/disjoint/full across seeds with a lower policy-channel learning-rate sweep for disjoint. Save a best feasible development checkpoint as well as the final checkpoint, using predeclared behavior/retention criteria and no direction criterion. Only then expand the policy-width sweep. Continue recording directions observationally. No new training was launched during this analysis.','',
      '## Files','',
      '- [Eight-page graphs](results.pdf)','- [Final scores](endpoints.csv)','- [Learning curves](development.csv)',
      '- [Raw losses](losses.csv)','- [Gate outcomes](phases.csv)','- [All-layer direction comparisons](directions.csv)',
      '- [Old-prompt refusal diagnostics](old_refusal_behavior.csv)','- [Observation-set behavior](observation_behavior.csv)',
      '- [One-hop failure diagnostics](one_hop_diagnostics.csv)','- [Machine-readable audit](audit.json)','',
      'Reproduce locally with `/usr/bin/python3 experiments/refusal_colocalization_v2/analyse.py`.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
