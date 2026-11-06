"""Recount archived predictions and plot the completed location/budget pilot."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch


HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'results/weatherwax-20261103'
ROOT=ARCHIVE/'results'
OUT=HERE/'analysis'
LABELS={'same':'Shared channels','different':'Disjoint channels','scattered':'Scattered weights','full':'Full model','base':'Frozen base'}
COLORS={'same':'#2563EB','different':'#D97706','scattered':'#0F766E','full':'#7C3AED','base':'#52525B'}


def main():
    if not ROOT.exists():raise SystemExit('Raw measurements were removed during the 5 November cleanup. See refusal/RAW_DATA.md to restore them; analysed tables and plots remain in this study.')
    OUT.mkdir(exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    manifest=read(ROOT/'manifest.json')
    cfg=manifest['protocol']
    stamp=digest(manifest)
    data=read(ROOT/'data.json')
    assert digest(data)==manifest['data_sha256']
    for name,sha in manifest['source'].items():
        p=ARCHIVE/name if name=='run.py' else ARCHIVE/'reference'/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
    status=read(ROOT/'status.json')
    assert status['complete']==status['total']==86 and not status['failed'] and not status['running']
    worlds=[]
    for panel in ('train','dev','test','wording'):
        worlds.extend({r['world'] for r in data[panel]})
        for row in data[panel]:
            path=[row['path'][0]]
            for _ in range(row['depth']):
                path.append(row['edges'][path[-1]])
            assert path==row['path'] and path[-1]==row['answer']
            assert (row['marker'] in path[1:-1])==(row['label']=='blocked')
            assert row['marker'] not in (path[0],path[-1])
    assert len(worlds)==len(set(worlds))
    tree=ast.parse((ARCHIVE/'run.py').read_text())
    functions=ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef)
                              and x.name in ('strict_score','metrics')],type_ignores=[])
    scoring={'re':re}
    exec(compile(functions,'archived_scoring','exec'),scoring)
    responses,evaluations=0,0

    def audit(report,panel):
        nonlocal responses,evaluations
        expected={r['id']:r for r in data[panel]}
        rows=report['rows']
        assert len(rows)==len(expected) and {r['id'] for r in rows}==set(expected)
        for row in rows:
            assert all(row[k]==v for k,v in expected[row['id']].items())
            assert all(row[k]==v for k,v in scoring['strict_score'](row,row['response']).items())
            assert isinstance(row['truncated'],bool)
        actual=scoring['metrics'](rows)
        assert actual.keys()==report['metrics'].keys()
        assert all(math.isclose(v,report['metrics'][k],abs_tol=1e-10) for k,v in actual.items())
        expected_groups={str(r.get('depth',r['task'])) for r in rows}
        assert expected_groups==report['by_depth_or_task'].keys()
        for group in expected_groups:
            actual=scoring['metrics']([r for r in rows if str(r.get('depth',r['task']))==group])
            assert actual==report['by_depth_or_task'][group]
        responses+=len(rows)
        evaluations+=1

    baseline=read(ROOT/'baseline_test.json')
    for panel,report in baseline.items():
        audit(report,panel)
    for panel,report in read(ROOT/'baseline_dev.json').items():
        audit(report,panel)
    base=baseline['test']['metrics']
    baseline_general=baseline['general_test']['metrics']['allowed_correct']
    endpoints,curves,losses,panels,ablations=[],[],[],[],[]
    reports={}
    marker_diagnostics=[]
    clone_disagreements=[]
    frozen_hashes=set()
    archived_states=0

    def marker_diagnostic(stage,job,rows):
        for label in ['allowed','blocked']:
            for depth in [None,2,3,4]:
                selected=[r for r in rows if r['label']==label and (depth is None or r['depth']==depth)]
                marker_diagnostics.append(dict(stage=stage,job=job,label=label,depth=depth or 'all',n=len(selected),
                    correct=np.mean([r['correct'] for r in selected]),
                    next_marker_agreement=np.mean([r['response']==r['edges'][r['marker']] for r in selected]),
                    next_start_agreement=np.mean([r['response']==r['edges'][r['path'][0]] for r in selected]),
                    oracle_next_marker_accuracy=np.mean([r['edges'][r['marker']]==r['answer'] for r in selected])))

    marker_diagnostic('base','base',baseline['test']['rows'])

    def row_summary(stage,job,report,layout):
        geometry,width,placement=job.split('__')
        condition='full' if geometry=='full' else 'scattered' if geometry=='scattered' else placement
        result=dict(stage=stage,job=job,geometry=geometry,condition=condition,width=int(width[1:]),
                    unique_trainable=layout['unique_trainable'],**report['test']['metrics'])
        result['selectivity']=result['blocked_refusal']-result['allowed_refusal']
        result['refusal_balanced_accuracy']=.5*(1+result['selectivity'])
        result['general_accuracy']=report['general_test']['metrics']['allowed_correct']
        result['general_change_pp']=100*(result['general_accuracy']-baseline_general)
        result['wording_joint_score']=report['wording']['metrics']['joint_score']
        result['wording_allowed_correct']=report['wording']['metrics']['allowed_correct']
        result['wording_blocked_refusal']=report['wording']['metrics']['blocked_refusal']
        result['wording_allowed_refusal']=report['wording']['metrics']['allowed_refusal']
        result['wording_selectivity']=result['wording_blocked_refusal']-result['wording_allowed_refusal']
        result['qualified']=result['blocked_refusal']>=.9 and result['allowed_correct']>=.8
        result['pointer_accuracy_both_categories']=.5*(result['allowed_correct']+result['blocked_correct'])
        for task,values in report['general_test']['by_depth_or_task'].items():
            result[task+'_accuracy']=values['allowed_correct']
            result[task+'_refusal']=values['allowed_refusal']
        return result

    for stage,count in [('capability',17),('joint',25)]:
        directories=sorted((ROOT/stage).iterdir())
        assert len(directories)==count
        for directory in directories:
            job=directory.name
            done=read(directory/'DONE.json')
            assert done['fingerprint']==stamp and done['step']==512
            layout=read(directory/'layout.json')
            if layout['frozen_hash']:
                frozen_hashes.add(layout['frozen_hash'])
            report=read(directory/'test.json')
            reports[(stage,job)]=report
            if stage=='capability':
                marker_diagnostic(stage,job,report['test']['rows'])
            for panel,value in report.items():
                audit(value,panel)
                panels.append(dict(stage=stage,job=job,panel=panel,**value['metrics']))
            final=row_summary(stage,job,report,layout)
            history=read(directory/'history.json')
            assert [x['step'] for x in history]==list(range(1,513))
            for x in history:
                assert all(np.isfinite(x[k]) for k in ['allowed_nll','blocked_nll','conserve_kl','grad_norm','train_seconds'])
                losses.append(dict(stage=stage,job=job,condition=final['condition'],width=final['width'],**x))
            final['train_seconds']=sum(x['train_seconds'] for x in history)
            final['allocated_gpu_train_seconds']=final['train_seconds']*(2 if final['geometry']=='full' else 1)
            paths=sorted(directory.glob('dev_*.json'))
            assert [int(p.stem.split('_')[1]) for p in paths]==list(range(0,513,64))
            for path in paths:
                value=read(path)
                for panel,r in value.items():
                    audit(r,panel)
                m=value['dev']['metrics']
                curves.append(dict(stage=stage,job=job,condition=final['condition'],width=final['width'],
                    step=int(path.stem.split('_')[1]),**m,
                    pointer_accuracy_both_categories=.5*(m['allowed_correct']+m['blocked_correct']),
                    general_accuracy=value['general_dev']['metrics']['allowed_correct']))
            if final['geometry']!='full':
                width=final['width']
                expected=2688*width*(2 if final['condition']=='different' else 1)
                assert layout['unique_trainable']==expected
                saved=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
                checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
                assert saved['fingerprint']==checkpoint['fingerprint']==stamp
                assert saved['step']==checkpoint['step']==512
                assert saved['parameters'].keys()==checkpoint['parameters'].keys()
                assert all(torch.equal(v,checkpoint['parameters'][n]) for n,v in saved['parameters'].items())
                assert all(torch.isfinite(v).all() for v in saved['parameters'].values())
                if final['geometry']=='native':
                    assert len(saved['parameters'])==6
                    if stage=='capability' or final['condition']=='same':
                        assert all(torch.count_nonzero(v)==0 for n,v in saved['parameters'].items() if n.endswith('deltas.1'))
                    assert sum(v.numel() for n,v in saved['parameters'].items() if n.endswith('deltas.0'))==2688*width
                    ix=layout['layout'][0]['indices']
                    assert set(ix[0]).isdisjoint(ix[1])
                    assert all(x['indices']==ix for x in layout['layout'])
                else:
                    assert sum(v.numel() for v in saved['parameters'].values())==2688*width
                    assert sum(len(x['flat_indices']) for x in layout['layout'])==2688*width
                archived_states+=1
            endpoints.append(final)
            if stage=='joint':
                teacher_job=job.replace('__different','__same')
                original=read(ROOT/'capability'/teacher_job/'dev_0512.json')
                current=read(directory/'dev_0000.json')
                disagreements=sum(a['response']!=b['response'] for panel in original
                    for a,b in zip(original[panel]['rows'],current[panel]['rows']))
                clone_disagreements.append(dict(job=job,disagreements=disagreements))
                cap=next(e for e in endpoints if e['stage']=='capability' and e['job']==teacher_job)
                final['allowed_change_from_capability_pp']=100*(final['allowed_correct']-cap['allowed_correct'])
                final['general_change_from_capability_pp']=100*(final['general_accuracy']-cap['general_accuracy'])
            if (directory/'interventions.json').exists():
                for name,value in read(directory/'interventions.json').items():
                    audit(value,'test')
                    ablations.append(dict(job=job,intervention=name,**value['metrics']))
                    if name=='all_updates_rolled_back':
                        assert [r['response'] for r in value['rows']]==[r['response'] for r in baseline['test']['rows']]
    assert len(frozen_hashes)==1 and archived_states==40
    assert not any(x['disagreements'] for x in clone_disagreements)
    assert len(list(ROOT.glob('*/*/test.json')))==43

    df=pd.DataFrame(endpoints)
    cv=pd.DataFrame(curves)
    ls=pd.DataFrame(losses)
    ab=pd.DataFrame(ablations)
    marker=pd.DataFrame(marker_diagnostics)
    marker.to_csv(OUT/'marker_shortcut_diagnostic.csv',index=False)
    joint=df[df.stage=='joint'].copy()
    cap=df[df.stage=='capability'].copy()
    # Resample matched worlds; these intervals do not estimate seed variation.
    worlds_test=sorted({r['world'] for r in data['test']})
    resamples=np.random.default_rng(20261104).integers(0,len(worlds_test),size=(5000,len(worlds_test)))
    vectors={}
    comparisons=[]
    for row in joint.itertuples():
        records=reports[('joint',row.job)]['test']['rows']
        by={(r['world'],r['label']):r for r in records}
        scores=np.array([.5*(by[(w,'allowed')]['correct']+by[(w,'blocked')]['refusal']) for w in worlds_test])
        discrimination=np.array([by[(w,'blocked')]['refusal']-by[(w,'allowed')]['refusal'] for w in worlds_test],dtype=float)
        vectors[row.job]=scores
        low,high=np.quantile(scores[resamples].mean(axis=1),[.025,.975])
        dlow,dhigh=np.quantile(discrimination[resamples].mean(axis=1),[.025,.975])
        joint.loc[joint.job==row.job,['joint_ci_low','joint_ci_high','selectivity_ci_low','selectivity_ci_high']]=[low,high,dlow,dhigh]
    for width in cfg['widths']:
        a=f'native__k{width}__same'
        b=f'native__k{width}__different'
        for kind,target in [('same_per_objective',a),('same_total',f'native__k{2*width}__same')]:
            if target not in vectors:
                continue
            difference=vectors[b]-vectors[target]
            lo,hi=np.quantile(difference[resamples].mean(axis=1),[.025,.975])
            comparisons.append(dict(comparison=kind,disjoint_width=width,shared_width=int(target.split('__')[1][1:]),
                joint_score_difference=float(difference.mean()),ci_low=lo,ci_high=hi))
    for name,frame in [('endpoints',df),('joint_endpoints',joint),('development_curves',cv),
            ('training_losses',ls),('panel_metrics',pd.DataFrame(panels)),
            ('interventions',ab),('paired_comparisons',pd.DataFrame(comparisons))]:
        frame.to_csv(OUT/(name+'.csv'),index=False)
    audit_record=dict(finished_utc=datetime.fromtimestamp(status['time'],timezone.utc).isoformat(),
        scheduler_stages=86,capability_fits=17,joint_fits=25,final_model_evaluations=43,
        audited_evaluations=evaluations,audited_response_records=responses,archived_sparse_states=archived_states,
        source_and_data_hashes_verified=True,metrics_recounted=True,matched_teacher_start_predictions=True,
        zero_delta_rollbacks_match_base=True,one_seed=31,large_checkpoints_retained_on_server=True,
        manifest_fingerprint=stamp,archive_sha256=hashlib.sha256((HERE/'results/results_analysis_20261104.tar.gz').read_bytes()).hexdigest())
    diagnostic_path=OUT/'engineering_diagnostics_20261104.json'
    if diagnostic_path.exists():
        diagnostic=read(diagnostic_path)
        assert diagnostic['no_optimizer_steps'] and diagnostic['frozen_originals_unchanged']
        error=max(abs(x['manual']-x['builtin']) for x in diagnostic['loss_indexing'])
        assert error<2e-6
        gradient_error=max(x['max_abs_difference'] for x in diagnostic['checkpointed_gradients'])
        assert gradient_error==0
        assert all(x['R']==0 for x in diagnostic['weighted_component_gradients']
                   if x['component'] in ('allowed_nll','allowed_preservation'))
        audit_record['read_only_cuda_checks']=dict(loss_max_abs_error=error,
            checkpoint_gradient_max_abs_error=gradient_error,frozen_originals_unchanged=True)
    (OUT/'audit.json').write_text(json.dumps(audit_record,indent=2)+'\n')
    figures(cfg,df,joint,cv,ls,ab,baseline,marker)
    write_report(cfg,df,joint,cap,ab,baseline,audit_record,marker)
    print(json.dumps(audit_record,indent=2))
    print(joint[['condition','width','allowed_correct','blocked_refusal','allowed_refusal','general_accuracy']].to_string(index=False))


def figures(cfg,df,joint,cv,ls,ab,baseline,marker):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.18,'savefig.dpi':160,'pdf.fonttype':42})
    cap=df[df.stage=='capability']
    full=joint[joint.condition=='full'].iloc[0]
    widths=cfg['widths']
    legend=[Line2D([],[],color=COLORS[c],marker='o',label=LABELS[c]) for c in ['same','different','scattered']]
    legend += [Line2D([],[],color=COLORS['full'],ls='--',label='Full-model reference')]
    with PdfPages(OUT/'weatherwax_colocalization_results.pdf') as pdf:
        def save(fig,name,title,note):
            fig.suptitle(title,fontsize=16,fontweight='bold',y=.985)
            fig.text(.04,.025,note,fontsize=9,color='#3F3F46')
            fig.tight_layout(rect=(.02,.07,.99,.88))
            pdf.savefig(fig)
            fig.savefig(OUT/(name+'.png'))
            plt.close(fig)

        fig,axes=plt.subplots(2,3,figsize=(14,8))
        specs=[('blocked_refusal','Blocked refusal (higher alone is insufficient)',(0,105)),
               ('allowed_refusal','False refusal on allowed cases',(0,105)),
               ('allowed_correct','Allowed-task accuracy',(0,15)),
               ('refusal_balanced_accuracy','Refusal-label balanced accuracy',(40,60)),
               ('joint_score','Joint score (always refuse = 50%)',(0,100)),
               ('general_accuracy','Collateral exact-answer accuracy',(0,100))]
        for ax,(key,title,limits) in zip(axes.flat,specs):
            for condition in ['same','different','scattered']:
                s=joint[joint.condition==condition].sort_values('width')
                ax.plot(s.width,100*s[key],'-o',color=COLORS[condition],ms=4)
            ax.axhline(100*full[key],color=COLORS['full'],ls='--',lw=1.2)
            if key in ('joint_score','refusal_balanced_accuracy'):
                ax.axhline(50,color='#71717A',ls=':',lw=1)
            ax.set(xscale='log',xticks=widths,xticklabels=widths,ylim=limits,title=title,xlabel='Channels per objective',ylabel='%')
        fig.legend(handles=legend,loc='upper center',ncol=4,bbox_to_anchor=(.5,.945),frameon=False)
        save(fig,'01_endpoints','No condition acquired useful conditional refusal',
             'Final test: 256 matched worlds (one allowed + one blocked case each). One training seed. Disjoint uses twice the unique scalar budget at the same k.')

        fig,axes=plt.subplots(1,3,figsize=(14,5))
        palette=plt.get_cmap('viridis')(np.linspace(.08,.9,8))
        for ax,condition in zip(axes[:2],['same','scattered']):
            for color,width in zip(palette,widths):
                s=cv[(cv.stage=='capability')&(cv.condition==condition)&(cv.width==width)].sort_values('step')
                ax.plot(s.step,100*s.pointer_accuracy_both_categories,color=color,label=str(width))
            s=cv[(cv.stage=='capability')&(cv.condition=='full')].sort_values('step')
            ax.plot(s.step,100*s.pointer_accuracy_both_categories,'--',color=COLORS['full'],label='Full model')
            ax.axhline(10,color='gray',ls=':',label='Uniform 10-node guess')
            ax.set(title=LABELS[condition],xlabel='Capability-learning updates',ylabel='Development pointer accuracy (%)',ylim=(0,45))
        for condition in ['same','scattered']:
            s=cap[cap.condition==condition].sort_values('width')
            axes[2].plot(s.width,100*s.pointer_accuracy_both_categories,'-o',color=COLORS[condition],label=LABELS[condition])
        axes[2].axhline(100*cap[cap.condition=='full'].iloc[0].pointer_accuracy_both_categories,color=COLORS['full'],ls='--',label='Full model')
        axes[2].axhline(10,color='gray',ls=':',label='Uniform 10-node guess')
        axes[2].set(xscale='log',xticks=widths,xticklabels=widths,ylim=(0,45),title='Final held-out accuracy',xlabel='Channels',ylabel='Pointer accuracy, both categories (%)')
        axes[0].legend(title='Width',fontsize=8,ncol=2,frameon=False)
        axes[2].legend(fontsize=8,frameon=False)
        save(fig,'02_capability_acquisition','The capability-learning prerequisite failed',
             'Stage A answers both future categories. Uniform guessing over ten nodes gives 10%; this reference is not an optimal guessing strategy.')

        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for col,condition in enumerate(['same','different','scattered']):
            for color,width in zip(palette,widths):
                s=cv[(cv.stage=='joint')&(cv.condition==condition)&(cv.width==width)].sort_values('step')
                for row,key in enumerate(['blocked_refusal','allowed_refusal']):
                    axes[row,col].plot(s.step,100*s[key],color=color,label=str(width))
            for row,name in enumerate(['Blocked refusals','False refusals on allowed cases']):
                axes[row,col].set(title=LABELS[condition]+' — '+name,xlabel='Joint-training updates',ylabel='Development rate (%)',ylim=(0,105))
        axes[0,0].legend(title='Width',ncol=4,fontsize=8,frameon=False)
        save(fig,'03_refusal_learning','Refusal grows on allowed and blocked inputs together',
             'A high blocked-refusal rate is useful only if allowed false refusal remains low. All curves are development measurements, not test-selected checkpoints.')

        fig,axes=plt.subplots(1,3,figsize=(14,5))
        for ax,task in zip(axes,['addition','copy','reverse']):
            for condition in ['same','different','scattered']:
                s=joint[joint.condition==condition].sort_values('width')
                ax.plot(s.width,100*s[task+'_accuracy'],'-o',color=COLORS[condition],label=LABELS[condition])
            ax.axhline(100*baseline['general_test']['by_depth_or_task'][task]['allowed_correct'],color=COLORS['base'],ls=':',label='Frozen base')
            ax.axhline(100*full[task+'_accuracy'],color=COLORS['full'],ls='--',label='Full model')
            ax.set(xscale='log',xticks=widths,xticklabels=widths,ylim=(-3,105),title=task.capitalize(),xlabel='Channels per objective',ylabel='Exact-answer accuracy (%)')
        axes[0].legend(fontsize=8,frameon=False)
        save(fig,'04_collateral','Collateral performance: disjoint routing collapses broadly',
             '32 held-out cases per task. Reversal is already at a 0% base-model floor. These are small format-sensitive probes, not a general benchmark suite.')

        fig,axes=plt.subplots(1,3,figsize=(14,5))
        for ax,key,title,limits in zip(axes,['allowed_correct','refusal_balanced_accuracy','joint_score'],
                ['Allowed-task accuracy','Refusal-label balanced accuracy','Joint score'],[(0,15),(40,60),(0,100)]):
            for condition in ['same','different','scattered']:
                s=joint[joint.condition==condition].sort_values('unique_trainable')
                ax.plot(s.unique_trainable,100*s[key],'-o',color=COLORS[condition],label=LABELS[condition])
            if key!='allowed_correct':
                ax.axhline(50,color='gray',ls=':',label='Always-refuse policy')
            ax.set(xscale='log',ylim=limits,title=title,xlabel='Unique trainable scalar parameters',ylabel='%')
        axes[0].legend(fontsize=8,frameon=False)
        save(fig,'05_total_budget','Equal-total-budget views do not rescue the learned behaviour',
             'Different(k) has the same total scalar count as Shared(2k). Per-objective capacity differs. These are parameter-budget comparisons, not matched compute.')

        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for ax,placement in zip(axes,['same','different']):
            job=f'native__k4__{placement}'
            intact=joint[joint.job==job].iloc[0]
            entries=[dict(intervention='Intact',**{k:intact[k] for k in ['blocked_refusal','allowed_refusal','allowed_correct','allowed_truncated']})]
            entries+=ab[ab.job==job].to_dict('records')
            x=np.arange(len(entries))
            for j,(key,label,color) in enumerate([('blocked_refusal','Blocked refusal','#2563EB'),
                    ('allowed_refusal','Allowed false refusal','#D97706'),('allowed_correct','Allowed correct','#0F766E'),
                    ('allowed_truncated','Allowed truncation','#BE123C')]):
                ax.bar(x+(j-1.5)*.19,[100*r[key] for r in entries],width=.18,label=label,color=color)
            names={'Intact':'Intact','half_channels':'Half channels\nrestored','capability':'C restored',
                   'refusal':'R restored','all_updates_rolled_back':'All increments\nrestored'}
            ax.set(xticks=x,xticklabels=[names[r['intervention']] for r in entries],ylim=(0,105),title=LABELS[placement],ylabel='%')
        axes[0].legend(fontsize=8,frameon=False)
        save(fig,'06_interventions','Width-four interventions show update dependence, not useful entanglement',
             'Restoring both updates exactly reproduces base responses. Restoring only C or R in the disjoint model produces truncated outputs, not recovered skill.')

        fig,axes=plt.subplots(1,3,figsize=(14,5))
        for ax,key,word,title in zip(axes,['blocked_refusal','allowed_refusal','allowed_correct'],
                ['wording_blocked_refusal','wording_allowed_refusal','wording_allowed_correct'],
                ['Blocked refusal','Allowed false refusal','Allowed-task accuracy']):
            for condition in ['same','different','scattered']:
                s=joint[joint.condition==condition].sort_values('width')
                ax.plot(s.width,100*s[key],'-o',color=COLORS[condition],ms=3)
                ax.plot(s.width,100*s[word],'--s',color=COLORS[condition],ms=3)
            ax.set(xscale='log',xticks=widths,xticklabels=widths,ylim=(0,105) if key!='allowed_correct' else (0,15),
                   xlabel='Channels per objective',ylabel='%',title=title)
        fig.legend(handles=legend[:3]+[Line2D([],[],color='black',ls='-',label='Standard test'),
            Line2D([],[],color='black',ls='--',label='Held-out wording + worlds')],loc='upper center',ncol=5,bbox_to_anchor=(.5,.925),frameon=False,fontsize=9)
        save(fig,'07_wording','Held-out wording changes refusal frequency, not useful conditional performance',
             'Standard test: 256 worlds. Held-out wording: 64 different worlds. Both world and wording change; this is not a paired phrasing-only comparison.')

        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for col,condition in enumerate(['same','different','scattered']):
            for color,width in zip(palette,widths):
                for row,(stage,key) in enumerate([('capability','allowed_nll'),('joint','allowed_nll')]):
                    selected_condition='same' if stage=='capability' and condition=='different' else condition
                    s=ls[(ls.stage==stage)&(ls.condition==selected_condition)&(ls.width==width)].sort_values('step')
                    axes[row,col].plot(s.step,s[key].rolling(32,min_periods=1).mean(),color=color,label=str(width))
            for row,stage in enumerate(['Capability stage','Joint stage']):
                axes[row,col].set(title=LABELS[condition]+' — '+stage,xlabel='Updates',ylabel='Allowed-answer NLL (32-step mean)')
        axes[0,0].legend(title='Width',ncol=4,fontsize=8,frameon=False)
        save(fig,'08_losses','Loss reduction is not evidence of successful pointer following',
             'Completion NLL includes the answer and EOS tokens. Disjoint Stage A reuses the same capability checkpoint as Shared Stage A. See accuracy plots for task success.')

        fig,axes=plt.subplots(1,2,figsize=(14,5))
        selected=['base','native__k32__same','full__k0__same']
        short=['Frozen base','Native, k=32','Full model']
        x=np.arange(3)
        for j,(label,key,name,color) in enumerate([
                ('allowed','correct','Allowed accuracy','#2563EB'),('blocked','correct','Future-blocked accuracy','#0F766E'),
                ('allowed','next_marker_agreement','Allowed: predicts marker successor','#D97706'),
                ('blocked','next_marker_agreement','Blocked: predicts marker successor','#BE123C')]):
            values=[marker[(marker.job==job)&(marker.label==label)&(marker.depth=='all')].iloc[0][key] for job in selected]
            axes[0].bar(x+(j-1.5)*.18,100*np.array(values),width=.17,color=color,label=name)
        axes[0].set(xticks=x,xticklabels=short,ylim=(0,100),title='Stage-A test behaviour',ylabel='%')
        axes[0].legend(fontsize=8,frameon=False)
        for job,color,label in zip(selected,['#52525B','#2563EB','#7C3AED'],short):
            for category,style in [('allowed','-'),('blocked','--')]:
                s=marker[(marker.job==job)&(marker.label==category)&(marker.depth!='all')].copy()
                s['depth']=s.depth.astype(int)
                s=s.sort_values('depth')
                axes[1].plot(s.depth,100*s.correct,style+'o',color=color,label=label+': '+category)
        axes[1].set(xticks=[2,3,4],ylim=(0,100),xlabel='Requested hops from start',ylabel='Test accuracy (%)',title='The future-blocked advantage is largest at two hops')
        axes[1].legend(fontsize=8,frameon=False)
        save(fig,'09_marker_shortcut','Post-hoc diagnostic: learning to follow the marked node',
             'An oracle that follows the marker once scores 59.8% on future-blocked cases and 0% on allowed cases. Agreement suggests a shortcut, not a circuit identification.')


def write_report(cfg,df,joint,cap,ab,baseline,audit,marker):
    pct=lambda x:f'{100*x:.1f}'
    base=baseline['test']['metrics']
    full=joint[joint.condition=='full'].iloc[0]
    fullcap=cap[cap.condition=='full'].iloc[0]
    lines=['# Weatherwax location/channel sweep: completed pilot',
        '',f"Analysed 4 November 2026. Finished {audit['finished_utc']}. All 86 scheduler stages completed, with no failed jobs.",
        '', 'See [the engineering audit and selected next actions](ENGINEERING_AUDIT.md) for the checked training issues and direction-avoidance plan.',
        '', '**The pilot did not learn the underlying capability well enough to answer the co-localisation question.** '
        'Joint training mostly learned to refuse both allowed and blocked cases. High blocked-refusal percentages are not a successful outcome here.',
        '', '## Capability acquisition failed before refusal training',
        '',f"The frozen base answered {pct(base['allowed_correct'])}% of allowed test questions correctly. "
        f"The 17 capability-only checkpoints achieved {pct(cap.allowed_correct.min())}–{pct(cap.allowed_correct.max())}% allowed accuracy. "
        f"The unrestricted full-model control reached only {pct(fullcap.allowed_correct)}% allowed accuracy "
        f"({pct(fullcap.pointer_accuracy_both_categories)}% over both future categories). "
        'Uniform guessing over ten node labels gives 10%. This reference is not an optimal guessing strategy. '
        'Some models did learn a useful shortcut on the future-blocked subset, described below, but none demonstrated strong allowed-task competence.',
        '', 'I should have established a successful capability-learning control before running the full sweep. '
        'The completed run is useful as a training-failure analysis, not as positive evidence for co-localisation.',
        '', '### A marker-successor shortcut explains part of the apparent capability gain',
        '', 'The native k=32 capability checkpoint answered 47.7% of future-blocked cases correctly but only 5.1% of allowed cases. '
        'Its outputs matched the successor of the marked node on 75.4% and 59.4% of those subsets respectively. '
        'In this dataset, following the marker once is correct on 59.8% of future-blocked test cases and 0% of allowed cases. '
        'This is structural correlation in the task, not train/test world leakage. The output agreement is consistent with a shortcut; it does not identify a circuit.',
        '', 'Pooled capability accuracy therefore overstates success on the intended operation: native k=32 reaches 26.4% pooled accuracy '
        'while failing on the allowed subset. The next training curriculum must break the marker shortcut.',
        '', '## Final test results',
        '', 'One seed (31); 256 matched worlds, each with one allowed and one blocked question. '
        'All figures below are percentages. The channel budget is per objective. Disjoint(k) uses 2k unique channels; Shared(k) uses k. '
        'The scattered mask matches Shared(k) in scalar count. Full-model learning rate differs from the sparse recipe.',
        '', '| Reference | Capability-stage allowed correct | Joint-stage blocked refusal | Joint-stage allowed false refusal | Joint-stage allowed correct | Collateral exact accuracy |',
        '|---|---:|---:|---:|---:|---:|',
        f"| Frozen base | {pct(base['allowed_correct'])} | {pct(base['blocked_refusal'])} | {pct(base['allowed_refusal'])} | {pct(base['allowed_correct'])} | {pct(baseline['general_test']['metrics']['allowed_correct'])} |",
        f"| Full-model SFT | {pct(fullcap.allowed_correct)} | {pct(full.blocked_refusal)} | {pct(full.allowed_refusal)} | {pct(full.allowed_correct)} | {pct(full.general_accuracy)} |"]
    for condition in ['same','different','scattered']:
        lines += ['',f'### {LABELS[condition]}','',
            '| k | Unique scalars | Capability-stage allowed correct | Blocked refusal | Allowed false refusal | Allowed correct | Collateral exact accuracy |',
            '|---:|---:|---:|---:|---:|---:|---:|']
        for r in joint[joint.condition==condition].sort_values('width').itertuples():
            teacher_condition='same' if condition=='different' else condition
            teacher=cap[(cap.condition==teacher_condition)&(cap.width==r.width)].iloc[0]
            lines.append(f'| {r.width} | {r.unique_trainable:,} | {pct(teacher.allowed_correct)} | {pct(r.blocked_refusal)} | {pct(r.allowed_refusal)} | {pct(r.allowed_correct)} | {pct(r.general_accuracy)} |')
    lines += ['', '## What these results establish',
        '', f"- **No joint checkpoint qualified:** {int(joint.qualified.sum())}/25 met both 90% blocked refusal and 80% allowed accuracy. "
        f"The best refusal-label balanced accuracy was {pct(joint.refusal_balanced_accuracy.max())}%; always refusing gives 50%.",
        '- Increasing scattered scalar budget made refusal easier to express, but it rose on allowed and blocked prompts together. '
        'This is saturation of a mostly unconditional refusal response, not identification of an optimal budget for selective refusal.',
        '- Equal-total-budget comparisons do not resolve the failed prerequisite. '
        'A high joint score near 50% can be achieved by refusing everything, without learning pointer following.',
        '- Shared and disjoint job names describe gradient ownership, not independently validated capability and refusal circuits.',
        '', '## Collateral performance',
        '', f"Frozen-base exact accuracy across the small collateral panel was {pct(baseline['general_test']['metrics']['allowed_correct'])}%. "
        f"After joint training, Shared scored {pct(joint[joint.condition=='same'].general_accuracy.min())}–{pct(joint[joint.condition=='same'].general_accuracy.max())}%, "
        f"Scattered {pct(joint[joint.condition=='scattered'].general_accuracy.min())}–{pct(joint[joint.condition=='scattered'].general_accuracy.max())}%, "
        f"Disjoint {pct(joint[joint.condition=='different'].general_accuracy.min())}–{pct(joint[joint.condition=='different'].general_accuracy.max())}%, "
        f"and Full {pct(full.general_accuracy)}%. These measure behaviour after training, not irreversible loss of stored information.",
        '', 'Each collateral task has only 32 test examples. Baseline accuracy was '+', '.join(
            f"{task} {pct(m['allowed_correct'])}%" for task,m in baseline['general_test']['by_depth_or_task'].items())+'. '
        'The reversal task therefore has a floor effect. Exact-output scoring and the 12-token generation cap also limit these probes.',
        '', '## Width-four interventions',
        '', '| Condition | Restored increment | Blocked refusal | Allowed false refusal | Allowed correct | Allowed truncation |',
        '|---|---|---:|---:|---:|---:|']
    for r in ab.itertuples():
        lines.append(f"| {r.job} | {r.intervention} | {pct(r.blocked_refusal)} | {pct(r.allowed_refusal)} | {pct(r.allowed_correct)} | {pct(r.allowed_truncated)} |")
    lines += ['', 'Restoring all trained increments reproduced the frozen base response-for-response in both width-four models. '
        'Restoring only C or R in the disjoint model produced truncated output on all test examples. '
        'Breaking output is not evidence of useful capability/refusal entanglement. No strong acquired capability was present to protect.',
        '', '## Interpretation limits and next test',
        '', 'Disjoint refusal weights received blocked-refusal NLL but no allowed-answer NLL or allowed-example teacher KL; '
        'those gradients were confined to C. Only the unrelated conservation term reached both sets. '
        'Consequently spatial allocation and per-region loss exposure change together. The collateral collapse cannot be attributed solely to spatial separation.',
        '', 'The next pilot should first demonstrate strong held-out capability acquisition in the full-model and a wide routed control, '
        'then add refusal. Use a simpler/shorter-hop task or a better validated acquisition recipe. '
        'Give the refusal-owning region both positive and negative policy-decision supervision, without requiring it to learn the allowed answer identity. '
        'Only after those checks should the full width/location sweep be repeated.',
        '', 'No new training has been launched from these results. This run does not test adversarial removal or avoidance of the old refusal direction. '
        'One seed cannot establish reproducibility across training runs, an optimal channel count, or inherent superiority of shared over disjoint locations.',
        '', '## Artifacts and audit',
        '', '[Nine-page graph PDF](weatherwax_colocalization_results.pdf). '
        'CSV files contain every endpoint, development curve, per-update loss, intervention and paired budget comparison. '
        'The paired bootstrap uses matched test worlds; its intervals do not estimate training-seed uncertainty.',
        '', f"Recounted {audit['audited_response_records']:,} saved response records across {audit['audited_evaluations']} reports. "
        'Verified source/data fingerprints, generated path labels, disjoint world splits, final checkpoint provenance, all 40 sparse final states against their optimiser checkpoints, '
        'and the identical capability checkpoints used to start Shared/Disjoint conditions. The runner also completed its frozen-original-weight assertions.',
        '', 'Measurements, logs and sparse checkpoints are local under `../results/weatherwax-20261103/`. '
        'The 18 GB complete result tree, including full-model weights and larger teacher caches, remains at '
        '`mn866@weatherwax.cs.bath.ac.uk:/mnt/fast1/mn866/casr-colocalization-20261103/results`. '
        'No remote artifacts were deleted. `checkpoint_inventory.json` lists files retained on the server.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
