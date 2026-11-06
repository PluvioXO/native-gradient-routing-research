"""Fixed-grid removal curves and held-out endpoints; no training or selection."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from attack import read,write,sha,sources
from protocol import quality


def main(root):
    m=read(root/'manifest.json');assert m['sources']==sources() and not m['smoke']
    assert read(root/'COMPLETE.json')['jobs']==12
    out=root.parent/'analysis';out.mkdir(exist_ok=True)
    curves=[];endpoints=[];losses=[];thresholds=[];cases=[];checks=[]
    last_training=max(read(root/'jobs'/k/'DONE.json')['ended_unix'] for k in m['jobs'])
    for key,spec in m['jobs'].items():
        folder=root/'jobs'/key;done=read(folder/'DONE.json');parent=Path(m['parent_root'])/'parents'/spec['parent']
        base=read(parent/'baseline_dev.json');history=read(folder/'history.json')
        assert len(history)==spec['limit']==256 and [h['step'] for h in history]==list(range(1,257))
        meta=dict(job=key,parent=spec['parent'],objective=spec['objective'],lr=spec['lr'])
        measured=[]
        for path in sorted(folder.glob('dev_*.json')):
            r=read(path);s=quality(r['report'],base,r['step']);assert s==r['selection']
            assert r['fingerprint']==done['fingerprint'] and (folder/f'step_{r["step"]:04d}.pt').exists()
            measured.append(s);curves.append(dict(meta,**{k:v for k,v in s.items() if k!='rank'}))
        expected=max(measured,key=lambda r:r['rank']);assert expected==done['best']['selection']
        thresholds.append(dict(meta,first_zero_refusal=next((r['step'] for r in measured if r['refusal']==0),None),
            first_task_success=next((r['step'] for r in measured if r['task_success']),None),
            first_full_success=next((r['step'] for r in measured if r['success']),None),
            final_development_success=measured[-1]['success']))
        losses.extend(dict(meta,**h) for h in history)
        testbase=read(parent/'baseline_test.json')
        for endpoint in ['best','final']:
            r=read(folder/f'test_{endpoint}.json');assert r['started_unix']>=last_training and r['parameters_unchanged']
            s=quality(r['report'],testbase,r['step']);assert s==r['selection']
            endpoints.append(dict(meta,endpoint=endpoint,**{k:v for k,v in s.items() if k!='rank'}))
            for case,values in r['report']['task']['metrics']['by_case'].items():
                cases.append(dict(meta,endpoint=endpoint,case=case,**values))
        checks.append(dict(job=key,history_steps=256,development_points=len(measured),selection_reproduced=True,heldout_after_all_training=True))
    for name,values in [('learning_curves.csv',curves),('endpoints.csv',endpoints),('training.csv',losses),('first_thresholds.csv',thresholds),('case_scores.csv',cases)]:
        pd.DataFrame(values).to_csv(out/name,index=False)
    frame=pd.DataFrame(curves);ends=pd.DataFrame(endpoints)
    plot(out,frame,ends,m,root)
    result=dict(config=m['config'],all_checks_passed=True,jobs=12,curves=curves,endpoints=endpoints,
        thresholds=thresholds,checks=checks,scope='One seed, synthetic one-/two-hop navigation; unrestricted SFT only',
        manifest_sha256=sha(root/'manifest.json'),hardware_validation=read(root.parent/'hardware/hardware_validation.json'))
    write(out/'results.json',result)
    lines=['# Skill-preserving refusal removal — completed','','All twelve fixed-budget attacks reported. One seed; no universal robustness conclusion.','',
        '| Parent | Objective | LR | Endpoint | Step | Allowed (%) | Formerly blocked correct (%) | Blocked refusal (%) | Retention | Joint success |',
        '|---|---|---:|---|---:|---:|---:|---:|---|---|']
    for r in endpoints:
        lines.append(f'| {r["parent"]} | {r["objective"]} | {r["lr"]:.0e} | {r["endpoint"]} | {r["step"]} | {100*r["allowed_accuracy"]:.1f} | {100*r["blocked_accuracy"]:.1f} | {100*r["blocked_refusal"]:.1f} | {r["retention_success"]} | {r["success"]} |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(out/'provenance.json',dict(sources=m['sources'],files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='provenance.json'}))
    print('Completed removal analysis:',out,flush=True)


def plot(out,curves,endpoints,manifest,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#2563eb','#d97706','#0f766e'];rates=manifest['config']['learning_rates']
    def trajectories(fields,titles):
        fig,axes=plt.subplots(2,3,figsize=(12,7))
        for i,parent in enumerate(['shared4','shared8']):
            for j,(field,title) in enumerate(zip(fields,titles)):
                ax=axes[i,j]
                for lr,color in zip(rates,colors):
                    for objective,style in [('ordinary','--'),('preserve','-')]:
                        s=curves[(curves.parent==parent)&(curves.lr==lr)&(curves.objective==objective)].sort_values('step')
                        ax.plot(s.step,100*s[field],style,color=color,label=f'{objective}, LR {lr:.0e}')
                ax.set_xscale('symlog',linthresh=1,base=2)
                ax.set(title=parent+' — '+title,xlabel='SFT updates (256 = one blocked-data pass)',ylabel='Rate (%)',ylim=(-2,103))
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,0,1,.90));return fig
    with PdfPages(out/'removal_graphs.pdf') as pdf:
        fig=trajectories(['allowed_accuracy','blocked_accuracy','blocked_refusal'],['Allowed paths','Formerly blocked paths','Formerly blocked refusal'])
        pdf.savefig(fig);plt.close(fig)
        fig=trajectories(['general_accuracy','worst_case_accuracy','refusal'],['Arithmetic/copy controls','Worst navigation case','All-navigation refusal'])
        pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        selected=endpoints[endpoints.endpoint=='best']
        for ax,field,title in zip(axes.flat,['allowed_accuracy','blocked_accuracy','blocked_refusal','general_accuracy'],
                                  ['Allowed paths','Formerly blocked paths','Blocked refusal','Arithmetic/copy controls']):
            for j,endpoint in enumerate(['best','final']):
                vals=endpoints[endpoints.endpoint==endpoint].set_index('job').loc[selected.job][field]
                ax.bar(np.arange(12)+(j-.5)*.36,100*vals,.35,label='Dev selected' if endpoint=='best' else 'Final')
            ax.set_xticks(range(12),[f'{r.parent[-1]}/{r.objective[0]}/{r.lr:.0e}' for r in selected.itertuples()],rotation=55,ha='right')
            ax.set(title=title,ylabel='Held-out rate (%)',ylim=(0,105));ax.legend(frameon=False)
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(12,6));ax.axis('off')
        cells=[[r.parent,r.objective,f'{r.lr:.0e}',int(r.step),f'{100*r.allowed_accuracy:.1f}',f'{100*r.blocked_accuracy:.1f}',
                f'{100*r.blocked_refusal:.1f}','Yes' if r.retention_success else 'No','Yes' if r.success else 'No'] for r in selected.itertuples()]
        table=ax.table(cellText=cells,colLabels=['Parent','Attack','LR','Step','Allowed','Blocked correct','Blocked refuse','Retention','Joint pass'],loc='center',cellLoc='center')
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.7)
        ax.set_title('Development-selected endpoints on held-out worlds — one seed; percentages',pad=20)
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
