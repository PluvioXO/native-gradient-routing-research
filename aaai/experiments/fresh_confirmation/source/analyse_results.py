"""Analyse the prespecified six-seed comparison only after every job completes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import numpy as np
from scipy import stats


def paired(values):
    values=np.asarray(values,float);n=len(values);assert n>=2 and np.isfinite(values).all()
    avg=float(values.mean());sd=float(values.std(ddof=1));se=sd/n**.5
    interval=float(stats.t.ppf(.975,n-1))*se
    p=float(stats.ttest_1samp(values,0).pvalue) if sd else (1. if avg==0 else 0.)
    return dict(n=n,paired_pp=values.tolist(),mean_pp=avg,sd_pp=sd,ci95_pp=[avg-interval,avg+interval],p_two_sided=p)


def holm(records):
    maximum=0.
    for rank,index in enumerate(sorted(range(len(records)),key=lambda i:records[i]['p_two_sided'])):
        adjusted=min(1.,(len(records)-rank)*records[index]['p_two_sided'])
        maximum=max(maximum,adjusted);records[index]['p_holm']=maximum
        r=records[index];r['meets_prespecified_success']=r['mean_pp']>0 and r['ci95_pp'][0]>0 and maximum<.05


def analyse(root):
    output=root/'outputs/main';plan=json.loads((output/'experiment_plan.json').read_text());assert not plan['smoke']
    status=json.loads((output/'SHARD_COMPLETE.json').read_text())
    assert status['launcher_state']=='complete' and status['completed']==len(plan['jobs']) and not status['failed']
    protocol=plan['protocol'];seeds=protocol['seeds'];records=[]
    for job in plan['jobs']:
        folder=output/'fits'/job['id'];done=json.loads((folder/'completed.json').read_text())
        if job['kind']!='merge':continue
        for name,expected in done['artifact_hashes'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
        summary=json.loads((folder/'summary.json').read_text())
        for r in summary['scores']:records.append(dict(support=job['support'],seed=job['seed'],mode=job['mode'],**r))
    lookup={(r['support'],r['seed'],r['mode'],r['split'],r['task'],r['mask']):100*r['accuracy'] for r in records}
    def score(s,seed,m,tasks=range(8),split='test',mask='all'):
        return st.mean(lookup[s,seed,m,split,t,mask] for t in tasks)
    primary=[];secondary=[]
    for support in protocol['supports']:
        deltas=[score(support,seed,'foreign_conserve')-score(support,seed,'self_conserve') for seed in seeds]
        primary.append(dict(support=support,contrast='foreign_conserve minus self_conserve',**paired(deltas)))
        for label,tasks in [('without_pointer',[t for t in range(8) if t!=1])]+[(f'task_{t}',[t]) for t in range(8)]:
            secondary.append(dict(support=support,contrast=label,**paired([score(support,seed,'foreign_conserve',tasks)-score(support,seed,'self_conserve',tasks) for seed in seeds])))
        secondary.append(dict(support=support,contrast='foreign_conserve minus conservation_only',**paired([score(support,seed,'foreign_conserve')-score(support,seed,'conservation_only') for seed in seeds])))
    holm(primary)
    destination=root/'analysis';destination.mkdir(exist_ok=True)
    with (destination/'per_task.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result=dict(primary=primary,secondary_exploratory=secondary,protocol=protocol,
        source_plan_sha256=hashlib.sha256((output/'experiment_plan.json').read_bytes()).hexdigest(),
        caveat='Six paired training/allocation seeds on one fixed fresh bank, one model, shared task families, and a prespecified harder depth distribution. Secondary comparisons are unadjusted exploratory results.')
    (destination/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors={'initial':'#64748B','conservation_only':'#8B5CF6','self_conserve':'#D97706','foreign_conserve':'#2563EB'}
    with PdfPages(destination/'confirmation_graphs.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,5.5))
        for ax,support in zip(axes,protocol['supports']):
            for x,mode in enumerate(colors):
                values=[score(support,seed,mode) for seed in seeds]
                ax.bar(x,st.mean(values),color=colors[mode],alpha=.7,yerr=st.stdev(values),capsize=3)
                ax.scatter(x+np.linspace(-.17,.17,len(values)),values,color=colors[mode],s=28,edgecolor='white',linewidth=.5)
            base=score(support,seeds[0],'initial',mask='base');ax.axhline(base,ls='--',color='black',label=f'Frozen base {base:.1f}%')
            ax.set(title=support.title(),xticks=range(4),xticklabels=['Independent','Conserve','Self + C','Foreign + C'],ylabel='All-eight test accuracy (%)',ylim=(0,100));ax.legend();ax.grid(axis='y',alpha=.15)
        fig.suptitle('Fresh-bank confirmation: fixed recipe, six new seeds');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5))
        for i,r in enumerate(primary):
            ax.scatter(r['paired_pp'],[i]*len(seeds),s=45,label=r['support']);ax.errorbar(r['mean_pp'],i,xerr=[[r['mean_pp']-r['ci95_pp'][0]],[r['ci95_pp'][1]-r['mean_pp']]],fmt='D',color='black',capsize=5)
        ax.axvline(0,ls='--',color='gray');ax.set(yticks=range(2),yticklabels=[r['support'] for r in primary],xlabel='Foreign minus self all-eight accuracy (pp)',title='Prespecified contrasts: individual seeds and paired mean ± 95% t interval')
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    print(json.dumps(primary,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();analyse(args.root)
