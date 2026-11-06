"""Export every defense/removal/repair trajectory; no post-hoc attack filtering."""
import argparse
from pathlib import Path
import pandas as pd
from experiment import check,digest,read,sha,write


def main(root):
    m=check(root);selection=read(root/'selection.json')
    assert (root/'COMPLETE.json').exists()
    out=root.parent/'analysis';out.mkdir(exist_ok=True)
    defense=[];curves=[];endpoints=[];history=[]
    for key,spec in m['defense_jobs'].items():
        folder=root/'defense'/key
        for f in sorted(folder.glob('dev_*.json')):
            r=read(f);defense.append(dict(candidate=key,**spec,**r['selection'],parameter_hash=r['parameter_hash']))
    last_training=max(read(root/'attacks'/k/'DONE.json')['ended_unix'] for k in selection['jobs'])
    for key,spec in selection['jobs'].items():
        folder=root/'attacks'/key
        updates=read(folder/'history.json');assert len(updates)==384
        history.extend(dict(job=key,**{**spec,**r}) for r in updates)
        for phase in ['remove','repair']:
            for f in sorted(folder.glob(phase+'_*.json')):
                if not f.stem.removeprefix(phase+'_').isdigit():continue
                r=read(f);curves.append(dict(job=key,**spec,phase=phase,
                    cumulative_updates=r['cumulative_updates'],**r['selection']))
        for f in sorted(folder.glob('test_*.json')):
            r=read(f);assert r['started_unix']>=last_training
            endpoints.append(dict(job=key,**spec,endpoint=r['endpoint'],phase=r['phase'],
                cumulative_updates=r['cumulative_updates'],**r['selection']))
    for name,rows in [('defense_candidates.csv',defense),('learning_curves.csv',curves),
                      ('endpoints.csv',endpoints),('training.csv',history)]:
        pd.DataFrame(rows).to_csv(out/name,index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    frame=pd.DataFrame(curves)
    with PdfPages(out/'coupling_trajectories.pdf') as pdf:
        for objective,lr in [('preserve',x) for x in m['config']['attack_lrs']]+[('ordinary',1e-6),('benign',1e-6)]:
            fig,axes=plt.subplots(2,4,figsize=(15,8),sharex=True,sharey=True)
            for i,parent in enumerate(m['config']['parents']):
                for j,arm in enumerate(m['config']['arms']):
                    ax=axes[i,j]
                    d=frame[(frame.parent==parent)&(frame.arm==arm)&(frame.objective==objective)&(frame.lr==lr)]
                    d=d.drop_duplicates('cumulative_updates',keep='last').sort_values('cumulative_updates')
                    for field,label,color,style in [('allowed_accuracy','Allowed correct','#2563eb','-'),
                        ('blocked_accuracy','Blocked correct','#d97706','-'),
                        ('blocked_refusal','Blocked refusal','#dc2626','--'),
                        ('general_accuracy','Arithmetic/copy','#64748b',':')]:
                        ax.plot(d.cumulative_updates,100*d[field],style,color=color,label=label,marker='.',markersize=3)
                    if d.empty:ax.text(.1,.5,'No qualified defense',transform=ax.transAxes)
                    ax.axvline(256,color='black',lw=.8,alpha=.5)
                    ax.set_xscale('symlog',linthresh=1,base=2);ax.set_xlim(0,384);ax.set_ylim(-3,103)
                    ax.set_title(parent+' / '+arm);ax.set_xlabel('Cumulative updates')
                    if j==0:ax.set_ylabel('Development rate (%)')
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,.95))
            fig.suptitle(f'{objective}, removal LR {lr:.0e}; repair begins at 256 (LR 1e-6) — one seed')
            fig.tight_layout(rect=(0,0,1,.88));pdf.savefig(fig);plt.close(fig)
    write(out/'provenance.json',dict(manifest_sha256=sha(root/'manifest.json'),selection=selection,
        heldout_after_all_training=True,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='provenance.json'}))
    print('Analysis saved',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
