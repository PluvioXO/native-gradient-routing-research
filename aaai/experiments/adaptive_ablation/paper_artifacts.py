"""Generate the final ablation tables and vector figures from audited bank records."""
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / 'figures/adaptive_ablation'
MODES = [('aware','Neither'),('preservation_only','Preservation'),('conservation_only','Conservation'),('preserve','Both')]


def main():
    audit = json.loads((HERE/'analysis/audit.json').read_text())
    assert audit['complete'] and audit['evaluated_banks'] == 24
    rows = json.loads((HERE/'analysis/per_bank.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    def vals(support, mode, field):
        group = sorted([r for r in rows if r['support']==support and r['mode']==mode],key=lambda r:r['seed'])
        assert [r['seed'] for r in group]==[31,47,71]
        return [r[field] for r in group]
    def cell(support, mode, field):
        v=vals(support,mode,field)
        return f'${st.mean(v):.1f} \\pm {st.stdev(v):.1f}$'
    lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r' & \multicolumn{2}{c}{Native, 4 channels} & \multicolumn{2}{c}{LoRA, rank 1}\\',r'Losses & Accuracy & $\Delta$PPL & Accuracy & $\Delta$PPL\\',r'\midrule',r'Frozen base & 29.1 & 0.0 & 29.1 & 0.0\\']
    for mode,label in MODES:
        lines.append(label+' & '+' & '.join(cell(s,mode,k) for s in ('native','lora') for k in ('accuracy_8','wiki_change'))+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'primary_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',r'Support & Losses & Owner & 2 & 4 & No owner & All 8 & Calibrated\\',r'\midrule']
    for support in ('native','lora'):
        for mode,label in MODES:
            lines.append(('Native' if support=='native' else 'LoRA')+' & '+label+' & '+' & '.join(cell(support,mode,k) for k in ('accuracy_1','accuracy_2','accuracy_4','accuracy_7','accuracy_8','calibrated_accuracy'))+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'composition_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',r'Support & Losses & MC $\Delta$Acc (pp) & RL hours & Response tokens ($10^3$)\\',r'\midrule']
    for support in ('native','lora'):
        for mode,label in MODES:
            h=[x/3600 for x in vals(support,mode,'training_seconds')]
            tok=[x/1000 for x in vals(support,mode,'response_tokens')]
            lines.append(('Native' if support=='native' else 'LoRA')+' & '+label+' & '+cell(support,mode,'mc_change')+f' & ${st.mean(h):.2f} \\pm {st.stdev(h):.2f}$ & ${st.mean(tok):.1f} \\pm {st.stdev(tok):.1f}$'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'retention_compute_table.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.5))
    for support,color,marker,label in [('native','#2166AC','s','Native, 4 channels'),('lora','#D97706','o','LoRA, rank 1')]:
        for ax,field in zip(axes,('accuracy_8','wiki_change')):
            groups=[vals(support,mode,field) for mode,_ in MODES]
            ax.errorbar(range(4),[st.mean(v) for v in groups],yerr=[st.stdev(v) for v in groups],color=color,marker=marker,capsize=3,label=label,lw=1.4)
    axes[0].axhline(29.0625,color='#666666',ls='--',lw=1,label='Frozen base')
    axes[1].axhline(0,color='#666666',ls='--',lw=1)
    axes[0].set(title='(a) All-eight test accuracy',ylabel='Accuracy (%)',ylim=(25,67))
    axes[1].set(title='(b) Natural-text retention',ylabel='WikiText perplexity increase (%)',ylim=(-1,22))
    for ax in axes:
        ax.set_xticks(range(4),['Neither','Preserve','Conserve','Both'])
        ax.set_xlabel('Auxiliary losses'); ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.10,1,1));fig.savefig(OUT/'loss_ablation.pdf');fig.savefig(OUT/'loss_ablation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),sharey=True)
    for ax,support in zip(axes,('native','lora')):
        for (mode,label),color,marker in zip(MODES,('#777777','#2166AC','#2A8D69','#D97706'),('s','o','D','^')):
            groups=[vals(support,mode,'accuracy_'+str(k)) for k in (1,2,4,8)]
            ax.errorbar([1,2,4,8],[st.mean(v) for v in groups],yerr=[st.stdev(v) for v in groups],label=label,color=color,marker=marker,capsize=2)
        ax.axhline(29.0625,color='#AAAAAA',ls='--');ax.set(title='Native' if support=='native' else 'LoRA',xlabel='Active task increments',ylabel='Accuracy (%)',xticks=[1,2,4,8],ylim=(25,70));ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.10,1,1));fig.savefig(OUT/'coexistence_curves.pdf');plt.close(fig)
    contrasts=[]
    for support in ('native','lora'):
        for field in ('accuracy_8','wiki_change','mc_change'):
            for mode,_ in MODES[1:]:
                paired=[a-b for a,b in zip(vals(support,mode,field),vals(support,'aware',field))]
                contrasts.append(dict(support=support,field=field,mode=mode,paired_changes=paired,mean=st.mean(paired),sample_sd=st.stdev(paired)))
    (HERE/'analysis/paired_contrasts.json').write_text(json.dumps(contrasts,indent=2))
    from analyse import sha
    archive = HERE/'results/casr-adaptive-ablation-results-20261031.tar.gz'
    provenance=dict(audit=audit,inputs={p.name:sha(p) for p in [archive,HERE/'analyse.py',HERE/'analysis/per_bank.json',Path(__file__)]},outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ('.tex','.pdf','.png')})
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print((OUT/'primary_table.tex').read_text())


if __name__=='__main__':
    main()
