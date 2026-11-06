"""Combine audited factorial cells, retaining paired seeds and original controls once."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'figures/preservation_components'
MODES=[('aware','None',0,0,0),('self_only','Self',1,0,0),('foreign_only','Foreign',0,1,0),('preservation_only','Self+foreign',1,1,0),
       ('conservation_only','None',0,0,1),('self_conserve','Self',1,0,1),('foreign_conserve','Foreign',0,1,1),('preserve','Self+foreign',1,1,1)]
SEEDS=[31,47,71]
FIELDS=('accuracy_1','accuracy_8','wiki_change','mc_change','calibrated_accuracy','training_seconds')


def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def main():
    audit=read(HERE/'analysis/audit.json')
    assert audit['complete'] and audit['archived_controls_unchanged'] and audit['cross_study_baseline_identical']
    old=HERE.parent/'adaptive_ablation/analysis'
    new_rows=read(HERE/'analysis/per_bank.json')
    old_rows=read(old/'per_bank.json')
    rows=[dict(r,origin='new' if r in new_rows else 'archived') for r in old_rows+new_rows]
    assert len(rows)==48 and len({(r['support'],r['mode'],r['seed']) for r in rows})==48
    by={(r['support'],r['mode'],r['seed']):r for r in rows}
    def vals(s,m,k):return [by[s,m,seed][k] for seed in SEEDS]
    def mean(s,m,k):return st.mean(vals(s,m,k))
    def cell(s,m,k):
        v=vals(s,m,k);avg=st.mean(v)
        if abs(avg)<.05:avg=0.
        return f'${avg:.1f} \\pm {st.stdev(v):.1f}$'
    save(HERE/'analysis/combined_per_bank.json',rows)
    with (HERE/'analysis/combined_per_bank.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[dict(support=s,mode=m,label=l,self_enabled=a,foreign_enabled=b,conservation=c,
                  **{k:dict(mean=mean(s,m,k),sd=st.stdev(vals(s,m,k)),seeds=vals(s,m,k)) for k in FIELDS})
             for s in ('native','lora') for m,l,a,b,c in MODES]
    save(HERE/'analysis/summary.json',summary)
    planned=[]
    for s in ('native','lora'):
        for c,left,right in ((0,'foreign_only','self_only'),(1,'foreign_conserve','self_conserve')):
            record=dict(support=s,conservation=c,seeds=SEEDS)
            for field,name,direction in [('accuracy_8','coexistence_foreign_minus_self',1),('accuracy_1','standalone_self_minus_foreign',-1)]:
                v=[direction*(a-b) for a,b in zip(vals(s,left,field),vals(s,right,field))]
                record[name]=dict(paired=v,mean=st.mean(v),sd=st.stdev(v),positive=sum(x>1e-9 for x in v))
            planned.append(record)
    save(HERE/'analysis/planned_contrasts.json',planned)
    lookup={(a,b,c):m for m,_,a,b,c in MODES}
    effects=[]
    for s in ('native','lora'):
        for i,term in enumerate(('self','foreign','conservation')):
            for bits,mode in lookup.items():
                if bits[i]:continue
                after=list(bits);after[i]=1;other=lookup[tuple(after)]
                for field in FIELDS[:4]:
                    v=[a-b for a,b in zip(vals(s,other,field),vals(s,mode,field))]
                    effects.append(dict(support=s,term=term,before=mode,after=other,field=field,paired=v,mean=st.mean(v),sd=st.stdev(v)))
    save(HERE/'analysis/conditional_effects.json',effects)
    task_rows=read(old/'per_task.json')+read(HERE/'analysis/per_task.json')
    task_effects=[]
    for s in ('native','lora'):
        for c,left,right in ((0,'foreign_only','self_only'),(1,'foreign_conserve','self_conserve')):
            for task in range(8):
                values={m:[100*next(r['accuracy'] for r in task_rows if r['support']==s and r['mode']==m and r['seed']==seed and r['task']==task and r['active_count']==8) for seed in SEEDS] for m in (left,right)}
                v=[a-b for a,b in zip(values[left],values[right])]
                task_effects.append(dict(support=s,conservation=c,task=task,paired=v,mean=st.mean(v)))
    save(HERE/'analysis/task_contrasts.json',task_effects)
    OUT.mkdir(parents=True,exist_ok=True)
    task_names=['Arithmetic','Pointer following','Boolean','Brackets','Sorting','Stack','Shortest path','Sudoku']
    task_lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r' & \multicolumn{2}{c}{Native} & \multicolumn{2}{c}{LoRA}\\',r'Task & C off & C on & C off & C on\\',r'\midrule']
    for t,name in enumerate(task_names):
        cells=[]
        for s in ('native','lora'):
            for c in (0,1):
                r=next(r for r in task_effects if r['support']==s and r['conservation']==c and r['task']==t)
                cells.append(f"${r['mean']:+.1f} \\pm {st.stdev(r['paired']):.1f}$")
        task_lines.append(name+' & '+' & '.join(cells)+r'\\')
    task_lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'task_contrasts_table.tex').write_text('\n'.join(task_lines)+'\n')
    lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r' & & \multicolumn{2}{c}{Native, 4 channels} & \multicolumn{2}{c}{LoRA, rank 1}\\',r'Conserve & Preserve & Accuracy & $\Delta$PPL & Accuracy & $\Delta$PPL\\',r'\midrule',r'--- & Frozen base & 29.1 & 0.0 & 29.1 & 0.0\\']
    for mode,label,a,b,c in MODES:
        if mode=='conservation_only':lines.append(r'\midrule')
        lines.append(('On' if c else 'Off')+' & '+label+' & '+' & '.join(cell(s,mode,k) for s in ('native','lora') for k in ('accuracy_8','wiki_change'))+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(OUT/'primary_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Support & Conservation & \multicolumn{2}{c}{All eight: foreign $-$ self} & \multicolumn{2}{c}{Owner: self $-$ foreign}\\',r' & & Mean $\pm$ SD & Seed changes & Mean $\pm$ SD & Seed changes\\',r'\midrule']
    for r in planned:
        pairs=[]
        for k in ('coexistence_foreign_minus_self','standalone_self_minus_foreign'):
            x=r[k];pairs += [f"${x['mean']:+.2f} \\pm {x['sd']:.2f}$",', '.join(f'{v:+.2f}' for v in x['paired'])]
        lines.append(('Native' if r['support']=='native' else 'LoRA')+' & '+('On' if r['conservation'] else 'Off')+' & '+' & '.join(pairs)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(OUT/'planned_contrasts_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{@{}lllrrrr@{}}',r'\toprule',r'Support & Conserve & Preserve & Owner & Calibrated & MC $\Delta$Acc & RL hours\\',r'\midrule']
    for s in ('native','lora'):
        for m,label,a,b,c in MODES:
            h=[x/3600 for x in vals(s,m,'training_seconds')]
            lines.append(('Native' if s=='native' else 'LoRA')+' & '+('On' if c else 'Off')+' & '+label+' & '+' & '.join(cell(s,m,k) for k in ('accuracy_1','calibrated_accuracy','mc_change'))+f' & ${st.mean(h):.2f} \\pm {st.stdev(h):.2f}$'+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}'];(OUT/'secondary_table.tex').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.3))
    labels=['Native, C off','Native, C on','LoRA, C off','LoRA, C on']
    for ax,key,title in zip(axes,('coexistence_foreign_minus_self','standalone_self_minus_foreign'),('(a) All-eight: foreign − self','(b) Owner-alone: self − foreign')):
        for j,(seed,color,offset) in enumerate(zip(SEEDS,('#2166AC','#D97706','#27866B'),(-.13,0,.13))):
            ax.scatter([r[key]['paired'][j] for r in planned],[i+offset for i in range(4)],s=18,color=color,label=f'Seed {seed}',zorder=3)
        ax.scatter([r[key]['mean'] for r in planned],range(4),marker='D',s=27,c='black',label='Mean',zorder=4)
        ax.axvline(0,color='#777777',ls='--',lw=1);ax.set(yticks=range(4),yticklabels=labels,title=title,xlabel='Accuracy difference (pp)');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].set_xlim(-3,10);axes[1].set_xlim(-1,3)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.13,1,1))
    fig.savefig(OUT/'component_contrasts.pdf');fig.savefig(OUT/'component_contrasts.png',dpi=190);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),sharey=True)
    for ax,s in zip(axes,('native','lora')):
        for c,color,marker in ((0,'#2166AC','s'),(1,'#D97706','o')):
            groups=[vals(s,m,'accuracy_8') for m,_,_,_,flag in MODES if flag==c]
            ax.errorbar(range(4),[st.mean(v) for v in groups],yerr=[st.stdev(v) for v in groups],marker=marker,color=color,capsize=3,label=f'Conservation {"on" if c else "off"}')
        ax.set(title='Native, four channels' if s=='native' else 'LoRA, rank one',xticks=range(4),xticklabels=['None','Self','Foreign','Both'],ylabel='All-eight accuracy (%)',xlabel='Specialist preservation',ylim=(43,65));ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False);fig.tight_layout(rect=(0,.1,1,1))
    fig.savefig(OUT/'factorial_accuracy.pdf');fig.savefig(OUT/'factorial_accuracy.png',dpi=190);plt.close(fig)
    lines=['# Preservation components: audited results','', 'Each comparison uses the same three seeds. The 24 archived controls are counted once.', '', '| Method | Conservation | Foreign − self, all-eight accuracy (pp) | Self − foreign, owner accuracy (pp) |','| --- | --- | ---: | ---: |']
    for r in planned:
        lines.append(f"| {r['support']} | {'on' if r['conservation'] else 'off'} | {r['coexistence_foreign_minus_self']['mean']:+.2f} | {r['standalone_self_minus_foreign']['mean']:+.2f} |")
    lines += ['', 'Foreign preservation has higher mean coexistence accuracy in all four comparisons and wins 11 of the 12 paired seed contrasts. These contrasts share seeds and are not 12 independent replications.', '', 'The standalone self-preservation advantage is small: +0.28 / 0.00 pp native and +0.62 / +0.38 pp LoRA (conservation off / on). This offers weak evidence for that part of the recorded hypothesis.', '', 'Foreign+conservation gives native 56.15% and LoRA 58.09% accuracy, with WikiText perplexity increases of 4.22% and 1.68%. Adding self preservation changes accuracy by +0.17 pp native and −0.63 pp LoRA. Do not declare an optimal objective from three seeds.', '', 'Saved-record audit only: no independent GPU-forward or optimizer rerun. No new inference about broad knowledge preservation, compute-matched superiority, or necessity of RL over replay-only distillation.']
    assert sum(r['coexistence_foreign_minus_self']['positive'] for r in planned)==11
    (HERE/'analysis/results.md').write_text('\n'.join(lines)+'\n')
    inputs=[HERE/'analysis/audit.json',HERE/'analysis/per_bank.json',old/'per_bank.json',HERE/'analyse.py',Path(__file__),HERE/'results/casr-preservation-components-results-20261102.tar.gz']
    save(OUT/'provenance.json',dict(audit=audit,inputs={str(p.relative_to(HERE.parents[1])):sha(p) for p in inputs},outputs={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.tex','.pdf','.png')}))
    print((OUT/'primary_table.tex').read_text());print(json.dumps(planned,indent=2))


if __name__=='__main__':main()
