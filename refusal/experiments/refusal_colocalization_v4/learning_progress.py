"""Describe end-of-budget learning without extrapolating an unrun continuation."""
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;OUT=HERE/'analysis'
COLORS={'native1':'#eab308','native2':'#d97706','native4':'#2563eb','native8':'#0f766e','full':'#7c3aed'}


def main():
    audit=json.loads((OUT/'audit.json').read_text());assert audit['checks_passed']
    cv=pd.read_csv(OUT/'development.csv');loss=pd.read_csv(OUT/'losses.csv');phase=pd.read_csv(OUT/'phases.csv')
    cv['worst_quality']=[ast.literal_eval(s)[1] for s in cv['rank']]
    rows=[]
    for p in phase.itertuples():
        s=cv[(cv.job==p.job)&(cv.stage==p.stage)].sort_values('step');h=loss[(loss.job==p.job)&(loss.stage==p.stage)].sort_values('step')
        assert h.step.tolist()==list(range(1,p.steps+1)) and s.step.iloc[-1]==p.steps
        window=min(256,p.steps//4);prior=s[(s.step>p.steps-2*window)&(s.step<=p.steps-window)];last=s[s.step>p.steps-window]
        assert len(last)==len(prior)>0
        row=dict(job=p.job,condition=s.condition.iloc[0],stage=p.stage,steps=p.steps,best_step=p.best_step,qualified=p.qualified,window_updates=window,previous_evaluations=len(prior),last_evaluations=len(last),all_feasible_checkpoints=int(s.feasible.sum()))
        for k in ['path_correct','min_depth_accuracy','min_case_accuracy','worst_quality','allowed_correct','mean_blocked_refusal','false_refusal','min_allowed_accuracy','blocked_refusal','max_false_refusal','joint_score','general']:
            if s[k].isna().all():continue
            row[k+'_previous_mean']=prior[k].mean();row[k+'_last_mean']=last[k].mean();row[k+'_change_pp']=100*(last[k].mean()-prior[k].mean());row[k+'_final']=s[k].iloc[-1]
        for k in ['capability_nll','positive_policy','negative_policy','refusal_grammar','preserve_full_kl','conserve_full_kl']:
            if h[k].isna().all():continue
            before=h[(h.step>p.steps-2*window)&(h.step<=p.steps-window)][k].mean();after=h[h.step>p.steps-window][k].mean()
            row[k+'_previous_mean']=before;row[k+'_last_mean']=after;row[k+'_relative_change_percent']=100*(after/before-1) if before else np.nan
        rows.append(row)
    summary=pd.DataFrame(rows);assert len(summary)==19
    summary.to_csv(OUT/'learning_progress.csv',index=False)
    cases=pd.read_csv(OUT/'case_development.csv')
    assert len(cases)==9*17*26
    exposure=json.loads((OUT/'exposure.json').read_text())
    assert exposure['positive_refusal_examples_by_position']=={'1':1450,'2':427,'3':171}
    assert sum(exposure['positive_refusal_examples_by_case'].values())==2048
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    with PdfPages(OUT/'learning_progress.pdf') as pdf:
        page=0
        def save(fig,title,note):
            nonlocal page
            page+=1;fig.suptitle(title,fontsize=16,fontweight='bold',y=.98);fig.text(.02,.015,note,fontsize=9,va='bottom');fig.tight_layout(rect=(0,.075,1,.94));pdf.savefig(fig);fig.savefig(OUT/f'learning_page_{page:02d}.png',dpi=110);plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(14,5))
        for c in ['native1','native2','native4','native8','full']:
            job='cap_'+c+'_s31';s=cv[(cv.job==job)&(cv.stage=='hard')&(cv.step>=1024)]
            for ax,col,title in zip(axs[:2],['path_correct','min_depth_accuracy'],['Overall development traces','Worst-depth development traces']):ax.plot(s.step,100*s[col],color=COLORS[c],label=c);ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel('%')
            h=loss[(loss.job==job)&(loss.stage=='hard')];h=h.assign(smooth=h.capability_nll.rolling(128,min_periods=1).mean());h=h[h.step>=1024];axs[2].plot(h.step,h.smooth,color=COLORS[c],label=c)
        axs[1].axhline(80,color='grey',ls='--');axs[0].legend();axs[2].set_title('Capability NLL (128-update average)');axs[2].set_ylabel('NLL')
        for ax in axs:ax.set_xlabel('Hard-stage updates')
        save(fig,'Extra acquisition training: two channels are still making progress','Overall accuracy and training loss can improve while the hardest depth stalls. The dashed 80% line is only one qualification criterion.\nDo not infer an absolute channel-capacity limit from this one-seed, fixed-step run.')
        factors=[.1,.3,1.];factor_colors=['#2563eb','#d97706','#7c3aed'];conditions=['shared4','shared8','full']
        fig,axs=plt.subplots(3,3,figsize=(14,11))
        for i,c in enumerate(conditions):
            for factor,color in zip(factors,factor_colors):
                job=f'policy_{c}_s31_lr{round(factor*100):03d}';s=cv[(cv.job==job)&(cv.stage=='policy')]
                for ax,col in zip(axs[i],['min_allowed_accuracy','blocked_refusal','max_false_refusal']):ax.plot(s.step,100*s[col],color=color,label=f'LR factor {factor:g}');ax.set_ylim(-2,104);ax.set_xlabel('Updates');ax.set_ylabel('%')
            for ax,threshold,title in zip(axs[i],[80,90,10],['Worst allowed accuracy','Worst blocked refusal','Worst false refusal']):ax.axhline(threshold,color='grey',ls='--');ax.set_title(c+': '+title)
        axs[0,0].legend(fontsize=9)
        save(fig,'All nine policy learning curves: continuing loss reduction is not joint success','All models received 1,024 updates. Thresholds must hold simultaneously in every case, plus collateral-retention gates.\nEight channels at LR factor 0.1 finish at their best ranked checkpoint, but still miss the required margins by a large amount.')
        fig,axs=plt.subplots(3,3,figsize=(14,11));fields=['positive_policy','negative_policy','capability_nll','preserve_full_kl'];colors=['#dc2626','#2563eb','#0f766e','#d97706']
        for i,c in enumerate(conditions):
            for j,factor in enumerate(factors):
                job=f'policy_{c}_s31_lr{round(factor*100):03d}';s=loss[(loss.job==job)&(loss.stage=='policy')]
                for col,color in zip(fields,colors):axs[i,j].plot(s.step,s[col].rolling(64,min_periods=1).mean(),label=col,color=color)
                axs[i,j].set_title(c+f', LR factor {factor:g}');axs[i,j].set_xlabel('Updates');axs[i,j].set_ylabel('Unweighted loss');axs[i,j].set_ylim(bottom=0)
        axs[0,0].legend(fontsize=8)
        save(fig,'Losses remain learnable, but some candidates oscillate between decisions','64-update averages of raw, unweighted losses. Plotting them together does not imply equal coefficients in the training objective.\nTraining loss alone cannot establish that more epochs would fix development selectivity, wording generalization or retention.')
        fig,axs=plt.subplots(3,3,figsize=(14,10));position_colors=['#dc2626','#2563eb','#0f766e']
        for i,c in enumerate(conditions):
            job=f'policy_{c}_s31_lr010';selected=phase[(phase.job==job)&(phase.stage=='policy')].iloc[0].best_step
            for j,depth in enumerate([2,3,4]):
                ax=axs[i,j]
                for position in range(1,depth):
                    s=cases[(cases.job==job)&(cases['case']==f'h{depth}_blocked_{position}')]
                    ax.plot(s.step,100*s.refusal,label=f'Marked at hop {position}',color=position_colors[position-1])
                ax.axhline(90,color='grey',ls='--');ax.axvline(selected,color='grey',ls=':');ax.set_ylim(-3,104);ax.set_title(f'{c}: depth {depth}');ax.set_xlabel('Policy updates');ax.set_ylabel('Correct refusal (%)');ax.legend(fontsize=8)
        save(fig,'Refusal often learns the first-hop condition without later-hop membership','Shown: all checkpoints at the chosen LR factor 0.1. Dashed horizontal line: 90% target. Dotted vertical: selected checkpoint.\nThe final full model is nearly perfect at first-hop refusal but misses later blocked positions; extra updates do not repair all cases together.')
    print(summary[summary.stage!='basic'][['job','stage','best_step','path_correct_change_pp','worst_quality_change_pp','capability_nll_relative_change_percent']].round(3).to_string(index=False))
    print(json.dumps(exposure,indent=2))


if __name__=='__main__':main()
