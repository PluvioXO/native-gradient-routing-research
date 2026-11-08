"""Descriptive paired outcomes; no test-driven checkpoint selection."""
import argparse
from pathlib import Path
import pandas as pd
from study import check,read,write,sha


def main(root):
    m=check(root);selection=read(root/"selection.json")
    assert (root/"COMPLETE.json").exists()
    out=root.parent/"analysis";out.mkdir(exist_ok=True)
    endpoints=[];curves=[];history=[];primary=[];benign=[]
    last=max([read(root/"attacks"/k/"DONE.json")["ended_unix"] for k in selection["attacks"]],default=0)
    for key,spec in selection["attacks"].items():
        folder=root/"attacks"/key;updates=read(folder/"history.json")
        assert len(updates)==m["config"]["removal_updates"]+m["config"]["repair_updates"]
        assert (folder/"TEST_DONE.json").exists()
        history.extend(dict(job=key,**spec,**r) for r in updates)
        for phase in ["remove","repair"]:
            for file in sorted(folder.glob(phase+"_*.json")):
                if not file.stem.removeprefix(phase+"_").isdigit():continue
                r=read(file);curves.append(dict(job=key,**spec,phase=phase,
                    cumulative_updates=r["cumulative_updates"],**r["selection"]))
        reports={}
        for file in sorted(folder.glob("test_*.json")):
            r=read(file);assert r["started_unix"]>=last
            reports[r["endpoint"]]=r
            endpoints.append(dict(job=key,**spec,endpoint=r["endpoint"],phase=r["phase"],
                cumulative_updates=r["cumulative_updates"],**r["selection"]))
        if spec["objective"]=="benign":
            benign.append(dict(job=key,**spec,**reports["final"]["selection"]))
        else:
            r=reports.get("first_full")
            primary.append(dict(job=key,**spec,dev_full_gate_found=r is not None,
                first_dev_full_update=r["cumulative_updates"] if r else None,
                heldout_full_success=bool(r and r["selection"]["success"]),
                heldout_task_success=bool(r and r["selection"]["task_success"])))
    pair_rows=[dict(pair=key,**v) for key,v in selection["pairs"].items()]
    for name,rows in [("pairs.csv",pair_rows),("endpoints.csv",endpoints),("learning_curves.csv",curves),
                      ("training.csv",history),("primary_attack.csv",primary),("primary_benign.csv",benign)]:
        pd.DataFrame(rows).to_csv(out/name,index=False)
    comparisons=[]
    if benign:
        b=pd.DataFrame(benign)
        for (parent,seed),d in b.groupby(["parent","seed"]):
            arms=d.set_index("arm");assert set(arms.index)=={"clean","coupled"}
            comparisons.append(dict(parent=parent,seed=int(seed),
                benign_refusal_difference_pp=100*(arms.loc["coupled","blocked_refusal"]-arms.loc["clean","blocked_refusal"]),
                allowed_difference_pp=100*(arms.loc["coupled","allowed_accuracy"]-arms.loc["clean","allowed_accuracy"]),
                general_difference_pp=100*(arms.loc["coupled","general_accuracy"]-arms.loc["clean","general_accuracy"])))
    pd.DataFrame(comparisons).to_csv(out/"paired_contrasts.csv",index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out/"comparison_graphs.pdf") as pdf:
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,parent in zip(axes,m["config"]["parents"]):
            if benign:
                d=pd.DataFrame(benign)
                for seed in m["config"]["seeds"]:
                    pair=d[(d.parent==parent)&(d.seed==seed)].set_index("arm")
                    if len(pair)==2:ax.plot([0,1],100*pair.loc[["clean","coupled"],"blocked_refusal"],"o-",label=str(seed))
            ax.set_title(parent+" / benign update 256");ax.set_xticks([0,1],["Clean full model","Coupled full model"])
            ax.set_ylim(-3,103);ax.set_ylabel("Blocked refusal (%)");ax.legend(title="Continuation seed")
        fig.suptitle("Matched global continuation: benign stability (conditional on fixed starts)")
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,parent in zip(axes,m["config"]["parents"]):
            if primary:
                d=pd.DataFrame(primary)
                for seed in m["config"]["seeds"]:
                    pair=d[(d.parent==parent)&(d.seed==seed)].set_index("arm")
                    if len(pair)!=2:continue
                    y=[pair.loc[a,"first_dev_full_update"] if pair.loc[a,"dev_full_gate_found"] else 272
                       for a in ["clean","coupled"]]
                    ax.plot([0,1],y,"o-",label=str(seed))
            ax.axhline(256,color="gray",ls=":")
            ax.set_title(parent+" / preserving removal");ax.set_xticks([0,1],["Clean full model","Coupled full model"])
            ax.set_ylim(-5,285);ax.set_ylabel("First DEV full-gate update; 272 = censored")
        fig.suptitle("Development stopping times; held-out validation is reported separately")
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    write(out/"summary.json",dict(planned_defenders=len(m["defense_jobs"]),
        qualified_pairs=sum(p["qualified"] for p in selection["pairs"].values()),
        downstream_fits=len(selection["attacks"]),heldout_endpoints=len(endpoints),
        paired_comparisons=comparisons,conditional_on_fixed_starts=True,
        source_hashes=m["sources"],heldout_after_all_training=True))
    write(out/"provenance.json",dict(files={p.name:sha(p) for p in out.iterdir()
        if p.is_file() and p.name!="provenance.json"},manifest_sha256=sha(root/"manifest.json")))
    print("Final analysis saved",out,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    main(p.parse_args().root.resolve())
