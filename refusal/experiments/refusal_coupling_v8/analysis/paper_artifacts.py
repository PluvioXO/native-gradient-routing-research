"""Export the completed V8 tables and immutable summary figure for the paper."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[2]/"figures/coupling"
LABELS={"clean":"Clean","global":"Global coupling","local":"Local coupling","lookahead":"Local + lookahead"}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    primary=pd.read_csv(HERE/"primary_comparison.csv")
    benign=pd.read_csv(HERE/"benign_comparison.csv")
    assert len(primary)==8 and primary.task_success.all()
    rows=[r"\begin{tabular}{llrrrrc}",r"\toprule",
          r"Start & Defender & Allowed & Blocked & Refusal & General $\Delta$ & Full pass\\",r"\midrule"]
    for r in primary.itertuples():
        rows.append(f"{r.parent[-1]} ch. & {LABELS[r.arm]} & {100*r.allowed_accuracy:.2f} & "
            f"{100*r.blocked_accuracy:.2f} & {100*r.refusal:.1f} & {r.general_change_pp:+.2f} & "
            +("Yes" if r.success else "No")+r"\\")
    rows += [r"\bottomrule",r"\end{tabular}"]
    (OUT/"lowrate_table.tex").write_text("\n".join(rows)+"\n")
    rows=[r"\begin{tabular}{lrr}",r"\toprule",r"Defender & 4-channel start & 8-channel start\\",r"\midrule"]
    b=benign[benign.endpoint=="final"]
    for arm,label in LABELS.items():
        values=[float(b[(b.arm==arm)&(b.parent==parent)].iloc[0].blocked_refusal) for parent in ["shared4","shared8"]]
        rows.append(f"{label} & {100*values[0]:.2f} & {100*values[1]:.2f}"+r"\\")
    rows += [r"\bottomrule",r"\end{tabular}"]
    (OUT/"benign_table.tex").write_text("\n".join(rows)+"\n")
    shutil.copy2(HERE/"summary_graphs.pdf",OUT/"summary_graphs.pdf")
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    provenance=dict(inputs={n:sha(HERE/n) for n in ["primary_comparison.csv","benign_comparison.csv","summary_graphs.pdf","audit.json"]},
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!="provenance.json"},
        script_sha256=sha(Path(__file__)))
    (OUT/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print("Saved V8 paper tables and figure",OUT)


if __name__=="__main__":
    main()
