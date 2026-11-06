"""Generate the standalone refusal paper's figures and tables from saved results."""
import hashlib
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    here = Path(__file__).resolve().parent
    out = here.parents[1] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    comparison = here / "results/casr-refusal-position-20261029/comparison.json"
    summary = here / "position_analysis/run_summary.json"
    audit = json.loads((here / "position_analysis/audit.json").read_text())
    assert audit["runs"] == 32 and audit["evaluations"] == 288 and audit["metrics_recomputed"]
    records = json.loads(comparison.read_text())
    summaries = json.loads(summary.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    styles = {"upstream": ("#1565C0", "Two blocks before", "-"),
              "downstream": ("#D36B18", "Two blocks after", "--"),
              "all": ("#777777", "Unrestricted reference", ":")}
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.8), sharex=True, sharey=True)
    for i, lr in enumerate((1e-6, 1e-5)):
        for j, layer in enumerate((3, 12, 22)):
            ax = axes[i, j]
            for scope, (color, label, linestyle) in styles.items():
                selected = [x for x in records if x["scope"] == scope and x["lr"] == lr
                            and x["source"].startswith(f"layer{layer:02d}_") and x["panel"] == "test"]
                curves = np.array([[x["blocked_refusal"] for x in sorted(selected, key=lambda x:x["step"])
                                    if x["source"] == source] for source in sorted({x["source"] for x in selected})]) * 100
                assert curves.shape == (2 if layer == 12 else 3, 9)
                ax.plot(range(9), curves.mean(0), color=color, label=label, ls=linestyle)
                ax.fill_between(range(9), curves.min(0), curves.max(0), color=color, alpha=.12)
            ax.set(title=f"Layer {layer}, LR {lr:g}", ylim=(-3,103), xticks=[0,2,4,6,8])
            if i == 1:
                ax.set_xlabel("Removal updates")
            if j == 0:
                ax.set_ylabel("Refusal (%)")
            ax.grid(alpha=.15)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0,.08,1,1))
    fig.savefig(out / "position_curves.pdf")
    fig.savefig(out / "position_curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.9))
    for panel, label, color in (("test", "Original wording", "#1565C0"),
                                ("transfer", "Alternate wording", "#D36B18")):
        curve = sorted([x for x in records if x["source"] == "layer22_s31" and x["scope"] == "upstream"
                        and x["lr"] == 1e-6 and x["panel"] == panel], key=lambda x:x["step"])
        axes[0].plot(range(9), [100*x["blocked_refusal"] for x in curve], label=label, color=color, marker=".")
    axes[0].set(title="(a) Wording sensitivity; before, LR 1e-6", ylabel="Refusal (%)")
    for seed, color in ((31,"#1565C0"),(47,"#3C8D5A"),(71,"#D36B18")):
        curve = sorted([x for x in records if x["source"] == f"layer22_s{seed}" and x["scope"] == "downstream"
                        and x["lr"] == 1e-5 and x["panel"] == "test"], key=lambda x:x["step"])
        axes[1].plot(range(9), [100*x["blocked_correct"] for x in curve], label=f"Seed {seed}", color=color, marker=".")
    axes[1].set(title="(b) Answer recovery; after, LR 1e-5", ylabel="Correct answers (%)")
    for ax in axes:
        ax.set(xlabel="Removal updates", ylim=(-3,103), xticks=[0,2,4,6,8])
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out / "failure_modes.pdf")
    fig.savefig(out / "failure_modes.png", dpi=170)
    plt.close(fig)
    lines = [r"\begin{tabular}{llcc}", r"\toprule", r"Layer & LR & Before & After \\", r"\midrule"]
    for layer in (3,12,22):
        for lr in (1e-6,1e-5):
            cells = []
            for scope in ("upstream", "downstream"):
                group = sorted([x for x in summaries if x["source"].startswith(f"layer{layer:02d}_")
                                and x["lr"] == lr and x["scope"] == scope], key=lambda x:x["source"])
                cells.append(", ".join(str(x["sustained_zero"]) if x["sustained_zero"] is not None else "$>8$" for x in group))
            lines.append(f"{layer} & $10^{{{int(np.log10(lr))}}}$ & {cells[0]} & {cells[1]} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "position_table.tex").write_text("\n".join(lines)+"\n")
    parent = here / "results/casr-layer-measurements-20261028.tar.gz"
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == json.loads((here / "parent_audit.json").read_text())["archive_sha256"]
    acquisitions, removal = {}, {}
    with tarfile.open(parent) as archive:
        for member in archive.getmembers():
            if member.name.startswith("runs/") and member.name.endswith("/acquisition.json"):
                acquisitions[member.name.split('/')[1]] = json.load(archive.extractfile(member))
            if member.name.startswith("runs/") and "/removal_lr_" in member.name and "/eval_" in member.name:
                value = json.load(archive.extractfile(member))
                key = (member.name.split('/')[1], value["lr"])
                removal.setdefault(key, []).append(value)
    recovery = {f"{source}_{lr:g}": next((v["step"] for v in sorted(curve,key=lambda x:x["step"])
                 if v["panels"]["test"]["metrics"]["blocked_correct"] >= .9), None)
                for (source,lr),curve in removal.items()}
    lines = [r"\begin{tabular}{lrrcc}", r"\toprule", r"Acquisition & Params & Pass & $10^{-6}$ & $10^{-5}$ \\", r"\midrule"]
    for prefix, label, params in (("layer03", "Layer 3", "10,752"),("layer12", "Layer 12", "10,752"),
                                  ("layer22", "Layer 22", "10,752"),("full", "Full model", "494M")):
        names = sorted(k for k,v in acquisitions.items() if k.startswith(prefix) and v["qualified"])
        cells = [", ".join(str(recovery[f"{n}_{lr:g}"]) for n in names) for lr in (1e-6,1e-5)]
        lines.append(f"{label} & {params} & {len(names)}/3 & {cells[0]} & {cells[1]} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "unrestricted_table.tex").write_text("\n".join(lines)+"\n")
    (out / "acquisition_records.json").write_text(json.dumps(acquisitions,indent=2))
    inputs = [comparison, summary, parent, here / "position_analysis/audit.json", Path(__file__)]
    outputs = sorted(p for p in out.iterdir() if p.suffix in (".tex", ".pdf", ".png"))
    provenance = {"inputs": {str(p.relative_to(here.parents[1])): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
                  "evidence": "Completed eight-update position study and completed parent layer study; 64-update extension excluded"}
    (out / "provenance.json").write_text(json.dumps(provenance,indent=2))
    print(out)


if __name__ == "__main__":
    main()
