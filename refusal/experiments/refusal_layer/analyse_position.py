"""Audit saved responses and plot the completed eight-update position study."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def main():
    if not (Path(__file__).resolve().parent/'results').exists():raise SystemExit('Raw measurements were removed during the 5 November cleanup. See refusal/RAW_DATA.md to restore them; analysed tables and plots remain in this study.')
    here = Path(__file__).resolve().parent
    root = here / "results/casr-refusal-position-20261029"
    out = here / "position_analysis"
    out.mkdir(exist_ok=True)
    # Use the archived scorer without importing the GPU training dependencies.
    tree = ast.parse((root / "source/removal.py").read_text())
    scoring = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name in ("classify", "metrics")], type_ignores=[])
    namespace = dict(re=re, np=np)
    exec(compile(scoring, "archived_scoring", "exec"), namespace)
    with tarfile.open(here / "results/casr-layer-measurements-20261028.tar.gz") as archive:
        baselines = {name.split('/')[1]: json.load(archive.extractfile(name))
                     for name in archive.getnames() if re.fullmatch(r"runs/layer\d+_s\d+/baseline.json", name)}
    records, summaries, losses = [], [], []
    evaluations = responses = 0
    for run in sorted((root / "runs").iterdir()):
        manifest = json.loads((run / "manifest.json").read_text())
        complete = json.loads((run / "complete.json").read_text())
        assert complete["final_step"] == 8 and complete["frozen_parameters_unchanged"]
        assert complete["fingerprint"] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        assert manifest["runner_sha256"] == hashlib.sha256((root / "source/explore_bypass.py").read_bytes()).hexdigest()
        paths = sorted(run.glob("eval_*.json"))
        assert len(paths) == 9
        job = manifest["job"]
        initial = None
        for step, path in enumerate(paths):
            value = json.loads(path.read_text())
            assert value["step"] == step and value["examples_seen"] == step * 4
            assert value["epochs"] == step / 128
            panels = value["panels"]
            if initial is None:
                initial = panels
            for panel, report in panels.items():
                rows = report["rows"]
                assert [x["id"] for x in rows] == [x["id"] for x in initial[panel]["rows"]]
                assert [x["prompt"] for x in rows] == [x["prompt"] for x in initial[panel]["rows"]]
                assert len(rows) == dict(test=256, transfer=128, utility=32, general=64)[panel]
                if panel == "general":
                    for row in rows:
                        assert row["predicted"] == int(np.argmax(row["choice_logprobs"]))
                        assert row["correct"] == (row["predicted"] == row["gold"])
                        assert math.isclose(row["gold_nll"], -row["choice_logprobs"][row["gold"]], abs_tol=1e-6)
                    measures = {task + "_" + key: float(np.mean([row[key] for row in rows if row["task"] == task]))
                                for task in {row["task"] for row in rows}
                                for key in ("correct", "gold_nll", "choice_mass")}
                else:
                    for row in rows:
                        scored = namespace["classify"](row["response"], row["answer"])
                        assert all(row[k] == v for k, v in scored.items()), (run.name, step, row["id"])
                    measures = namespace["metrics"](rows)
                assert measures.keys() == report["metrics"].keys()
                assert all(math.isclose(v, report["metrics"][k], abs_tol=1e-8) for k, v in measures.items())
                responses += len(rows)
                records.append(dict(source=job["source"], scope=job["scope"], lr=job["lr"], step=step, panel=panel, **measures))
            evaluations += 1
        curve = [x for x in records if x["source"] == job["source"] and x["scope"] == job["scope"]
                 and x["lr"] == job["lr"] and x["panel"] == "test"]
        def first(predicate, stable=False):
            return next((x["step"] for i, x in enumerate(curve) if predicate(x)
                         and (not stable or all(predicate(y) for y in curve[i:]))), None)
        summary = dict(job, first_zero=first(lambda x: x["blocked_refusal"] == 0),
                       sustained_zero=first(lambda x: x["blocked_refusal"] == 0, True),
                       first_90_correct=first(lambda x: x["blocked_correct"] >= .9),
                       sustained_90_correct=first(lambda x: x["blocked_correct"] >= .9, True))
        for panel, key in (("utility", "utility_correct"), ("general", "arc_easy_correct"),
                           ("general", "winogrande_correct")):
            final = panels[panel]["metrics"][key]
            summary[key] = dict(before=initial[panel]["metrics"][key], after=final,
                change_from_before=final-initial[panel]["metrics"][key],
                change_from_base=final-baselines[job["source"]][panel]["metrics"][key])
        summaries.append(summary)
        losses.extend(dict(source=job["source"], scope=job["scope"], lr=job["lr"], **row)
                      for row in json.loads((run / "training.json").read_text()))
    assert evaluations == 288 and len(summaries) == 32
    published = json.loads((root / "comparison.json").read_text())
    identity = lambda x: (x["source"], x["scope"], x["lr"], x["step"], x["panel"])
    assert sorted(records, key=identity) == sorted([x for x in published if x["scope"] != "all"], key=identity)
    audit = dict(runs=32, evaluations=evaluations, rescored_rows=responses,
                 frozen_checks_passed=32, metrics_recomputed=True,
                 limitation="32 examples per general benchmark; transfer reuses test codes with wording excluded from SFT but used in acquisition validation")
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    (out / "run_summary.json").write_text(json.dumps(summaries, indent=2))
    # Original unrestricted curves are reference data, audited in the parent study.
    reference = [x for x in published if x["scope"] == "all"]
    all_records = records + reference
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    styles = {"upstream": ("#1565C0", "Before frozen layer"),
              "downstream": ("#E07821", "After frozen layer"), "all": ("#777777", "Unrestricted reference")}
    with PdfPages(out / "position_results.pdf") as pdf:
        for panel, key, title in (("test", "blocked_refusal", "Refusal on the trained prompt forms"),
                                  ("test", "blocked_correct", "Correct answers on the refused category"),
                                  ("transfer", "blocked_refusal", "Refusal on wording excluded from SFT (used in acquisition validation)")):
            fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3), sharex=True, sharey=True)
            for row, lr in enumerate((1e-6, 1e-5)):
                for col, layer in enumerate((3, 12, 22)):
                    ax = axes[row, col]
                    for scope, (color, label) in styles.items():
                        selected = [x for x in all_records if x["scope"] == scope and x["lr"] == lr
                                    and x["source"].startswith(f"layer{layer:02d}_") and x["panel"] == panel]
                        values = np.array([[x[key] for x in sorted(selected, key=lambda x:x["step"]) if x["source"] == source]
                                           for source in sorted({x["source"] for x in selected})]) * 100
                        ax.plot(range(9), values.mean(0), color=color, label=label, ls="--" if scope == "all" else "-")
                        ax.fill_between(range(9), values.min(0), values.max(0), color=color, alpha=.12)
                    ax.set_title(f"Layer {layer}; learning rate {lr:g}")
                    ax.set(ylim=(-3, 103), xticks=range(9), xlabel="Removal updates", ylabel="Percent")
                    ax.grid(alpha=.15)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3)
            fig.suptitle(title + "\nMeans and seed ranges; early/late n=3, middle n=2", y=.98)
            fig.tight_layout(rect=(0, .055, 1, .92))
            pdf.savefig(fig)
            plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        for row, lr in enumerate((1e-6, 1e-5)):
            for col, (panel, key, title) in enumerate((("utility", "utility_correct", "Arithmetic (32 items)"),
                    ("general", "arc_easy_correct", "ARC-Easy (32 items)"),
                    ("general", "winogrande_correct", "WinoGrande (32 items)"))):
                ax = axes[row, col]
                for scope, (color, label) in list(styles.items())[:2]:
                    for layer, marker in ((3, "o"), (12, "s"), (22, "^")):
                        group = [x for x in summaries if x["scope"] == scope and x["lr"] == lr
                                 and x["source"].startswith(f"layer{layer:02d}_")]
                        vals = [x[key]["change_from_before"] * 100 for x in group]
                        pos = layer + (-.6 if scope == "upstream" else .6)
                        ax.scatter([pos]*len(vals), vals, color=color, marker=marker, alpha=.6)
                        ax.plot([pos-.4, pos+.4], [np.mean(vals)]*2, color=color, lw=3)
                ax.axhline(0, color="gray", lw=1)
                ax.set(title=f"{title}; LR {lr:g}", xlabel="Refusal-trained layer", ylabel="Change from pre-removal (pp)", xticks=[3,12,22])
                ax.grid(alpha=.15)
        fig.suptitle("Control performance after eight updates\nDots: individual seeds; bars: means. Blue: before; orange: after.")
        fig.tight_layout(rect=(0, 0, 1, .92))
        pdf.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
        for row, lr in enumerate((1e-6, 1e-5)):
            for col, layer in enumerate((3,12,22)):
                ax = axes[row,col]
                for scope, (color,label) in list(styles.items())[:2]:
                    selected = [x for x in losses if x["scope"]==scope and x["lr"]==lr and x["source"].startswith(f"layer{layer:02d}_")]
                    for source in sorted({x["source"] for x in selected}):
                        curve = sorted([x for x in selected if x["source"]==source],key=lambda x:x["step"])
                        ax.plot([x["step"] for x in curve],[x["loss"] for x in curve],color=color,alpha=.6)
                ax.set(title=f"Layer {layer}; LR {lr:g}", xlabel="Removal update", ylabel="Training loss")
                ax.grid(alpha=.15)
        fig.suptitle("Training loss for each seed\nBlue: before frozen layer; orange: after frozen layer")
        fig.tight_layout(rect=(0,0,1,.92))
        pdf.savefig(fig)
        plt.close(fig)
    print(json.dumps(audit))
    print(out / "position_results.pdf")


if __name__ == "__main__":
    main()
