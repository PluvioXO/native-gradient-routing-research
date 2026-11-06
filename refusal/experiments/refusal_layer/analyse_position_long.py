"""Audit saved responses and plot the completed 64-update position extension."""
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
    root = here / "results/casr-refusal-position-long-20261029"
    out = here / "position_long_analysis"
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
        assert complete["final_step"] == 64 and complete["frozen_parameters_unchanged"]
        assert complete["fingerprint"] == hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        assert manifest["runner_sha256"] == hashlib.sha256((root / "source/explore_bypass.py").read_bytes()).hexdigest()
        paths = sorted(run.glob("eval_*.json"))
        assert len(paths) == 65
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
    assert evaluations == 390 and len(summaries) == 6
    published = json.loads((root / "comparison.json").read_text())
    identity = lambda x: (x["source"], x["scope"], x["lr"], x["step"], x["panel"])
    assert sorted(records, key=identity) == sorted([x for x in published if x["scope"] != "all"], key=identity)
    short_root = here / "results/casr-refusal-position-20261029/runs"
    for run in sorted((root / "runs").iterdir()):
        for path in sorted(run.glob("eval_*.json"))[:9]:
            original = json.loads((short_root / run.name / path.name).read_text())
            current = json.loads(path.read_text())
            for panel in current["panels"]:
                assert current["panels"][panel]["metrics"] == original["panels"][panel]["metrics"]
    audit = dict(runs=6, evaluations=evaluations, rescored_rows=responses,
                 frozen_checks_passed=6, metrics_recomputed=True,
                 limitation="32 examples per general benchmark; transfer reuses test codes with wording excluded from SFT but used in acquisition validation")
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    (out / "run_summary.json").write_text(json.dumps(summaries, indent=2))
    (out / "records.json").write_text(json.dumps(records, indent=2))
    figure_dir = here.parents[1] / "figures"
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharex=True, sharey=True)
    for ax, key, title in zip(axes, ("blocked_refusal", "blocked_correct"),
                              ("(a) Original-wording refusal", "(b) Answer recovery")):
        for scope, color, label in (("upstream", "#1565C0", "Two blocks before"),
                                    ("downstream", "#D36B18", "Two blocks after")):
            curves = np.array([[x[key] * 100 for x in records if x["source"] == f"layer22_s{seed}"
                               and x["scope"] == scope and x["panel"] == "test"] for seed in (31,47,71)])
            assert curves.shape == (3, 65)
            ax.plot(range(65), curves.mean(0), color=color, label=label)
            ax.fill_between(range(65), curves.min(0), curves.max(0), color=color, alpha=.15)
        ax.axvline(8, color="gray", ls=":", lw=1)
        ax.set(title=title, xlabel="Removal updates", ylabel="Percent", ylim=(-3,103))
        ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0,.10,1,1))
    fig.savefig(figure_dir / "position_long_curves.pdf")
    fig.savefig(figure_dir / "position_long_curves.png", dpi=170)
    plt.close(fig)
    lines = [r"\begin{tabular}{llrr}", r"\toprule", r"Seed & Updated blocks & Zero refusal & 90\% correct \\", r"\midrule"]
    for item in sorted(summaries, key=lambda x:(x["source"],x["scope"])):
        seed=item["source"].split("s")[-1]
        scope="Before" if item["scope"]=="upstream" else "After"
        lines.append(f"{seed} & {scope} & {item['first_zero']} & {item['first_90_correct']} " + r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (figure_dir / "position_long_table.tex").write_text("\n".join(lines)+"\n")
    provenance = dict(archive_sha256=hashlib.sha256((here / "results/casr-refusal-position-long-measurements-20261029.tar.gz").read_bytes()).hexdigest(),
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), audit=audit,
        prefix_replay_metrics_match=True,
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [out/"records.json", out/"run_summary.json", figure_dir/"position_long_curves.pdf",
                  figure_dir/"position_long_table.tex"]})
    (figure_dir / "position_long_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
