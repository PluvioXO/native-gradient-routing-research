"""Build the report's graphs from saved measurements; performs no model inference.

Run from the project root with /usr/bin/python3 experiments/report_visuals.py.
Older studies use their saved aggregate tables, explicitly labelled as such.
The generator needs repository inputs; the ZIP includes plotted CSV data and
compact Colab/unrestricted-removal metric caches, not all original raw archives.
"""
from pathlib import Path
import hashlib
import json
import re
import statistics as stats
import tarfile
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/report_visuals"
DATA = OUT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, RED, GRAY = "#2563EB", "#D97706", "#15836B", "#B91C1C", "#64748B"
SEEDS = [31, 47, 71]
TASKS = ["Arithmetic", "Pointer", "Boolean", "Brackets", "Sorting", "Stack", "Shortest path", "Sudoku"]
INPUTS, CHARTS = {}, []
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "legend.fontsize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tracked(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    INPUTS[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = sha(path)
    return path


def load_json(path):
    return json.loads(tracked(path).read_text())


def table(path):
    """Read existing numeric LaTeX table rows without executing TeX."""
    result = []
    for line in tracked(path).read_text().splitlines():
        if " & " in line and line.rstrip().endswith(r"\\"):
            result.append([x.strip().strip("$") for x in line.rstrip()[:-2].split(" & ")])
    return result


def number(cell):
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", cell)
    return float(values[0])


def mean_sd(cell):
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", cell)
    return float(values[0]), float(values[1]) if len(values) > 1 else 0.0


def figgrid(rows=1, cols=2):
    return plt.subplots(rows, cols, figsize=(12, 7.8), squeeze=False)


def axis(ax, title, ylabel=None, percent=False):
    width = 27 if ax.get_subplotspec().get_gridspec().ncols >= 3 else 44
    title = "\n".join(textwrap.fill(line, width) for line in title.splitlines())
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel)
    if percent:
        ax.set_ylim(-2, 106)
        ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)


def baseline(ax, value, label="Frozen base", color=GRAY):
    ax.axhline(value, color=color, lw=1.5, ls="--", label=f"{label}: {value:.1f}%")


def bars(ax, categories, series, colors, errors=None, base=None, ylim=106):
    x = np.arange(len(categories))
    width = .78 / len(series)
    for j, (label, values) in enumerate(series.items()):
        positions = x + (j-(len(series)-1)/2)*width
        err = errors[label] if errors else None
        ax.bar(positions, values, width, color=colors[j], label=label, yerr=err,
               capsize=3, error_kw={"elinewidth": 1})
    ax.set_xticks(x, categories)
    ax.set_ylim(0, ylim)
    if base is not None:
        baseline(ax, base)


def heatmap(ax, matrix, rows, cols, *, difference=False, vmin=0, vmax=100, fmt=".1f"):
    a = np.asarray(matrix, float)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if difference else None
    ax.imshow(a, aspect="auto", cmap="RdBu" if difference else "YlGnBu",
              norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(range(len(cols)), cols)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.spines[:].set_visible(False)
    for (i, j), value in np.ndenumerate(a):
        dark = abs(value) > .58*vmax if difference else value > 60
        ax.text(j, i, format(value, "+"+fmt if difference else fmt), ha="center", va="center",
                color="white" if dark else "#102134", fontsize=10, fontweight="bold")


def save(fig, name, section, title, subtitle, takeaway, note, records):
    index = len(CHARTS) + 1
    stem = f"{index:02d}_{name}"
    wrapped = textwrap.fill(title, 88)
    title_artist = fig.suptitle(wrapped, x=.05, y=.97, ha="left", va="top", fontsize=18, fontweight="bold", color="#122238")
    title_artist.set_in_layout(False)
    fig.text(.05, .875 if "\n" in wrapped else .914, subtitle, fontsize=11, color="#475569")
    fig.tight_layout(rect=(.025, .20, .98, .84 if "\n" in wrapped else .865), w_pad=3, h_pad=2)
    fig.text(.05, .089, textwrap.fill(takeaway, 125), fontsize=11.5, fontweight="bold", color="#122238")
    fig.text(.05, .025, textwrap.fill(note, 155), fontsize=8.5, color="#526173")
    fig.text(.96, .025, str(index), fontsize=9, ha="right", color=GRAY)
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", dpi=160, facecolor="white")
    pd.DataFrame(records).to_csv(DATA / f"{stem}.csv", index=False)
    plt.close(fig)
    CHARTS.append(dict(stem=stem, section=section, title=title, takeaway=takeaway, note=note))
    print(f"Graph {index:02d}: {title}", flush=True)


def capability_graphs():
    # No original sweep trajectories are currently local: plot saved final aggregates.
    sweep = []
    for row in table("figures/native_full_budget_table.tex")[1:]:
        avg, sd = mean_sd(row[2])
        sweep.append(dict(allocation=row[0], method=row[0].split()[0], params_m=number(row[1]),
                          accuracy=avg, sd=sd, rl_minutes=number(row[4]), ppl_change=number(row[5])))
    assert len(sweep) == 19
    fig, axes = figgrid()
    for ax, zoom in zip(axes[0], [False, True]):
        for method, color, marker in [("Native", BLUE, "o"), ("LoRA", ORANGE, "s")]:
            rows = [r for r in sweep if r["method"] == method and (not zoom or r["params_m"] <= .75)]
            ax.errorbar([r["params_m"]*1000 for r in rows], [r["accuracy"] for r in rows],
                        yerr=[r["sd"] for r in rows], color=color, marker=marker, capsize=3, label=method)
            if zoom:
                for r in rows:
                    ax.annotate(r["allocation"].replace("Native ", "N ").replace("LoRA ", "L "),
                                (r["params_m"]*1000, r["accuracy"]), xytext=(-14 if r["params_m"]>.6 else 4, 9 if method=="Native" else -16), textcoords="offset points", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("Trainable scalars per task (thousands; log scale)")
        axis(ax, "Small-budget detail" if zoom else "All 19 allocation settings", "Pointer accuracy (%)", True)
        baseline(ax, 38.3)
        ticks = [150,200,300,500,750] if zoom else [100,200,500,1000,2000,5000,10000]
        ax.set_xticks(ticks,[str(v) for v in ticks]); ax.set_xticks([],minor=True)
        ax.set_ylim(-2,114)
    axes[0, 0].legend(loc="lower right")
    save(fig, "budget_accuracy", "4.1", "Small native allocations can learn pointer following",
         "57 fits | MLP + attention | final update 160 | 3 seeds per setting",
         "Two native channels plus attention reach 96.1%; the six-channel dip shows that wider is not always better.",
         "Saved rounded table aggregates; bars show sample SD, not confidence intervals. Common LR and unequal budgets; not tuned-method superiority.", sweep)

    fig, axes = figgrid()
    for method, color in [("Native", BLUE), ("LoRA", ORANGE)]:
        rows = [r for r in sweep if r["method"] == method]
        for ax, key in zip(axes[0], ["rl_minutes", "ppl_change"]):
            ax.plot([r["params_m"]*1000 for r in rows], [r[key] for r in rows], "o-", color=color, label=method)
            ax.set_xscale("log"); ax.set_xlabel("Trainable scalars (thousands; log scale)")
    axis(axes[0, 0], "Training time does not shrink with allocation", "Recorded RL minutes")
    axes[0, 0].set_ylim(0, 65); axes[0, 0].legend()
    axis(axes[0, 1], "Measured text-modeling drift", "WikiText perplexity increase (%)")
    axes[0, 1].axhline(0, color=GRAY, ls="--", label="Frozen base: 0% change"); axes[0, 1].legend()
    save(fig, "budget_cost_retention", "4.1", "A smaller update is not automatically faster or less disruptive",
         "Same 57-fit sweep | lower perplexity increase is better",
         "Most fits take roughly 50 recorded RL minutes; retention and task accuracy are separate measurements.",
         "Table aggregates. RL clocks exclude evaluation and are not FLOPs. The later tuned rank-one LoRA pointer result is 99.7%, under a different recipe.", sweep)

    rows = table("figures/native_factorial_summary_table.tex")[1:]
    records = [dict(support=r[0], rule=r[1], arithmetic_owner=number(r[2]), arithmetic_joint=number(r[3]),
                    pointer_owner=number(r[4]), pointer_joint=number(r[5])) for r in rows]
    fig, axes = figgrid()
    for ax, support in zip(axes[0], ["Native", "LoRA"]):
        selected = [r for r in records if r["support"] == support]
        matrix = [[17.8, 17.8, 36.7, 36.7]] + [[r[k] for k in ("arithmetic_owner", "arithmetic_joint", "pointer_owner", "pointer_joint")] for r in selected]
        heatmap(ax, matrix, ["Frozen base", "Neither", "Routing", "Preservation", "Both"],
                ["Arithmetic\nalone", "Arithmetic\njoint", "Pointer\nalone", "Pointer\njoint"])
        ax.set_title(support, loc="left", fontweight="bold", pad=14)
    save(fig, "two_task_factorial", "4.2", "Two-task training has strong native and LoRA solutions",
         "24 fits | MLP only | 320 updates | heatmap cells are mean accuracy (%)",
         "Native routing + preservation works well; LoRA preservation without reward routing is equally strong here.",
         "Rounded means across 3 seeds; per-cell seed uncertainty is not displayed. Native 117 channels versus LoRA rank 18; not an attention ablation.", records)

    curve_rows = table("figures/native_coexistence_curves_table.tex")[2:]
    coexist = []
    for r in curve_rows:
        d = dict(allocation=r[0], params_k=number(r[1]))
        for key, cell in zip(["one", "two", "four", "eight", "no_owner", "calibrated"], r[2:]):
            d[key], d[key+"_sd"] = mean_sd(cell)
        coexist.append(d)
    assert len(coexist) == 4
    fig, axes = figgrid()
    for ax, rows, title in zip(axes[0], [coexist[:2], coexist[2:]], ["Small allocations", "Medium allocations"]):
        for r, color in zip(rows, [BLUE, ORANGE]):
            ax.errorbar([1, 2, 4, 8], [r[k] for k in ("one", "two", "four", "eight")],
                        yerr=[r[k+"_sd"] for k in ("one", "two", "four", "eight")], color=color, marker="o", capsize=4, label=r["allocation"])
        axis(ax, title, "Mean task accuracy (%)", True); baseline(ax, 29.1)
        ax.set_xticks([1, 2, 4, 8]); ax.set_xlabel("Number of independently trained updates enabled"); ax.legend()
    save(fig, "independent_coexistence", "4.3", "Separately learned skills interfere when enabled together",
         "96 specialist fits | 8 procedural tasks | MLP + attention | 3 seeds",
         "Strong standalone specialists lose about 21–29 percentage points when all eight updates are active.",
         "Means and sample SD from saved aggregate tables. Two updates average all pairs; four average eight balanced subsets. No oracle task selection.", coexist)

    tasks = table("figures/native_coexistence_tasks_table.tex")[1:]
    tasks = [r for r in tasks if r[0] in ("Small", "Medium")]
    task_records = []
    fig, axes = figgrid()
    for ax, budget in zip(axes[0], ["Small", "Medium"]):
        selected = [r for r in tasks if r[0] == budget]
        mat = []
        for r in selected:
            vals = [number(x) for x in r[2:]]
            mat.append([vals[0], vals[1], vals[2], vals[5], vals[6]])
            task_records.append(dict(budget=budget, task=r[1], base=vals[0], native_owner=vals[1], native_eight=vals[2], lora_owner=vals[5], lora_eight=vals[6]))
        heatmap(ax, mat, TASKS, ["Base", "Native\nalone", "Native\neight", "LoRA\nalone", "LoRA\neight"])
        ax.set_title(budget+" allocation", loc="left", fontweight="bold", pad=14)
    save(fig, "taskwise_interference", "4.3", "Arithmetic and pointer following show especially large combination losses",
         "Same independent-specialist study | each task has 120 test questions",
         "The macro score hides strong skill suppression and weakly learned tasks; Sudoku remains difficult even alone.",
         "Rounded three-seed means, without cellwise error bars. These are separate task prompts, not chained multi-skill solutions.", task_records)

    lang = [r for r in table("figures/native_coexistence_language_table.tex")[1:] if r[0] != "Frozen base"]
    fig, axes = figgrid()
    x = np.arange(4)
    for ax in axes[0]:
        ax.set_xticks(x, ["Native 4", "LoRA r1", "Native 26", "LoRA r4"])
    bars(axes[0, 0], [r["allocation"] for r in coexist], {"Unit-strength eight": [r["eight"] for r in coexist], "Calibrated eight": [r["calibrated"] for r in coexist]}, [GRAY, GREEN],
         errors={"Unit-strength eight": [r["eight_sd"] for r in coexist], "Calibrated eight": [r["calibrated_sd"] for r in coexist]}, base=29.1, ylim=75)
    axis(axes[0, 0], "Calibration changes the support comparison", "All-eight accuracy (%)"); axes[0, 0].legend(loc="upper left")
    bars(axes[0, 1], [r[0] for r in lang], {"Unit strength": [number(r[2]) for r in lang], "Calibrated": [number(r[4]) for r in lang]}, [GRAY, GREEN], ylim=26)
    axis(axes[0, 1], "Calibration also changes language-model drift", "WikiText perplexity increase (%)"); axes[0, 1].legend()
    save(fig, "calibration_retention", "4.3", "A fair comparison includes development-only mixture calibration",
         "Coefficients selected on development data; test/retention answers do not select them",
         "Calibrated small-budget LoRA exceeds native; medium budgets nearly tie. Retention ordering also changes.",
         "Accuracy bars show sample SD. Perplexity bars are rounded aggregate means without uncertainty in the saved table. Some calibrated coefficients are zero.",
         [dict(**r, unit_ppl_change=number(l[2]), calibrated_ppl_change=number(l[4])) for r, l in zip(coexist, lang)])

    retained = table("figures/native_coexistence_retention_table.tex")[1:]
    assert len(retained)==9
    base_scores=[number(v) for v in retained[0][2:]]
    retained=retained[1:]
    fig,axes=figgrid()
    retention_records=[]
    for ax,mix,title in zip(axes[0],["Unit","Cal."],["All-eight, unit-strength updates","All-eight, calibrated updates"]):
        chosen=[r for r in retained if r[1]==mix]
        changes=[]
        for r in chosen:
            values=[re.findall(r"[-+]?\d+(?:\.\d+)?",c) for c in r[2:]]
            changes.append([float(v[1]) for v in values])
            retention_records += [dict(allocation=r[0],mix=mix,benchmark=b,base=base_scores[i],accuracy=float(v[0]),change_pp=float(v[1])) for i,(b,v) in enumerate(zip(["ARC-Easy","HellaSwag","MMLU","PIQA"],values))]
        heatmap(ax,changes,[r[0] for r in chosen],["ARC-Easy","HellaSwag","MMLU","PIQA"],difference=True,vmax=6)
        ax.set_title(title,loc="left",fontweight="bold",pad=14)
    save(fig,"benchmark_retention","4.3","Benchmark effects are mixed, rather than uniform information loss",
         "Change from the frozen base in percentage points | blue improves, red regresses | 996 fixed questions",
         "MMLU declines in every displayed condition; ARC-Easy usually improves. Calibration changes these trade-offs too.",
         "Saved rounded aggregate means, without benchmark-wise uncertainty. This fixed, repeatedly inspected panel is descriptive and cannot establish general knowledge retention.",retention_records)

    comp = [r for r in table("figures/native_coexistence_compounds_table.tex")[1:] if r[0] in ("Pointer--arithmetic", "Stack--sorting")]
    fig, axes = figgrid()
    for ax, family, base in zip(axes[0], ["Pointer--arithmetic", "Stack--sorting"], [15, 32.5]):
        selected = [r for r in comp if r[0] == family]
        bars(ax, [r[1] for r in selected], {"Direct two-update model": [number(r[5]) for r in selected], "Explicit generated pipeline": [number(r[9]) for r in selected]}, [BLUE, GREEN], base=base)
        axis(ax, family.replace("--", " → "), "Compound accuracy (%)"); ax.tick_params(axis="x", labelsize=10)
        ax.legend(loc="upper left", fontsize=9)
    save(fig, "compound_tasks", "4.3", "Combining updates is different from executing skills in a pipeline",
         "Neither compound family was trained directly | generated intermediates are passed between specialists",
         "Pointer → arithmetic benefits greatly from explicit orchestration; stack → sorting does not show the same gain.",
         "Rounded means. Direct and pipeline conditions use different prompts/execution; this is not a controlled estimate of one-pass compositional ability.",
         [dict(family=r[0], allocation=r[1], base=number(r[2]), direct_both=number(r[5]), generated_pipeline=number(r[9])) for r in comp])

    initial = table("figures/aware_primary_table.tex")[1:]
    retain = [r for r in table("figures/aware_retention_table.tex")[1:] if len(r) == 4]
    fig, axes = figgrid()
    modes = ["Independent", "Joint-aware", "Preservation + conservation"]
    labels = ["Native\nsmall", "Native\nmedium", "LoRA\nsmall", "LoRA\nmedium"]
    series = {m: [mean_sd(r[j+2])[0] for r in initial] for j, m in enumerate(modes)}
    errors = {m: [mean_sd(r[j+2])[1] for r in initial] for j, m in enumerate(modes)}
    bars(axes[0, 0], labels, series, [GRAY, BLUE, GREEN], errors=errors, base=29.1, ylim=75)
    axis(axes[0, 0], "All-eight skill performance", "Accuracy (%)"); axes[0, 0].legend(fontsize=9)
    series = {m: [mean_sd(retain[3*i+j][2])[0] for i in range(4)] for j, m in enumerate(modes)}
    errors = {m: [mean_sd(retain[3*i+j][2])[1] for i in range(4)] for j, m in enumerate(modes)}
    bars(axes[0, 1], labels, series, [GRAY, BLUE, GREEN], errors=errors, ylim=34)
    axis(axes[0, 1], "Text-modeling drift", "Perplexity increase (%)"); axes[0, 1].legend(fontsize=9)
    save(fig, "first_continuation", "4.4", "Joint continuation improves coexistence; conservation reduces text drift",
         "36 continuation banks | 80 additional updates per task | 3 seeds per condition",
         "Training in the shared model already helps. The extra losses motivated the later component ablations.",
         "Saved rounded means and sample SD. This first continuation study is separate from the later 48-bank factorial; results are not pooled.",
         [dict(allocation=r[0], mode=m, accuracy=mean_sd(r[j+2])[0], accuracy_sd=mean_sd(r[j+2])[1], ppl=mean_sd(retain[3*i+j][2])[0], ppl_sd=mean_sd(retain[3*i+j][2])[1]) for i, r in enumerate(initial) for j, m in enumerate(modes)])

    old = load_json("experiments/adaptive_ablation/analysis/per_bank.json")
    combined = load_json("experiments/preservation_components/analysis/combined_per_bank.json")
    assert len(old) == 24 and len(combined) == 48
    assert len({(r["support"], r["mode"], r["seed"]) for r in combined}) == 48
    def summarize(rows, mode, support, key):
        v = [r[key] for r in rows if r["mode"] == mode and r["support"] == support]
        assert len(v) == 3
        return stats.mean(v), stats.stdev(v)
    def objective_plot(rows, modes, labels, name, section, title, takeaway):
        fig, axes = figgrid()
        for ax, field, heading, ylabel, limit in zip(axes[0], ["accuracy_8", "wiki_change"], ["All-eight accuracy", "WikiText drift"], ["Accuracy (%)", "Perplexity increase (%)"], [78, 23]):
            series = {("Native" if s=="native" else "LoRA"): [summarize(rows,m,s,field)[0] for m in modes] for s in ("native", "lora")}
            errors = {("Native" if s=="native" else "LoRA"): [summarize(rows,m,s,field)[1] for m in modes] for s in ("native", "lora")}
            bars(ax, labels, series, [BLUE, ORANGE], errors=errors, base=29.0625 if field == "accuracy_8" else None, ylim=limit)
            axis(ax, heading, ylabel); ax.tick_params(axis="x", labelsize=10); ax.legend(fontsize=9)
        save(fig, name, section, title, "Native width 4 versus LoRA rank 1 | all eight updates active | three seeds", takeaway,
             "Means ± sample SD. All arms retain joint-aware RL/reference KL. The follow-up reuses 24 archived controls once, rather than adding replications.", rows)
    objective_plot(old, ["aware", "preservation_only", "conservation_only", "preserve"],
                   ["Neither", "Preserve", "Conserve", "Both"], "objective_ablation", "4.5",
                   "Preservation and conservation address different outcomes",
                   "Preservation improves combined skills; conservation gives the largest reduction in measured text drift.")
    fig,axes=figgrid(2,2)
    for i,support in enumerate(["native","lora"]):
        for j,field in enumerate(["accuracy_8","wiki_change"]):
            modes_by_c={"Conservation off":["aware","self_only","foreign_only","preservation_only"],
                        "Conservation on":["conservation_only","self_conserve","foreign_conserve","preserve"]}
            series={label:[summarize(combined,m,support,field)[0] for m in modes] for label,modes in modes_by_c.items()}
            errors={label:[summarize(combined,m,support,field)[1] for m in modes] for label,modes in modes_by_c.items()}
            ax=axes[i,j];bars(ax,["None","Self","Foreign","Self + foreign"],series,[BLUE,GREEN],errors=errors,base=29.1 if j==0 else None,ylim=75 if j==0 else 21)
            axis(ax,("Native" if i==0 else "LoRA")+(" | all-eight accuracy" if j==0 else " | WikiText drift"),"Accuracy (%)" if j==0 else "Perplexity increase (%)")
            ax.tick_params(axis="x",labelsize=10)
    axes[0,0].legend(fontsize=8,ncol=2)
    save(fig,"component_means","4.6","The complete factorial separates coexistence gains from text retention",
         "All 8 objective combinations × 2 parameterizations × 3 seeds | 48 banks, including 24 reused controls",
         "Foreign preservation usually helps combined skills; conservation sharply lowers perplexity drift across preservation settings.",
         "Means ± sample SD. Width 4 native versus rank 1 LoRA. All conditions retain joint-aware RL/reference KL; this is not a native-versus-LoRA superiority test.",combined)

    fig,axes=figgrid()
    for ax,support in zip(axes[0],["native","lora"]):
        series={label:[summarize(combined,m,support,"mc_change")[0] for m in modes] for label,modes in modes_by_c.items()}
        errors={label:[summarize(combined,m,support,"mc_change")[1] for m in modes] for label,modes in modes_by_c.items()}
        bars(ax,["None","Self","Foreign","Self + foreign"],series,[BLUE,GREEN],errors=errors,ylim=3)
        ax.set_ylim(-3,3);ax.axhline(0,ls="--",color=GRAY,label="Frozen base: zero change")
        axis(ax,"Native" if support=="native" else "LoRA","Multiple-choice accuracy change (pp)");ax.legend(fontsize=9)
    save(fig,"component_mc_retention","4.6","Reduced text drift does not establish broad accuracy gains",
         "Same 48-bank factorial | aggregate accuracy on the fixed 996-question retention panel",
         "Multiple-choice changes remain small and mixed; these measurements qualify what the conservation objective establishes.",
         "Means ± sample SD across three seeds. Negative = lower accuracy than base. Absolute accuracy is not shown. Repeatedly inspected benchmarks, not a fresh confirmatory test.",combined)
    planned = load_json("experiments/preservation_components/analysis/planned_contrasts.json")
    fig, axes = figgrid()
    for ax, key, title in zip(axes[0], ["coexistence_foreign_minus_self", "standalone_self_minus_foreign"], ["Joint use: foreign − self", "Standalone: self − foreign"]):
        for j, (seed, color) in enumerate(zip(SEEDS, [BLUE, ORANGE, GREEN])):
            ax.scatter([r[key]["paired"][j] for r in planned], np.arange(4)+(j-1)*.12, color=color, s=55, label=f"Seed {seed}")
        ax.scatter([r[key]["mean"] for r in planned], range(4), marker="D", color="#122238", s=65, label="Mean")
        ax.axvline(0, ls="--", color=GRAY)
        ax.set_yticks(range(4), [f"{'Native' if r['support']=='native' else 'LoRA'}, C {'on' if r['conservation'] else 'off'}" for r in planned]); ax.invert_yaxis()
        axis(ax, title); ax.set_xlabel("Accuracy difference (percentage points)"); ax.grid(axis="x", alpha=.15)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    save(fig, "paired_component_contrasts", "4.6", "The foreign-preservation advantage is clearer than the self advantage",
         "The recorded prediction compared the same three seeds in each condition | C = conservation",
         "Foreign beats self in 11 of 12 paired comparisons; those comparisons share three seeds and are not 12 independent replications.",
         "Individual seeds and their mean; no population confidence intervals. Positive values match the predicted term in each panel; axes differ.",
         [dict(support=r["support"], conservation=r["conservation"], seed=seed, foreign_minus_self=r["coexistence_foreign_minus_self"]["paired"][i], self_minus_foreign_owner=r["standalone_self_minus_foreign"]["paired"][i]) for r in planned for i, seed in enumerate(SEEDS)])
    task_contrasts = load_json("experiments/preservation_components/analysis/task_contrasts.json")
    fig, axes = figgrid(cols=1)
    mat = [[next(r["mean"] for r in task_contrasts if r["task"] == t and r["support"] == s and r["conservation"] == c) for s,c in [("native",0),("native",1),("lora",0),("lora",1)]] for t in range(8)]
    heatmap(axes[0, 0], mat, TASKS, ["Native, C off", "Native, C on", "LoRA, C off", "LoRA, C on"], difference=True, vmax=40)
    save(fig, "taskwise_components", "4.6", "Pointer following drives much of the preservation gain",
         "All-eight accuracy: foreign preservation minus self preservation, in percentage points",
         "Blue cells improve with foreign preservation; red cells regress. Better macro accuracy does not mean every skill is protected.",
         "Paired three-seed means from saved analysis. This plot is a component contrast, not change relative to the pretrained base.", task_contrasts)


def refusal_graphs():
    # Preserve compact copies of the Colab measurements so reruns need not depend on Downloads.
    pilot_zip = Path.home() / "Downloads/refusal_removal_copy_c4_03_measurements.zip"
    if pilot_zip.exists():
        tracked(pilot_zip)
        with zipfile.ZipFile(pilot_zip) as z:
            for name in ("before_removal.csv", "curves.csv", "endpoints.csv"):
                (DATA / ("colab_"+name)).write_bytes(z.read(name))
            b = json.loads(z.read("baseline.json"))
            (DATA / "colab_baseline_metrics.json").write_text(json.dumps({k:v["metrics"] for k,v in b.items()}, indent=2))
    before = pd.read_csv(tracked(DATA / "colab_before_removal.csv"))
    curves = pd.read_csv(tracked(DATA / "colab_curves.csv"))
    assert len(curves) == 480 and set(curves.arm) == {"channels", "scattered", "full"}
    arms = ["original_base", "channels", "scattered", "full"]
    labels = ["Base", "Routed", "Scattered", "Full model"]
    colors = [GRAY, BLUE, ORANGE, GREEN]
    fig, axes = figgrid()
    for ax, panel, heading in zip(axes[0], ["test", "transfer"], ["Original wording", "Alternate wording"]):
        rows = [before[(before.arm == a) & (before.panel == panel)].iloc[0] for a in arms]
        bars(ax, labels, {"Blocked-category refusal": [100*r.blocked_refusal for r in rows], "Allowed-category correct": [100*r.allowed_correct for r in rows]}, [RED, GREEN])
        axis(ax, heading, "Before-removal rate (%)", True)
        for container in ax.containers:
            ax.bar_label(container,fmt="%.1f%%",padding=3,fontsize=8)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc="lower center",bbox_to_anchor=(.5,.125),ncol=2,fontsize=10,frameon=False)
    save(fig, "colab_acquisition", "5.1", "The initial Colab pilot learned the original wording, with weak transfer",
         "One seed | four routed MLP channels across 8 blocks + attention | scalar-matched scattered / full controls",
         "Primary refusal was strong; alternate wording failed badly for routed and scattered updates in this pilot.",
         "Actual saved Colab measurements, not the initial failed arithmetic task. Routed/scattered train 200,704 scalars; the server's single-layer study uses 10,752.", before.to_dict("records"))

    fig, axes = figgrid(2, 2)
    lrs = sorted(curves.lr.unique())
    assert len(lrs) == 2
    for i, lr in enumerate(lrs):
        for j, (key, heading) in enumerate([("blocked_refusal", "Blocked-category refusal"), ("blocked_correct", "Blocked-category correct answers")]):
            ax = axes[i, j]
            for arm, color in zip(arms[1:], colors[1:]):
                rows = curves[(curves.arm == arm) & (curves.lr == lr) & (curves.panel == "test")].sort_values("step")
                ax.plot(rows.step, 100*rows[key], "o-", ms=3, color=color, label=arm.title())
            ax.set_xscale("symlog", linthresh=4, base=2); ax.set_xticks([0, 1, 2, 4, 8, 16, 32, 64, 128])
            ax.set_xticklabels([0, 1, 2, 4, 8, 16, 32, 64, 128], fontsize=9)
            axis(ax, f"{heading} | LR {lr:g}", "Rate (%)", True)
            ax.set_xlabel("Removal updates (expanded early steps)")
    axes[0, 0].legend(fontsize=9)
    save(fig, "colab_removal", "5.1", "The Colab pilot tracked refusal and answer recovery throughout removal",
         "Same learned starts | two removal learning rates | 128 updates = 2 epochs",
         "Routed and scattered starts recover primary answers after one observed update; the full-model start takes two here.",
         "One-seed pilot. X axis is linear through update 4 and logarithmic thereafter. Dots are actual observations; joining lines do not imply measured intermediate points.", curves.to_dict("records"))

    acq = load_json("../refusal/figures/acquisition_records.json")
    parent_archive = ROOT.parent / "refusal/experiments/refusal_layer/results/casr-layer-measurements-20261028.tar.gz"
    parent = []
    if parent_archive.exists():
        with tarfile.open(tracked(parent_archive)) as archive:
            for member in archive.getmembers():
                match = re.fullmatch(r"runs/((?:layer\d+|full)_s\d+)/seed_\d+/(?:channels|full)/removal_lr_([^/]+)/eval_(\d+)\.json", member.name)
                if not match:
                    continue
                value = json.load(archive.extractfile(member))
                for panel, report in value["panels"].items():
                    parent.append(dict(source=match[1], lr=float(match[2]), step=value["step"], panel=panel, **report["metrics"]))
        (DATA / "server_unrestricted_metrics.json").write_text(json.dumps(parent,indent=2)+"\n")
        (DATA / "server_unrestricted_cache_origin.json").write_text(json.dumps(dict(source_archive=str(parent_archive.relative_to(ROOT)),source_archive_sha256=sha(parent_archive)),indent=2)+"\n")
    else:
        parent = load_json(DATA / "server_unrestricted_metrics.json")
        load_json(DATA / "server_unrestricted_cache_origin.json")
    assert len({(r["source"],r["lr"],r["step"]) for r in parent}) == 330
    groups = ["layer03", "layer12", "layer22", "full"]
    group_labels = ["Layer 3", "Layer 12", "Layer 22", "Full model"]
    fig, axes = figgrid()
    counts = [sum(v["qualified"] for k,v in acq.items() if k.startswith(g+"_")) for g in groups]
    assert counts == [3, 2, 3, 3]
    ax = axes[0, 0]; ax.bar(group_labels, counts, color=[BLUE,BLUE,BLUE,ORANGE])
    ax.set_ylim(0, 3.8); ax.set_yticks([0,1,2,3]); axis(ax, "Acquisition qualification", "Qualified models out of 3")
    for i,n in enumerate(counts): ax.text(i,n+.08,f"{n}/3",ha="center",fontweight="bold")
    endpoints = []
    for source in sorted({r["source"] for r in parent}):
        for lr in (1e-6, 1e-5):
            rows = sorted([r for r in parent if r["source"] == source and r["lr"] == lr and r["panel"] == "test"], key=lambda r:r["step"])
            first = next(r["step"] for r in rows if r["blocked_correct"] >= .9)
            endpoints.append(dict(source=source, lr=lr, first_90_correct=first))
            i = groups.index(source.split("_s")[0]); seed=int(source.split("_s")[1])
            axes[0, 1].scatter(i+(-.12 if lr == 1e-6 else .12)+(SEEDS.index(seed)-1)*.035, first,
                               color=BLUE if lr == 1e-6 else ORANGE, s=55, marker="o" if lr == 1e-6 else "s",
                               label=f"LR {lr:g}" if source==sorted({r['source'] for r in parent})[0] else None)
    axes[0, 1].set_xticks(range(4),group_labels); axes[0, 1].set_ylim(0,14)
    axis(axes[0, 1], "First recorded ≥90% answer recovery", "Removal updates"); axes[0, 1].legend()
    save(fig, "single_layer_qualification", "5.2", "Localized refusal can be acquired, then rapidly removed",
         "9 localized + 3 full-model acquisitions | 22 unrestricted removal trajectories",
         "Eight of nine localized starts qualify; every qualified localized model recovers answers within 1–2 updates.",
         "Dots are individual seeds. Acquisition budgets/rates differ: 10,752 native scalars versus 494M full-model parameters. Later thresholds are interval-censored by evaluation spacing.", endpoints)

    fig, axes = figgrid(2,2)
    for i,lr in enumerate([1e-6,1e-5]):
        for j,(key,title) in enumerate([("blocked_refusal","Refusal"),("blocked_correct","Correct answers")]):
            ax=axes[i,j]
            for group,label,color in zip(groups,group_labels,[BLUE,GREEN,"#8B5CF6",ORANGE]):
                selected=[r for r in parent if r["source"].startswith(group+"_") and r["lr"]==lr and r["panel"]=="test"]
                plot_curve(ax, selected, key, color, label)
            ax.set_xlim(0,32); ax.set_xticks([0,2,4,8,16,24,32]); ax.set_xlabel("Full-model removal updates")
            axis(ax,f"{title} | LR {lr:g}","Rate (%)",True)
    axes[0,0].legend(fontsize=8,ncol=2)
    save(fig,"unrestricted_removal_curves","5.2","Full-model removal can behave differently from simple refusal disappearance",
         "All parameters train during removal | means and observed seed ranges | original wording",
         "Refusal can disappear, return, or give way to wrong answers; report answer correctness alongside refusal.",
         "Step 0 is after refusal acquisition. The original base answered these copying prompts. Localized groups n=3/2/3; full-model n=3. Curves use saved evaluations only.",parent)

    bypass_archive=tracked("../refusal/experiments/refusal_layer/results/casr-refusal-bypass-measurements-20261028.tar.gz")
    with tarfile.open(bypass_archive) as archive:
        exploratory=json.load(archive.extractfile("comparison.json"))
    comparison_grid(exploratory,[("inside_layer",BLUE,"Inside acquisition block"),("outside_layer",ORANGE,"Outside acquisition block"),("all",GRAY,"All parameters")],
                    "exploratory_inside_outside","5.3","Exploratory bypass: protecting a region does not protect its behaviour",
                    "32 restricted eight-update runs | inside/outside masks have unequal parameter counts",
                    "This was a mechanism exploration; the next graph uses matched two-block update budgets.")
    position=load_json("../refusal/experiments/refusal_layer/position_launch/comparison.json")
    comparison_grid(position,[("upstream",BLUE,"Two blocks before"),("downstream",ORANGE,"Two blocks after"),("all",GRAY,"Unrestricted reference")],
                    "matched_position","5.3","Other blocks can override refusal while the acquisition block stays frozen",
                    "32 matched adjacent-block runs | 29,824,768 trainable scalars per restricted mask",
                    "Late-layer low-rate refusal survives eight updates; every high-rate adjacent-block run reaches zero within four.")

    fig,axes=figgrid()
    left=[]
    for panel,label,color in [("test","Original wording",BLUE),("transfer","Alternate wording",ORANGE)]:
        rows=sorted([r for r in position if r["source"]=="layer22_s31" and r["scope"]=="upstream" and r["lr"]==1e-6 and r["panel"]==panel],key=lambda r:r["step"])
        axes[0,0].plot([r["step"] for r in rows],[100*r["blocked_refusal"] for r in rows],"o-",color=color,label=label);left+=rows
    axis(axes[0,0],"One plateau depends on prompt wording","Refusal (%)",True)
    axes[0,0].set_xlabel("Removal updates | layer 22, seed 31, before-block mask");axes[0,0].legend()
    final=[next(r for r in position if r["source"]==f"layer22_s{seed}" and r["scope"]=="downstream" and r["lr"]==1e-5 and r["panel"]=="test" and r["step"]==8) for seed in SEEDS]
    bottom=np.zeros(3)
    for key,label,color in [("blocked_correct","Correct",GREEN),("blocked_other","Other / wrong",GRAY),("blocked_refusal","Refusal",RED)]:
        values=np.array([100*r[key] for r in final]);axes[0,1].bar(range(3),values,bottom=bottom,color=color,label=label);bottom+=values
    axes[0,1].set_xticks(range(3),[f"Seed {s}" for s in SEEDS]);axis(axes[0,1],"Zero refusal can coexist with wrong answers","Share of blocked-category answers (%)",True)
    axes[0,1].legend(loc="lower center",fontsize=9)
    for i,r in enumerate(final):axes[0,1].text(i,100*r["blocked_correct"]/2,f"{100*r['blocked_correct']:.1f}%\ncorrect",ha="center",va="center",color="white",fontweight="bold")
    save(fig,"refusal_failure_modes","5.3","Refusal rate alone can mischaracterize a removal trajectory",
         "Two diagnostics from the matched adjacent-block experiment",
         "One model loses refusal on alternate wording first; others stop refusing but still produce wrong answers.",
         "Alternate wording was excluded from SFT but used in acquisition validation; it is not untouched OOD. Right: layer 22, after-block mask, LR 1e-5, update 8.",left+final)

    long=load_json("../refusal/experiments/refusal_layer/position_long_analysis/records.json")
    long_summary=load_json("../refusal/experiments/refusal_layer/position_long_analysis/run_summary.json")
    fig,axes=figgrid()
    for ax,key,title in zip(axes[0],["blocked_refusal","blocked_correct"],["Original-form refusal","Original-form answer accuracy"]):
        for seed,color in zip(SEEDS,[BLUE,ORANGE,GREEN]):
            for scope,style,tag in [("upstream","-","before"),("downstream","--","after")]:
                rows=sorted([r for r in long if r["source"]==f"layer22_s{seed}" and r["scope"]==scope and r["panel"]=="test"],key=lambda r:r["step"])
                assert len(rows)==65
                ax.plot([r["step"] for r in rows],[100*r[key] for r in rows],ls=style,color=color,label=f"Seed {seed}, {tag}")
        ax.axvline(8,ls=":",color=GRAY);ax.set_xlabel("Removal updates | 4 examples per update")
        axis(ax,title,"Rate (%)",True);ax.set_xticks([0,8,16,32,48,64])
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.125),ncol=3,fontsize=9,frameon=False)
    save(fig,"extended_late_layer","5.4","The late-layer plateau is a delay, not lasting protection",
         "6 extended trajectories | layer 22 remains frozen | LR 1e-6 | same three acquisition seeds",
         "All six runs eventually reach zero refusal, first at updates 13–37; answer recovery can occur at a different time.",
         "Each curve is an individual trajectory. Dotted line marks the original eight-update window. Frozen acquisition block verified; these are extensions, not fresh seeds.",long)

    dist=load_json("../refusal/experiments/refusal_distribution/analysis/summary.json")
    short=load_json("../refusal/experiments/refusal_layer/position_analysis/run_summary.json")
    fig,axes=figgrid(1,3)
    control_records=[]
    for ax,rows,title in zip(axes[0],[short,long_summary,dist],["8-update adjacent blocks\n32 trajectories","64-update late-layer extension\n6 trajectories","Distributed acquisition\n14 full-model removal trajectories"]):
        for i,(key,label,color) in enumerate([("utility_correct","Arithmetic",BLUE),("arc_easy_correct","ARC-Easy","#8B5CF6"),("winogrande_correct","WinoGrande",ORANGE)]):
            values=[]
            for r in rows:
                v=r.get("controls",r)[key];delta=100*(v["after"]-v["before"]);values.append(delta)
                control_records.append(dict(study=title.split("\n")[0],run=r.get("name",r.get("condition")),seed=r.get("seed"),task=label,before=100*v["before"],after=100*v["after"],change_pp=delta))
            ax.bar(i,stats.mean(values),color=color,alpha=.65,width=.6)
            jitter=np.linspace(-.18,.18,len(values));ax.scatter(i+jitter,values,s=22,color=color,edgecolor="white",linewidth=.3,zorder=4)
            ax.annotate(f"mean {stats.mean(values):+.1f}",(i,stats.mean(values)),xytext=(0,7 if stats.mean(values)>=0 else -18),textcoords="offset points",ha="center",fontsize=9,fontweight="bold")
        ax.axhline(0,color=GRAY,lw=1);ax.set_xticks(range(3),["Arithmetic","ARC-Easy","WinoGrande"],rotation=25,ha="right",fontsize=10)
        axis(ax,title,"Change from pre-removal accuracy (pp)");ax.set_ylim(-46,62)
    save(fig,"refusal_capability_changes","5.4","Capability damage depends on the task and removal protocol",
         "Bars = trajectory means; dots = individual trajectories | all three panels share the same vertical scale",
         "Arithmetic drops in 5/6 extended runs, averaging −10.9 pp; other protocols show mixed losses and gains.",
         "32 items per diagnostic task; one changed answer = 3.125 pp. Trajectories reuse acquisition seeds. Green copying remains 100% at the recorded endpoints.",control_records)

    acquisition=load_json("../refusal/experiments/refusal_distribution/analysis/acquisition.json")
    drecords=load_json("../refusal/experiments/refusal_distribution/analysis/records.json")
    conditions=["single12","dual08_16","four03_09_15_21"]
    condition_labels=["4 channels\n1 layer","2 channels\n2 layers","1 channel\n4 layers"]
    fig,axes=figgrid()
    counts=[sum(r["qualified"] for r in acquisition if r["condition"]==c) for c in conditions]
    assert counts==[2,3,2] and len(dist)==14
    axes[0,0].bar(condition_labels,counts,color=[BLUE,GREEN,ORANGE]);axes[0,0].set_ylim(0,3.8);axes[0,0].set_yticks(range(4))
    axis(axes[0,0],"Equal acquisition budget: 10,752 scalars","Qualified acquisitions out of 3")
    for i,n in enumerate(counts):axes[0,0].text(i,n+.08,f"{n}/3",ha="center",fontweight="bold")
    for condition,label,color in zip(conditions,condition_labels,[BLUE,GREEN,ORANGE]):
        for lr,style in [(1e-6,"-"),(1e-5,"--")]:
            rows=[r for r in drecords if r["condition"]==condition and r["lr"]==lr and r["panel"]=="test" and r["step"]<=8]
            plot_curve(axes[0,1],rows,"blocked_refusal",color,label.replace("\n",", ")+f", {lr:g}",style)
    axis(axes[0,1],"Unrestricted removal: first eight updates","Blocked-category refusal (%)",True)
    axes[0,1].set_xticks(range(9));axes[0,1].set_xlabel("Full-model SFT updates");axes[0,1].legend(fontsize=8,ncol=2,loc="upper right")
    save(fig,"distributed_refusal","5.5","Spreading the same small refusal budget did not give durable protection",
         "9 acquisition jobs; 7 qualify; 14 removal trajectories | one-based layers: 12 / 8+16 / 3+9+15+21",
         "Every qualified trajectory first reaches zero refusal and ≥90% correct answering within one or two updates.",
         "Removal was measured through update 64; chart zooms to 0–8. Lines are means with observed seed ranges; group n=2/3/2. Acquisition failures are retained on the left.",drecords)


def plot_curve(ax, rows, key, color, label, style="-"):
    steps=sorted({r["step"] for r in rows})
    values=[[100*r[key] for r in rows if r["step"]==step] for step in steps]
    ax.plot(steps,[stats.mean(v) for v in values],color=color,label=label,ls=style,lw=2)
    ax.fill_between(steps,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.10)


def comparison_grid(records, scopes, name, section, title, subtitle, takeaway):
    fig,axes=figgrid(2,3)
    for i,lr in enumerate([1e-6,1e-5]):
        for j,layer in enumerate([3,12,22]):
            ax=axes[i,j]
            for scope,color,label in scopes:
                rows=[r for r in records if r["source"].startswith(f"layer{layer:02d}_") and r["scope"]==scope and r["lr"]==lr and r["panel"]=="test" and r["step"]<=8]
                plot_curve(ax,rows,"blocked_refusal",color,label,":" if scope=="all" else "-")
            axis(ax,f"Layer {layer} | LR {lr:g}","Refusal (%)",True);ax.set_xlabel("Removal updates");ax.set_xticks([0,2,4,6,8])
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.125),ncol=3,frameon=False,fontsize=10)
    save(fig,name,section,title,subtitle,takeaway,
         "Means and observed seed ranges. Early/late n=3; middle n=2 qualified acquisitions. Unrestricted reference has a different parameter budget. Original prompt forms.",records)


def historical_and_status():
    fig,axes=figgrid(cols=1);ax=axes[0,0];ax.axis("off")
    items=[["COSMIC source runner","Present"],["Completed COSMIC measurements in this project","Not found"],["Direction independence established by the report","No"],["New refusal training launched for these graphs","No"]]
    tbl=ax.table(cellText=items,colLabels=["Evidence item","Status"],cellLoc="left",colLoc="left",loc="center",colWidths=[.75,.25])
    tbl.auto_set_font_size(False);tbl.set_fontsize(13);tbl.scale(1,3)
    for (row,col),cell in tbl.get_celld().items():
        cell.set_edgecolor("white");cell.set_facecolor("#e8eef5" if row else "#c9d7e8")
    save(fig,"cosmic_status","5.6","COSMIC: implementation exists, but no completed outcome is available here",
         "Status panel only — it is not a measured zero effect or a negative experiment",
         "There is no numerical COSMIC result to graph in this checkout; the absence of a local export does not prove it never ran.",
         "The earlier A/B direction pilot in Downloads is a different experiment and is not substituted for the authors' COSMIC procedure.",
         [dict(item=a,status=b) for a,b in items])
    for name,title in [
        ("soft_routing_stability","Historical numerical check: routing and dynamical stability"),
        ("readout_gain_control","Historical numerical check: control for readout-gain confounds"),
        ("teacher_student_comparison","Historical numerical check: teacher–student learning comparisons"),
        ("coupling_capacity_tradeoff","Historical numerical check: coupling and capacity trade-off")]:
        fig,axes=figgrid(1,1)
        path=tracked(f"notes/legacy/results/{name}.png");axes[0,0].imshow(plt.imread(path));axes[0,0].axis("off")
        save(fig,name,"9",title,"Earlier work reproduced from saved figures | outside the 19 October–2 November experiment window",
             "These are earlier numerical/theory controls, not the new Qwen capability or refusal experiments.",
             "Original historical figures are reproduced without recomputing their data. CASR/V6/V7 raw runs are not all available locally; no missing trajectories are invented.",
             [dict(source=f"notes/legacy/results/{name}.png")])


def assemble():
    cover,ax=plt.subplots(figsize=(12,7.8));ax.axis("off")
    cover.text(.06,.92,"Visual companion to report.pdf",fontsize=25,fontweight="bold",color="#122238")
    cover.text(.06,.86,"Native capability adaptation and refusal localization | saved experimental evidence",fontsize=13,color=GRAY)
    section_labels={"4.1":"Channel/rank budget sweep", "4.2":"Two-task routing × preservation", "4.3":"Independent skills: mixing, calibration, retention",
                    "4.4":"First coexistence-aware continuation", "4.5":"Preservation × conservation ablation", "4.6":"Complete self / foreign / conservation factorial",
                    "5.1":"Colab acquisition and removal pilot", "5.2":"Single-layer refusal and unrestricted SFT", "5.3":"Frozen-block bypass and failure modes",
                    "5.4":"Extended removal and capability changes", "5.5":"Equal-budget distributed refusal", "5.6":"COSMIC: implementation status only", "9":"Earlier numerical and theory checks"}
    for x,sections,heading in [(.06,[s for s in section_labels if s.startswith("4")],"Capabilities"),(.54,[s for s in section_labels if not s.startswith("4")],"Refusal and historical context")]:
        cover.text(x,.78,heading,fontsize=16,fontweight="bold",color=BLUE)
        y=.71
        for section in sections:
            selected=[c for c in CHARTS if c["section"]==section]
            ids=[int(c["stem"][:2]) for c in selected]
            span=str(ids[0]) if len(ids)==1 else f"{ids[0]}–{ids[-1]}"
            cover.text(x,y,f"§{section}  |  Graphs {span}",fontsize=11,fontweight="bold",va="top")
            cover.text(x,y-.028,textwrap.fill(section_labels[section],52),fontsize=10.5,va="top")
            y-=.079
    cover.text(.06,.09,"Read the title for the result, the subtitle for the experiment, and the footer for its limits.",fontsize=12,fontweight="bold")
    cover.text(.06,.05,"Error bars are sample SD; refusal-curve shading is the observed seed range. Neither is a population confidence interval.",fontsize=10,color=GRAY)
    cover.savefig(OUT/"00_contents.pdf");cover.savefig(OUT/"00_contents.png",dpi=150);plt.close(cover)
    writer=PdfWriter();writer.append(str(OUT/"00_contents.pdf"),outline_item="Contents")
    for c in CHARTS:
        writer.append(str(OUT/(c["stem"]+".pdf")),outline_item=f"{c['stem'][:2]} · §{c['section']} · {c['title']}")
    writer.add_metadata({"/Title":"Visual companion: capabilities and refusal experiments","/Subject":"Saved measurements; no new model training"})
    with (ROOT/"notes/supervisor/report_graphs.pdf").open("wb") as f:writer.write(f)
    assert len(PdfReader(ROOT/"notes/supervisor/report_graphs.pdf").pages)==len(CHARTS)+1
    # Insert section-specific figure includes into report.tex separately, preserving prose.
    for section in sorted({c["section"] for c in CHARTS}):
        chunks=[]
        for c in CHARTS:
            if c["section"]!=section:continue
            caption=c["title"].replace("–","--").replace("&",r"\&").replace("%",r"\%")
            chunks.append("\\begin{figure}[htbp]\n\\centering\n"
                          +f"\\includegraphics[width=\\linewidth]{{figures/report_visuals/{c['stem']}.pdf}}\n"
                          +f"\\caption{{{caption}.}}\n\\end{{figure}}\n")
        (OUT/("section_"+section.replace(".","_")+".tex")).write_text("\n".join(chunks)+"\n\\FloatBarrier\n")
    pd.DataFrame(CHARTS).to_csv(OUT/"coverage.csv",index=False)
    integrate_report()
    manifest=dict(generator_sha256=sha(Path(__file__)),input_hashes=INPUTS,charts=CHARTS,graph_count=len(CHARTS),companion_pages=len(CHARTS)+1,
                  unrestricted_cache_origin=json.loads((DATA/"server_unrestricted_cache_origin.json").read_text()),
                  output_hashes={p.name:sha(p) for p in OUT.iterdir() if p.suffix in (".pdf",".png",".tex")},
                  data_hashes={p.name:sha(p) for p in DATA.iterdir() if p.is_file()},
                  report_graphs_sha256=sha(ROOT/"notes/supervisor/report_graphs.pdf"),
                  limitation="Saved records and rounded historical tables; no training, model inference, or independent raw-prediction audit.")
    (OUT/"provenance.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(ROOT/"notes/supervisor/report_graphs.zip","w",zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT/"notes/supervisor/report_graphs.pdf","report_graphs.pdf")
        z.write(Path(__file__).resolve(),str(Path(__file__).resolve().relative_to(ROOT)))
        for p in OUT.rglob("*"):
            if p.is_file():z.write(p,str(p.relative_to(ROOT)))
    print(f"Created report_graphs.pdf: {len(CHARTS)+1} pages, {len(CHARTS)} graph/status pages.",flush=True)


def integrate_report():
    path=ROOT/"report.tex"
    source=re.sub(r"\n*% BEGIN REPORT VISUALS[^\n]*\n.*?% END REPORT VISUALS\n*", "\n\n", path.read_text(), flags=re.S)
    for package in ["graphicx","placeins"]:
        if "\\usepackage{"+package+"}" not in source:
            source=source.replace(r"\usepackage{longtable}",r"\usepackage{longtable}"+"\n\\usepackage{"+package+"}")
    # Actual subsection numbers are determined from the source, not guessed from keyword matches.
    section=0;subsection=0;headings=[]
    for match in re.finditer(r"\\(section|subsection)\{",source):
        if match[1]=="section": section+=1;subsection=0;key=str(section)
        else: subsection+=1;key=f"{section}.{subsection}"
        headings.append((match.start(),key))
    additions=[]
    for i,(start,key) in enumerate(headings):
        include=OUT/("section_"+key.replace(".","_")+".tex")
        if include.exists():
            end=headings[i+1][0] if i+1<len(headings) else source.index(r"\end{document}")
            additions.append((end,f"\n% BEGIN REPORT VISUALS {key}\n\\input{{figures/report_visuals/{include.name}}}\n% END REPORT VISUALS\n\n"))
    assert len(additions)==13, "Every experiment section and historical context must be covered"
    for end,addition in reversed(additions):source=source[:end]+addition+source[end:]
    source=source.replace("references on page eight", "references beginning on page seven")
    intro="\\noindent\\textbf{Visual edition:} numbered graphs accompany each experimental study below. A larger, graph-only version is available in \\texttt{report\\_graphs.pdf}; CSV plot data and provenance are retained in \\path{figures/report_visuals/}.\n\n"
    if "\\textbf{Visual edition:}" not in source:
        source=source.replace(r"\section{Executive summary}",intro+r"\section{Executive summary}")
    if "Experiment graph companion &" not in source:
        source=source.replace("Colab refusal experiment &", "Experiment graph companion & \\texttt{report\\_graphs.pdf} \\\\\nIndividual graphs and CSV data & \\path{figures/report_visuals/} \\\\\nColab refusal experiment &")
    path.write_text(source)


def main():
    # Only remove this script's generated numbered outputs; retain cached Colab inputs.
    for directory,pattern in [(OUT,"[0-9][0-9]_*.pdf"),(OUT,"[0-9][0-9]_*.png"),(DATA,"[0-9][0-9]_*.csv"),(OUT,"section_*.tex")]:
        for path in directory.glob(pattern):path.unlink()
    capability_graphs()
    refusal_graphs()
    historical_and_status()
    assemble()


if __name__ == "__main__":
    main()
