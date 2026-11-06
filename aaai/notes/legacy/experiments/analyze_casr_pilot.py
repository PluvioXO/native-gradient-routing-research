"""Regenerate the CASR paper tables, figure, and per-cue checkpoint replay."""
import argparse
import hashlib
import io
import json
import math
import random
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import LinearOperator, lsmr
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, jacrev, jvp, vjp


def theory_checks():
    J = np.array([[1., 0.], [2., 1.]])
    C = J[1:]
    N = np.eye(2) - np.linalg.pinv(C) @ C
    a = np.array([-1., -1.])
    d = N @ a
    assert np.allclose(C @ d, 0.)
    assert np.isclose(a @ d, d @ d)
    r = np.array([.3])
    s = -np.linalg.pinv(C) @ r - .1 * N @ a
    assert np.allclose(r + C @ s, 0.)
    assert np.allclose(r + C @ (-.1 * d), r)  # tangent updates retain existing drift
    # Feasibility restoration can increase the task loss.
    before, after = .5 * (1. - 2.)**2, .5 * (0. - 2.)**2
    assert after > before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("experiments/casr_pilot/20260930T153702Z-18bc816b.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/casr_pilot"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    theory_checks()
    with zipfile.ZipFile(args.archive) as archive:
        roots = {name.split("/")[0] for name in archive.namelist()}
        assert len(roots) == 1
        prefix = roots.pop() + "/"
        read_json = lambda name: json.loads(archive.read(prefix + name))
        read_csv = lambda name: pd.read_csv(io.BytesIO(archive.read(prefix + name)))
        manifest = read_json("manifest.json")
        for name, digest in manifest["file_sha256"].items():
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest, name
        runs = read_csv("arithmetic_runs.csv").set_index("method")
        synthetic = read_csv("synthetic/runs.csv")
        traces = read_csv("arithmetic_traces.csv")
        notebook = json.loads(Path("notebooks/CASR_colab.ipynb").read_text())
        sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        digest = hashlib.sha256("".join(s for s in sources if "SOURCE_SHA256 =" not in s).encode()).hexdigest()
        assert digest == manifest["source_sha256"], "Replay requires the recorded notebook template."
        namespace = dict(globals(), CFG=manifest["config"], DEVICE=torch.device("cpu"), ADAPT_DTYPE=torch.float64)
        # Reuse the trusted, checksum-matched implementation; train no additional model.
        for marker in ["class ArithmeticTransformer", "class PrefixProblem"]:
            source = next(s for s in sources if marker in s)
            exec(compile(source, "recorded-notebook-definitions", "exec"), namespace)
        torch.set_num_threads(2)
        config = manifest["config"]
        model = namespace["ArithmeticTransformer"](config["width"], config["heads"], config["layers"], config["mlp_ratio"]).double().eval()
        load = lambda name: torch.load(io.BytesIO(archive.read(prefix + name)), map_location="cpu", weights_only=True)
        model.load_state_dict(load("arithmetic/seed-0/reference.pt"))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        pairs = read_json("arithmetic/seed-0/splits.json")["test"]
        records = namespace["examples"](pairs, True)
        greedy = namespace["greedy"]
        predictions = {"checkpoint": greedy(records, lambda ids: model(ids))}
        for method in ["casr", "random_casr", "span_adapter"]:
            state = load("arithmetic/seed-0/primary/" + method + ".pt")
            cls = namespace["AdapterProblem" if method == "span_adapter" else "PrefixProblem"]
            problem = cls(model, state["basis"], config["scope"], config["cut"])
            predictions[method] = greedy(records, lambda ids: problem.logits(state["flat"], ids))
            assert np.isclose(namespace["accuracy"](records, predictions[method]), runs.loc[method, "alias_accuracy"], atol=1e-12)
        assert np.isclose(namespace["accuracy"](records, predictions["checkpoint"]), runs.iloc[0]["alias_reference_accuracy"])
        tokens = namespace["TOKENS"]
        rows = [dict(method=method, a=r["a"], b=r["b"], operation=r["op"],
                     prediction=" ".join(tokens[t] for t in prediction),
                     target=" ".join(tokens[t] for t in r["answer"]), correct=prediction == r["answer"])
                for method, preds in predictions.items() for r, prediction in zip(records, preds)]
        replay = pd.DataFrame(rows)
        replay.to_csv(args.output_dir / "replayed_predictions.csv", index=False)

        names = {"ordinary": "Ordinary upstream FT", "activation_only": "Activation-only routing",
                 "casr_no_guard": "CASR without guard", "casr": "CASR", "random_casr": "Random-span CASR",
                 "coordinate_routing": "Coordinate routing", "span_adapter": "Span-output adapter"}
        table = [r"\begin{table}[htbp]\centering\small", r"\setlength{\tabcolsep}{4pt}",
                 r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
                 r"Method & Alias (\%) & Original (\%) & Restore KL & Test drift & Steps\\\midrule",
                 f"Checkpoint & {100*runs.iloc[0].alias_reference_accuracy:.1f} & {100*runs.iloc[0].original_reference_accuracy:.1f} & 0 & 0 & 0" + r"\\"]
        for method, r in runs.iterrows():
            table.append(f"{names[method]} & {100*r.alias_accuracy:.1f} & {100*r.original_accuracy:.1f} & {r.restoration_kl:.2e} & {r.heldout_max_drift:.2e} & {int(r.accepted_steps)}" + r"\\")
        table += [r"\bottomrule\end{tabular}", r"\caption{Completed arithmetic pilot: one checkpoint/data seed, 22 alias test prompts. Test drift is the maximum complete-interface complement norm; the guard budget is 0.05. Restore KL compares checkpoint and restored next-token distributions on matched teacher-forced response positions. Steps are actual accepted updates, not matched computation.}\label{tab:pilot}", r"\end{table}"]
        (args.output_dir / "arithmetic_table.tex").write_text("\n".join(table) + "\n")

        table = [r"\begin{table}[htbp]\centering\small", r"\begin{tabular}{@{}lrrrr@{}}\toprule",
                 r"Method & Affine loss & Affine drift & Curved loss & Curved drift\\\midrule"]
        for method in ["ordinary", "activation_only", "casr_no_guard", "casr", "span_adapter"]:
            means = synthetic[(synthetic.kappa == 2) & (synthetic.method == method)].groupby("gamma").mean(numeric_only=True)
            values = [means.loc[0., "test_loss"], means.loc[0., "test_max_drift"], means.loc[.2, "test_loss"], means.loc[.2, "test_max_drift"]]
            table.append(names[method] + " & " + " & ".join(f"{x:.2e}" for x in values) + r"\\")
        table += [r"\bottomrule\end{tabular}", r"\caption{Completed synthetic pilot at coupling $\kappa=2$. Entries are means over two seeds; drift is the mean of per-run held-out maxima. The prefix is affine or curved, but the reader is nonlinear in both settings. These values are calculated from the supplied run archive.}\label{tab:synthetic}", r"\end{table}"]
        (args.output_dir / "synthetic_table.tex").write_text("\n".join(table) + "\n")

        plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        casr = runs.loc["casr"]
        accepted_drift = traces[(traces.method == "casr") & (traces.accepted == True)].iloc[0].post_active_drift
        drift = [accepted_drift, casr.persistent_anchors_max_drift, casr.last_active_max_drift, casr.heldout_max_drift]
        axes[0].bar(range(4), drift, color=["#24796e", "#24796e", "#b75d3c", "#b75d3c"])
        axes[0].axhline(config["drift_budget"], color="black", linestyle="--", label="Guard budget")
        axes[0].set_xticks(range(4), ["Accepted\nbatch + anchors", "Persistent\nanchors", "Next\nbatch + anchors", "Held-out\naliases"])
        axes[0].set(ylabel="Maximum complement drift", title="Constraint coverage", ylim=(0, .25))
        axes[0].legend(frameon=False, fontsize=9)
        for j, value in enumerate(drift): axes[0].text(j, value + .007, f"{value:.3f}", ha="center", fontsize=9)
        for offset, method, label, color in [(-.18, "checkpoint", "Checkpoint", "#778b9c"), (.18, "casr", "CASR", "#24796e")]:
            scores = replay[replay.method == method].groupby("operation").correct.mean()
            axes[1].bar(np.arange(2) + offset, 100*scores.reindex(["ADD", "SUB"]), width=.36, label=label, color=color)
        axes[1].set_xticks(range(2), ["RULE-A: addition", "RULE-B: subtraction"])
        axes[1].set(ylabel="Exact-match accuracy (%)", title="Cue-specific performance", ylim=(0, 110))
        axes[1].legend(frameon=False, fontsize=9)
        fig.savefig(args.output_dir / "pilot_diagnostics.pdf")
        fig.savefig(args.output_dir / "pilot_diagnostics.png", dpi=170)
        plt.close(fig)
        provenance = dict(archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                          analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          run_id=manifest["run_id"], replay_device="cpu", torch=torch.__version__, numpy=np.__version__,
                          notebook_code_sha256=digest, recorded_aggregate_accuracies_recovered=True,
                          status="post hoc checkpoint replay; no additional training", theory_checks="passed")
        (args.output_dir / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print("Verified archive, theory probes, replayed predictions, and regenerated paper assets:", args.output_dir)


if __name__ == "__main__":
    main()
