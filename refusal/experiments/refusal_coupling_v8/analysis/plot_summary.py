"""Plot the fixed low-rate comparison and benign controls."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

OUT = Path(__file__).resolve().parent
ARMS = ["clean", "global", "local", "lookahead"]
LABELS = ["Clean", "Global coupling", "Local coupling", "Local + lookahead"]
COLORS = ["#374151", "#dc2626", "#2563eb", "#7c3aed"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def main():
    curves = pd.read_csv(OUT / "learning_curves.csv")
    primary = pd.read_csv(OUT / "primary_comparison.csv")
    benign = pd.read_csv(OUT / "benign_comparison.csv")
    assert len(primary) == 8 and len(benign) == 16
    with PdfPages(OUT / "summary_graphs.pdf") as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        for column, parent in enumerate(["shared4", "shared8"]):
            for arm, label, color in zip(ARMS, LABELS, COLORS):
                d = curves[(curves.parent == parent) & (curves.arm == arm)
                           & (curves.objective == "preserve") & (curves.lr == 1e-6)]
                d = d.drop_duplicates("cumulative_updates", keep="last").sort_values("cumulative_updates")
                assert len(d) == 23
                for row, field in enumerate(["allowed_accuracy", "blocked_refusal"]):
                    axes[row, column].plot(d.cumulative_updates, 100*d[field],
                        color=color, label=label, marker=".", markersize=4, lw=1.8)
            axes[0, column].set_title(parent[-1] + "-channel starting model")
            for ax in axes[:, column]:
                ax.axvline(256, color="#64748b", ls=":", lw=1)
                ax.set_xscale("symlog", linthresh=1, base=2)
                ax.set_xlim(0, 400)
                ax.set_ylim(-2, 103)
                ticks = [0, 2, 4, 8, 16, 32, 64, 128, 256, 384]
                ax.set_xticks(ticks, [str(x) for x in ticks], rotation=35)
                ax.grid(axis="y", alpha=.15)
            axes[1, column].set_xlabel("Cumulative SFT updates")
        axes[0, 0].set_ylabel("Allowed navigation correct (%)")
        axes[1, 0].set_ylabel("Blocked navigation refused (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(.5, .93), frameon=False)
        fig.suptitle("Low-rate preserving attack: temporary disruption is not irreversible skill loss", y=.98)
        fig.text(.5, .025, "Development measurements; one seed. Removal LR 1e-6. "
                 "Fresh-data repair starts at 256, also LR 1e-6.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .055, 1, .87))
        pdf.savefig(fig)
        fig.savefig(OUT / "low_rate_trajectories.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        x = list(range(4))
        short = ["Clean", "Global", "Local", "Lookahead"]
        for column, parent in enumerate(["shared4", "shared8"]):
            d = primary[primary.parent == parent].set_index("arm").loc[ARMS]
            ax = axes[0, column]
            ax.bar([v-.18 for v in x], 100*d.allowed_accuracy, .34, label="Allowed correct", color="#2563eb")
            ax.bar([v+.18 for v in x], 100*d.blocked_accuracy, .34, label="Formerly blocked correct", color="#d97706")
            ax.scatter(x, 100*d.baseline_allowed_accuracy, marker="D", s=22, color="#111827", label="Starting allowed")
            ax.set_title(parent[-1] + "-channel start: removal step 256 (0% refusal)")
            ax = axes[1, column]
            a = benign[(benign.parent == parent) & (benign.endpoint == "final")].set_index("arm").loc[ARMS]
            b = benign[(benign.parent == parent) & (benign.endpoint == "repair_final")].set_index("arm").loc[ARMS]
            ax.bar([v-.25 for v in x], 100*a.baseline_blocked_refusal, .24, label="Defended start", color="#94a3b8")
            ax.bar(x, 100*a.blocked_refusal, .24, label="After 256 benign updates", color="#0891b2")
            ax.bar([v+.25 for v in x], 100*b.blocked_refusal, .24, label="After 384 benign updates", color="#7c3aed")
            ax.set_title("Allowed-only continuation: refusal stability")
            for ax in axes[:, column]:
                ax.set_xticks(x, short)
                ax.set_ylim(0, 105)
                ax.grid(axis="y", alpha=.15)
        axes[0, 0].set_ylabel("Held-out navigation correct (%)")
        axes[1, 0].set_ylabel("Held-out blocked refusal (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, .91), frameon=False)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.5, .065), frameon=False)
        fig.suptitle("Held-out outcomes: preserving attacks bypass every arm\n"
                     "The global arm retains more refusal under benign training", y=.98)
        fig.text(.5, .015, "Both panels use LR 1e-6 and one seed. Benign controls never receive blocked-answer targets; "
                 "all 384 updates stay allowed-only.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .10, 1, .86))
        pdf.savefig(fig)
        fig.savefig(OUT / "heldout_summary.png", dpi=170)
        plt.close(fig)
    names = ["FINDINGS.md", "summary.json", "audit.json", "defended_test_baselines.csv", "retention_by_task.csv",
             "temporal_outcomes.csv", "primary_comparison.csv", "benign_comparison.csv",
             "selection_mechanism.csv", "summary_graphs.pdf", "low_rate_trajectories.png", "heldout_summary.png"]
    inputs = ["provenance.json", "endpoints.csv", "learning_curves.csv", "training.csv"]
    checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = dict(inputs={n: checksum(OUT / n) for n in inputs},
        scripts={n: checksum(OUT / n) for n in ["summarise.py", "plot_summary.py"]},
        outputs={n: checksum(OUT / n) for n in names}, model_runs=0)
    (OUT / "final_analysis_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Saved two-page summary and derived-artifact hashes.")


if __name__ == "__main__":
    main()
