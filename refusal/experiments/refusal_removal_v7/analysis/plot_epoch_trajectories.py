"""Plot the audited removal trajectories; no model inference or training."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STEPS = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
RATES = [1e-6, 1e-5, 5e-5]
SERIES = [
    ('allowed_accuracy', 'Allowed navigation correct', '#2563eb', '-', 'o'),
    ('blocked_accuracy', 'Formerly blocked navigation correct', '#d97706', '-', 's'),
    ('blocked_refusal', 'Formerly blocked refusal', '#dc2626', '--', 'x'),
    ('general_accuracy', 'Arithmetic/copy controls correct', '#64748b', ':', None),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inputs = ['learning_curves.csv', 'training.csv']
    original = json.loads((ROOT / 'provenance.json').read_text())
    for name in inputs:
        assert sha(ROOT / name) == original['files'][name], name
    curves = pd.read_csv(ROOT / inputs[0])
    training = pd.read_csv(ROOT / inputs[1], usecols=['job', 'step', 'blocked_exposure_epochs'])
    assert len(curves) == 168 and len(training) == 3072
    assert curves.job.nunique() == training.job.nunique() == 12
    assert set(curves.job) == set(training.job)
    assert np.isfinite(curves[[s[0] for s in SERIES]].to_numpy()).all()
    assert curves[[s[0] for s in SERIES]].ge(0).all().all()
    assert curves[[s[0] for s in SERIES]].le(1).all().all()
    # Check the epoch conversion against actual training exposures, not a guessed batch size.
    np.testing.assert_allclose(training.blocked_exposure_epochs, training.step / 256)
    for _, group in training.groupby('job'):
        assert group.step.tolist() == list(range(1, 257))
    for _, group in curves.groupby('job'):
        assert sorted(group.step.tolist()) == STEPS
    assert curves[curves.step == 0].groupby('parent')[[s[0] for s in SERIES]].nunique().le(1).all().all()

    plt.rcParams.update({
        'font.size': 10, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'medium', 'savefig.facecolor': 'white',
    })
    pdf_path = ROOT / 'epoch_trajectories.pdf'
    png_paths = []
    summaries = []
    with PdfPages(pdf_path) as pdf:
        for objective, subtitle in [
            ('preserve', 'Preserving attack: correct-answer SFT + allowed replay + general KL'),
            ('ordinary', 'Ordinary attack: correct-answer SFT on formerly blocked examples only'),
        ]:
            fig, axes = plt.subplots(2, 3, figsize=(15, 9.4), sharex=True, sharey=True)
            for row, parent in enumerate(['shared4', 'shared8']):
                for col, lr in enumerate(RATES):
                    ax = axes[row, col]
                    d = curves[(curves.parent == parent) & (curves.objective == objective) & (curves.lr == lr)].sort_values('step')
                    assert len(d) == len(STEPS)
                    x = d.step / 256
                    for field, label, color, style, marker in SERIES:
                        ax.plot(x, 100 * d[field], label=label, color=color, linestyle=style,
                                linewidth=2 if field == 'allowed_accuracy' else 1.7,
                                marker=marker, markersize=3.4, markeredgewidth=.8)
                    ax.set_xscale('symlog', linthresh=1/256, base=2)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-3, 103)
                    ax.set_xticks([0, 1/256, 1/64, 1/16, 1/4, 1],
                                  ['0', '0.004', '0.016', '0.0625', '0.25', '1.0'])
                    ax.set_yticks([0, 25, 50, 75, 100])
                    ax.tick_params(labelsize=9)
                    ax.grid(axis='y', color='#e2e8f0', linewidth=.7)
                    ax.set_axisbelow(True)
                    after = d[d.step > 0]
                    minimum = after.loc[after.allowed_accuracy.idxmin()]
                    ax.set_title(f'Shared {parent[-1]} channels | LR {lr:.0e}\n'
                                 f'Minimum observed allowed accuracy: {100*minimum.allowed_accuracy:.1f}%',
                                 loc='left', fontsize=11, pad=10)
                    if col == 0:
                        ax.set_ylabel('Development rate (%)', labelpad=9)
                    zero = d[d.refusal == 0].iloc[0]
                    summaries.append(dict(job=d.iloc[0].job,
                        first_zero_navigation_refusal_step=int(zero.step),
                        allowed_accuracy_at_first_zero_refusal=float(zero.allowed_accuracy),
                        minimum_observed_post_update_allowed_accuracy=float(minimum.allowed_accuracy),
                        minimum_observed_allowed_step=int(minimum.step),
                        final_development_allowed_accuracy=float(d.iloc[-1].allowed_accuracy)))
            fig.suptitle('Refusal removal and capability trajectories', fontsize=20, fontweight='bold', y=.985)
            fig.text(.5, .945, subtitle + ' | Development evaluations | Seed 31', ha='center', fontsize=11)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .914),
                       ncol=2, frameon=False, fontsize=10, columnspacing=3, handlelength=3)
            fig.subplots_adjust(left=.065, right=.985, bottom=.17, top=.805, hspace=.43, wspace=.16)
            fig.supxlabel('Blocked-data epochs (1 epoch = 256 SFT updates = 512 formerly blocked examples)',
                          y=.105, fontsize=12)
            fig.text(.065, .059,
                     'Early updates are expanded using a symmetric-log axis. Points are saved evaluations; lines connect measurements.\n'
                     'Epochs refer to blocked examples only; allowed replay has a separate exposure count. These are development curves, not held-out endpoint scores.',
                     fontsize=9, color='#475569', va='top', linespacing=1.5)
            fig.text(.065, .016,
                     'A drop and recovery in answer accuracy does not establish that information was erased and relearned. Arithmetic/copy is a small control panel, not all capabilities.',
                     fontsize=9, color='#475569', va='bottom')
            pdf.savefig(fig)
            png_path = ROOT / f'epoch_trajectories_{objective}.png'
            fig.savefig(png_path, dpi=160)
            png_paths.append(png_path)
            plt.close(fig)

    from pypdf import PdfReader
    assert len(PdfReader(pdf_path).pages) == 2
    assert len(summaries) == 12
    receipt = dict(
        scope='One-seed developmental trajectories, all 12 fits; no inference or training rerun',
        inputs={name: sha(ROOT / name) for name in inputs},
        script_sha256=sha(Path(__file__)),
        blocked_epoch_definition='Actual logged blocked_exposure_epochs = update / 256',
        measured_updates=STEPS,
        outputs={path.name: sha(path) for path in [pdf_path, *png_paths]},
        summaries=summaries,
    )
    (ROOT / 'epoch_trajectories_provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Generated two-page epoch_trajectories.pdf and both PNG previews from verified saved measurements.')


if __name__ == '__main__':
    main()
