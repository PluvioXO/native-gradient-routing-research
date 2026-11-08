"""Resume V8 with allocator-only recovery and the original frozen stages."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

JOB = "shared8__global__benign__lr1e-06"
ALLOCATOR = "expandable_segments:True"


def same_updates(actual, expected):
    """Check replayed measurements, excluding elapsed wall time."""
    return all(
        {k: v for k, v in a.items() if k != "seconds"}
        == {k: v for k, v in b.items() if k != "seconds"}
        for a, b in zip(actual, expected)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    source = root.parent / "source"
    ops = root.parent / "operations"
    ops.mkdir(exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ALLOCATOR
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(source))
    import experiment as ex

    manifest = ex.check(root)
    selection = ex.read(root / "selection.json")
    assert selection["manifest"] == ex.digest(manifest)
    assert len(selection["jobs"]) == 40
    run_id = ex.digest(manifest)
    state_path = ops / "recovery_status.json"

    def state(stage, **values):
        ex.write(state_path, dict(stage=stage, updated_unix=time.time(),
                                 run_id=run_id, allocator=ALLOCATOR, **values))

    if args.probe:
        receipt = ops / "allocator_probe.json"
        folder = root / "attacks" / JOB
        if (folder / "DONE.json").exists():
            ex.write(receipt, dict(passed=True, already_complete=True, run_id=run_id))
            return
        baseline = ops / "pre_recovery_history.json"
        if not baseline.exists():
            shutil.copy2(folder / "history.json", baseline)
        expected = ex.read(baseline)
        assert len(expected) == 259
        original_write, original_save = ex.write, ex.save
        replayed = 0

        def checked_write(path, value):
            nonlocal replayed
            if Path(path) == folder / "history.json":
                assert same_updates(value, expected), "Replay measurements changed"
                replayed = max(replayed, min(len(value), len(expected)) - 256)
            original_write(path, value)

        def stop_at_checkpoint(path, value, needed):
            original_save(path, value, needed)
            if (Path(path) == folder / "resume.pt"
                    and value["phase"] == "repair" and value["step"] >= 16):
                assert replayed == 3
                assert ex.check(root)["sources"] == manifest["sources"]
                import torch
                original_write(receipt, dict(
                    passed=True, run_id=run_id, allocator=ALLOCATOR,
                    replayed_repair_updates_checked=replayed,
                    existing_parameter_hash_checks="original observe() checks at removal 256 and repair 0/1/2",
                    repair_checkpoint_step=value["step"],
                    cuda_peak=[torch.cuda.max_memory_allocated(i)
                               for i in range(torch.cuda.device_count())],
                    frozen_sources_unchanged=True,
                ))
                raise SystemExit(0)

        assert {0, 1}.issubset(set(ex.api.available_gpus([0, 1])))
        ex.write, ex.save = checked_write, stop_at_checkpoint
        ex.attack(root, JOB)
        raise RuntimeError("Probe ended without its checkpoint receipt")

    lock = (ops / "recovery.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ["status.json", "selection.json"]:
        backup = ops / ("before_recovery_" + name)
        if not backup.exists():
            shutil.copy2(root / name, backup)
    hardware = ex.read(root.parent / "hardware/hardware_validation.json")
    assert hardware["sources"] == manifest["sources"] and hardware["actual_resume_exact"]

    def execute(stage, command, gpu_pair=None):
        state(stage, command=command)
        environment = dict(os.environ)
        if gpu_pair is not None:
            environment["CUDA_VISIBLE_DEVICES"] = gpu_pair
        with (ops / (stage + ".log")).open("a") as log:
            subprocess.run(command, cwd=source, env=environment,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                           check=True)

    try:
        execute("allocator_probe",
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--root", str(root), "--probe"], "0,1")
        probe = ex.read(ops / "allocator_probe.json")
        assert probe["passed"] and probe["run_id"] == run_id
        execute("attack", [sys.executable, "-u", str(source / "experiment.py"),
                          "--root", str(root), "--phase", "attack"])
        assert all((root / "attacks" / job / "DONE.json").exists()
                   for job in selection["jobs"])
        execute("heldout", [sys.executable, "-u", str(source / "experiment.py"),
                           "--root", str(root), "--phase", "heldout"])
        assert all((root / "attacks" / job / "TEST_DONE.json").exists()
                   for job in selection["jobs"])
        assert ex.read(root / "COMPLETE.json")["jobs"] == 40
        execute("analysis", [sys.executable, "-u", str(source / "analyse.py"),
                            "--root", str(root)])
        analysis = root.parent / "analysis"
        provenance = ex.read(analysis / "provenance.json")
        assert all(ex.sha(analysis / name) == value
                   for name, value in provenance["files"].items())
        with (analysis / "endpoints.csv").open() as stream:
            endpoints = list(csv.DictReader(stream))
        assert {row["job"] for row in endpoints} == set(selection["jobs"])
        assert (analysis / "coupling_trajectories.pdf").stat().st_size > 0
        state("complete", completed_fits=40, heldout_jobs=40,
              heldout_endpoints=len(endpoints), analysis_verified=True,
              frozen_sources_unchanged=ex.check(root)["sources"] == manifest["sources"])
    except Exception as error:
        previous = ex.read(state_path)
        state("failed", failed_stage=previous["stage"], error=repr(error))
        raise


if __name__ == "__main__":
    main()
