"""Run the unchanged adaptive protocol on all six Ogg GPUs."""
import argparse
import json
from pathlib import Path
import signal
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "source"))
    import run_aware as study
    import scheduler

    output = root / "outputs/main"
    plan = json.loads((output / "experiment_plan.json").read_text())
    assert plan == study.make_plan(False), "Scientific protocol or source changed"
    smoke = json.loads((root / "outputs/smoke/SHARD_COMPLETE.json").read_text())
    assert smoke["completed"] == 6 and not smoke["failed"]
    assert smoke["execution_revision"] == study.execution_revision(output, plan)["id"]
    assigned = [job["id"] for job in plan["jobs"]]
    devices = [str(i) for i in range(6)]
    assert len(assigned) == len(set(assigned)) == 48
    completed = sum(study.validate_completed(output, job, plan) for job in plan["jobs"])
    print(f"Verified unchanged plan: {completed}/48 completed; GPUs {devices}", flush=True)
    if args.check:
        return

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with study.exclusive(output / "launcher.lock"):
        study.atomic(root / "runtime_allocation.json", dict(gpus=devices,
            plan_sha256=study.digest(plan), scientific_protocol_unchanged=True,
            reason="User paused refusal research and assigned all resources to AAAI capabilities"))
        scheduler.supervise(study, plan, output, assigned, devices)


if __name__ == "__main__":
    main()
