#!/usr/bin/env python3
"""Run one disjoint assignment using the unchanged, pinned composition workers."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def check_assignment(record, plan, study):
    assert record["plan_sha256"] == study.digest(plan)
    groups = record["assignments"]
    ids = [key for jobs in groups.values() for key in jobs]
    expected = {j["id"] for j in plan["jobs"] if j["kind"] in ("main", "merge")}
    assert len(ids) == len(set(ids)) and set(ids) == expected, "Incomplete or duplicate assignment"
    for jobs in groups.values():
        for job in plan["jobs"]:
            if job["id"] in jobs:
                assert set(job.get("dependencies", [])) <= set(jobs), "Merge dependencies on another host"


def supervise(study, plan, output, assigned, devices, interval=10):
    jobs = {j["id"]: j for j in plan["jobs"]}
    validated = set()
    def done(key):
        if key not in validated and study.validate_completed(output, jobs[key], plan):
            validated.add(key)
        return key in validated
    for job in plan["jobs"]:
        if job["kind"] in ("baseline", "pilot"):
            assert done(job["id"]), "Copy completed baseline and pilots before launch"
    pending = [key for key in assigned if not done(key)]
    for key in pending:
        with study.exclusive(output / "fits" / key / "worker.lock"):
            pass
    processes, attempts, failed = {}, {}, []
    available = list(devices)
    report = {}
    try:
        while True:
            if pending or processes:
                study.checked_space(output)
            ready = [key for key in pending if all(done(dep) for dep in jobs[key].get("dependencies", []))]
            for key in ready[:len(available)]:
                device = available.pop(0)
                pending.remove(key)
                log = output / "logs" / f"{key}.log"
                log.parent.mkdir(exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                with log.open("a") as stream:
                    process = subprocess.Popen([sys.executable, "-u", str(study.ROOT / "run_study.py"),
                        "worker", "--output", str(output), "--job", key], env=env,
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                processes[key] = process, device
                attempts[key] = attempts.get(key, 0) + 1
                print("START", key, "GPU", device, "PID", process.pid, flush=True)
            for key, (process, device) in list(processes.items()):
                code = process.poll()
                if code is None:
                    continue
                del processes[key]
                available.append(device)
                if code == 75:
                    raise RuntimeError(f"Another worker owns {key}")
                if code == 0 and done(key):
                    print("DONE", key, flush=True)
                elif attempts[key] < 3:
                    pending.append(key)
                    print("RETRY", key, code, flush=True)
                else:
                    failed.append(dict(id=key, exit=code))
                    print("FAILED", key, code, flush=True)
            active = []
            for key, (process, device) in processes.items():
                path = output / "fits" / key / "durable/fit/resume.json"
                state = json.loads(path.read_text()) if path.exists() else {}
                active.append(dict(id=key, pid=process.pid, gpu=device, step=state.get("step", 0),
                                   total=jobs[key].get("steps")))
            report = dict(host=socket.gethostname(), updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                planned=len(assigned), completed=sum(done(key) for key in assigned), active=active,
                pending=len(pending), failed=failed, launcher_state="running",
                by_kind={kind:dict(done=sum(done(key) for key in assigned if jobs[key]["kind"] == kind),
                    total=sum(jobs[key]["kind"] == kind for key in assigned)) for kind in sorted({jobs[key]["kind"] for key in assigned})},
                execution_revision=study.execution_revision(output, plan)["id"],
                free_gib=study.shutil.disk_usage(output).free / 2**30)
            study.atomic(output / "shard_status.json", report)
            if not pending and not processes:
                if failed:
                    raise RuntimeError(f"Failed jobs: {failed}")
                break
            if pending and not processes and not any(all(done(dep) for dep in jobs[key].get("dependencies", [])) for key in pending):
                raise RuntimeError("No runnable jobs: failed dependencies")
            time.sleep(interval)
        report["launcher_state"] = "complete"
        study.atomic(output / "shard_status.json", report)
        study.atomic(output / "SHARD_COMPLETE.json", report)
    except BaseException as error:
        report.update(launcher_state="stopped", stop_reason=f"{type(error).__name__}: {error}")
        study.atomic(output / "shard_status.json", report)
        raise
    finally:
        for process, _ in processes.values():
            process.terminate()
        for process, _ in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.study.resolve()))
    import run_study as study
    output = args.output.resolve()
    plan = json.loads((output / "experiment_plan.json").read_text())
    study.execution_revision(output, plan, study.make_plan()["source_hashes"])
    record = json.loads(args.assignment.read_text())
    check_assignment(record, plan, study)
    assigned = record["assignments"][socket.gethostname()]
    devices = args.gpus.split(",")
    assert devices and len(set(devices)) == len(devices) and all(d.isdigit() for d in devices)
    if args.check:
        for job in plan["jobs"]:
            if job["id"] in assigned or job["kind"] in ("baseline", "pilot"):
                study.validate_completed(output, job, plan)
        print("Assignment and existing completions valid:", socket.gethostname(), len(assigned))
        return
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    with study.exclusive(output / "launcher.lock"):
        supervise(study, plan, output, assigned, devices)


if __name__ == "__main__":
    main()
