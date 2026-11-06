"""Fixed-budget refusal allocation across one, two, or four layers; three-GPU workers."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import removal as r


LAYOUTS = {"single12": (4, (11,)), "dual08_16": (2, (7, 15)),
           "four03_09_15_21": (1, (2, 8, 14, 20))}


def config(condition, seed):
    width, layers = LAYOUTS[condition]
    return r.Config(width=width, layers=len(layers), layer_indices=layers,
        attention=False, seeds=(seed,), arms=("channels",), multiple_wordings=True,
        two_gpu=True, model_gpu_count=3, removal_lrs=(1e-6, 1e-5),
        removal_steps=tuple(range(17)) + tuple(range(20, 65, 4)),
        eval_batch=4, checkpoint_every=10)


def jobs():
    return [dict(name=f"{condition}_s{seed}", condition=condition, seed=seed)
        for seed in (31, 47, 71) for condition in LAYOUTS]


def smoke(root):
    cfg = config("single12", 31)
    tok, data, fingerprint = r.initialise(cfg, root)
    model = r.load_model(cfg, "cuda")
    development = r.evaluate(model, tok, data["dev"], cfg)
    r.core.atomic_json(root / "baseline_development.json", development)
    by_wording = {w: r.metrics([x for x in development["rows"] if x["wording"] == w]) for w in range(3)}
    print("BASELINE BY WORDING", by_wording, flush=True)
    if not r.baseline_ok(development, cfg):
        raise RuntimeError("Baseline failed; no study launched. Inspect baseline_development.json")
    baseline = r.evaluation_panels(model, tok, data, cfg)
    baseline["dev"] = development
    r.core.atomic_json(root / "baseline.json", baseline)
    layout = r.allocate(model, cfg, "channels", 31)
    assert layout["trainable"] == 4 * 896 * 3
    assert all(m["name"].startswith("model.layers.11.mlp.") for m in layout["matrices"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = r.torch.optim.AdamW(trainable, lr=3e-4, weight_decay=0, foreach=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoded = [r.core.encode_example(tok, x, cfg) for x in data["train"]]
    sparse = r.train_update(model, tok, encoded, cfg, opt, 31, 0)
    assert any(p.detach().abs().max() > 0 for p in trainable)
    del opt, trainable
    r.merge_native(model)
    opt = r.torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0, foreach=False)
    encoded = [r.core.encode_example(tok, x, cfg) for x in data["removal"]]
    full = [r.train_update(model, tok, encoded, cfg, opt, 9031, i) for i in range(2)]
    opt.zero_grad(set_to_none=True)
    r.evaluation_panels(model, tok, {**data, "test": data["test"][:8], "transfer": data["transfer"][:8],
        "utility": data["utility"][:4], "benchmarks": data["benchmarks"][:4]}, cfg)
    r.checkpoint(root / "smoke.pt", model, opt, 2, full, fingerprint)
    assert r.resume(root / "smoke.pt", model, opt, fingerprint)[0] == 2
    (root / "smoke.pt").unlink()
    result = dict(passed=True, device_map=model.hf_device_map, sparse=sparse, full=full,
        peak_allocated_bytes=[r.torch.cuda.max_memory_allocated(i) for i in range(cfg.model_gpu_count)],
        by_wording=by_wording, torch=r.torch.__version__, config=asdict(cfg))
    r.core.atomic_json(root / "smoke_pass.json", result)
    print("SMOKE PASSED", result, flush=True)


def worker(root, name):
    job = next(j for j in jobs() if j["name"] == name)
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = config(job["condition"], job["seed"])
    _, data, _ = r.initialise(cfg, out)
    # The base is independent of allocation; reuse only exact inputs, weights, and source.
    reference = json.loads((root / "preflight" / "manifest.json").read_text())
    current = json.loads((out / "manifest.json").read_text())
    assert current["data_sha256"] == reference["data_sha256"]
    assert current["sources"] == reference["sources"]
    for key in ("model", "revision", "max_prompt", "max_new_tokens", "eval_batch", "two_gpu", "model_gpu_count"):
        assert current["config"][key] == reference["config"][key]
    if not (out / "baseline.json").exists():
        shutil.copy2(root / "preflight" / "baseline.json", out / "baseline.json")
        r.core.atomic_json(out / "baseline_provenance.json", dict(source=str(root / "preflight"),
            data_sha256=current["data_sha256"]))
    r.run(cfg, out)


def launch(root):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((root / "preflight" / "smoke_pass.json").read_text())["passed"]
    plan = dict(jobs=jobs(), configurations={j["name"]: asdict(config(j["condition"], j["seed"])) for j in jobs()},
        gpu_pairs=[[3, 4, 5]], baseline_source="preflight/baseline.json",
        comparison="Removal resistance only for matched, qualified primary AND unseen-wording starts",
        note="Artificial blue-category refusal; acquisition failure is a result. No real safety claim.")
    if (root / "plan.json").exists():
        assert json.loads((root / "plan.json").read_text()) == json.loads(json.dumps(plan))
    r.core.atomic_json(root / "plan.json", plan)
    (root / "logs").mkdir(exist_ok=True)
    pending, done, failed, active = [], [], [], {}
    for job in jobs():
        path = root / "runs" / job["name"] / "status.json"
        if path.exists() and json.loads(path.read_text())["status"] in ("complete", "inconclusive"):
            done.append(job["name"])
        else:
            pending.append(job)
    def stop(*_):
        for proc, _, _ in active.values():
            proc.terminate()
        for proc, _, _ in active.values():
            proc.wait()
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while pending or active:
        for slot, (proc, job, log) in list(active.items()):
            result = proc.poll()
            if result is not None:
                log.close()
                (done if result == 0 else failed).append(job["name"])
                print("FINISHED", job["name"], "exit", result, flush=True)
                del active[slot]
        for slot, pair in enumerate(plan["gpu_pairs"]):
            if slot in active or not pending:
                continue
            if shutil.disk_usage(root).free < 60 * 1024**3:
                raise OSError("Less than 60 GiB free: queued jobs stopped; existing worker checkpoints retained")
            job = pending.pop(0)
            log = (root / "logs" / (job["name"] + ".log")).open("a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, pair)), OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", MPLBACKEND="Agg")
            proc = subprocess.Popen([sys.executable, "-u", __file__, "worker", "--root", str(root),
                "--job", job["name"]], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active[slot] = proc, job, log
            print("STARTED", job["name"], "GPUs", pair, "PID", proc.pid, flush=True)
        r.core.atomic_json(root / "progress.json", dict(updated=time.time(), total=len(jobs()), finished=done,
            failed=failed, pending=[j["name"] for j in pending],
            running=[dict(job=j["name"], pid=p.pid, gpus=plan["gpu_pairs"][slot]) for slot, (p,j,_) in active.items()]))
        if active:
            time.sleep(10)
    print("STUDY FINISHED", len(done), "finished", len(failed), "failed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "worker", "launch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.mode == "smoke":
        smoke(args.root / "preflight")
    elif args.mode == "worker":
        worker(args.root, args.job)
    else:
        launch(args.root)
