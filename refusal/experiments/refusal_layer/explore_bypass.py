"""Locate removal edits using immutable learned checkpoints from the layer study."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import removal as r


def restrict(model, layer, scope):
    if scope not in ("all", "outside_layer", "inside_layer", "upstream", "downstream"):
        raise ValueError(scope)
    prefix = f"model.layers.{layer}."
    selected = [n for n, _ in model.named_parameters() if n.startswith(prefix)]
    if not selected:
        raise ValueError("Selected block has no parameters")
    adjacent = None
    if scope in ("upstream", "downstream"):
        if layer < 2 or layer + 2 >= len(model.model.layers):
            raise ValueError("Matched two-block comparison requires two blocks on both sides")
        sides = ((layer - 2, layer - 1), (layer + 1, layer + 2))
        counts = [sum(p.numel() for n,p in model.named_parameters()
            if n.startswith(tuple(f"model.layers.{i}." for i in side))) for side in sides]
        if counts[0] != counts[1]:
            raise ValueError("Upstream and downstream parameter counts differ")
        adjacent = sides[0 if scope == "upstream" else 1]
    for name, parameter in model.named_parameters():
        inside = name.startswith(prefix)
        enabled = (name.startswith(tuple(f"model.layers.{i}." for i in adjacent)) if adjacent is not None
            else scope == "all" or (inside if scope == "inside_layer" else not inside))
        parameter.requires_grad_(enabled)
    return dict(scope=scope, layer=layer,
        trainable_blocks=adjacent,
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))


def frozen_hash(model):
    return r.core.tensor_hash([(n, p) for n, p in model.named_parameters() if not p.requires_grad])


def inputs(parent, name):
    source = parent / "runs" / name
    manifest = json.loads((source / "manifest.json").read_text())
    for file, expected in manifest["sources"].items():
        if hashlib.sha256(Path(__file__).with_name(file).read_bytes()).hexdigest() != expected:
            raise ValueError("Parent implementation differs: " + file)
    cfg = r.Config(**manifest["config"])
    assert cfg.arms == ["channels"] and len(cfg.layer_indices) == 1
    directory = source / f"seed_{cfg.seeds[0]}" / "channels"
    assert json.loads((directory / "acquisition.json").read_text())["qualified"]
    data = json.loads((source / "data.json").read_text())
    assert r.core.digest(data) == manifest["data_sha256"]
    return source, directory, manifest, cfg, data


def jobs(parent, study="bypass", reference=None):
    if study == "position_long" and reference is None:
        raise ValueError("The extended study requires the completed position study")
    result = []
    for source in sorted((parent / "runs").glob("layer*")):
        acquisition = next(source.glob("seed_*/channels/acquisition.json"))
        if not json.loads(acquisition.read_text())["qualified"]:
            continue
        for lr in (1e-6, 1e-5):
            for scope in (("upstream", "downstream") if study.startswith("position") else ("outside_layer", "inside_layer")):
                result.append(dict(name=f"{source.name}__{scope}__lr{lr:g}", source=source.name, scope=scope, lr=lr))
    if study == "position_long":
        result = [dict(j, reference=str(reference / "runs" / j["name"])) for j in result
                  if j["source"].startswith("layer22_") and j["lr"] == 1e-6]
    return result


def compare_metrics(actual, expected):
    if actual.keys() != expected.keys():
        raise ValueError("Replay panels differ")
    for panel in actual:
        a, b = actual[panel]["metrics"], expected[panel]["metrics"]
        if a.keys() != b.keys():
            raise ValueError("Replay metric names differ")
        for key, value in b.items():
            if not math.isclose(a[key], value, rel_tol=1e-4, abs_tol=1e-5):
                raise ValueError(f"Replay differs: {panel}/{key}: {a[key]} != {value}")


def run_job(parent, out, job, steps=8, replay=False):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source, directory, manifest, cfg, data = inputs(parent, job["source"])
    saved = r.core.load_pt(directory / "learned.pt")
    assert saved["fingerprint"] == r.core.digest(manifest)
    spec = dict(job=job, steps=steps, parent=str(source), parent_fingerprint=saved["fingerprint"],
        learned_sha256=hashlib.sha256((directory / "learned.pt").read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note=("Two adjacent blocks; equal upstream/downstream parameter counts; embeddings and head frozen"
            if job["scope"] in ("upstream", "downstream") else
            "Same examples, optimizer and rates; trainable parameter counts differ by design"))
    reference = Path(job["reference"]) if job.get("reference") else None
    if reference:
        previous = json.loads((reference / "manifest.json").read_text())
        completed = json.loads((reference / "complete.json").read_text())
        assert completed["final_step"] == 8 and completed["frozen_parameters_unchanged"]
        assert completed["fingerprint"] == r.core.digest(previous)
        assert all(previous["job"][key] == job[key] for key in ("source", "scope", "lr"))
        assert previous["learned_sha256"] == spec["learned_sha256"]
        spec["reference_fingerprint"] = completed["fingerprint"]
    fingerprint = r.core.digest(spec)
    if (out / "manifest.json").exists():
        assert json.loads((out / "manifest.json").read_text()) == spec
    r.core.atomic_json(out / "manifest.json", spec)
    if (out / "complete.json").exists():
        assert json.loads((out / "complete.json").read_text())["fingerprint"] == fingerprint
        return
    tok = r.AutoTokenizer.from_pretrained(str(source / "tokenizer"))
    tok.padding_side = "left"
    model = r.load_model(cfg, "cuda")
    layout = r.allocate(model, cfg, "channels", cfg.seeds[0])
    assert layout == saved["layout"]
    r.load_parameters(model, saved["parameters"])
    del saved
    r.merge_native(model)
    allocation = restrict(model, cfg.layer_indices[0], job["scope"])
    unchanged = frozen_hash(model)
    r.core.atomic_json(out / "allocation.json", dict(allocation, frozen_sha256=unchanged))
    opt = r.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=job["lr"], weight_decay=0, foreach=False)
    start, history = r.resume(out / "latest.pt", model, opt, fingerprint)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    encoded = [r.core.encode_example(tok, row, cfg) for row in data["removal"]]
    assert all(row is not None for row in encoded)
    before = json.loads((directory / "before_removal.json").read_text())
    for step in range(start, steps + 1):
        opt.zero_grad(set_to_none=True)
        evaluation = out / f"eval_{step:04d}.json"
        if not evaluation.exists():
            panels = before if step == 0 else r.evaluation_panels(model, tok, data, cfg)
            if reference and step <= 8:
                expected = json.loads((reference / f"eval_{step:04d}.json").read_text())
                compare_metrics(panels, expected["panels"])
                print("PREFIX REPLAY MATCHED", job["name"], step, flush=True)
            if replay and step:
                old = json.loads((directory / f"removal_lr_{job['lr']:g}" / f"eval_{step:04d}.json").read_text())["panels"]
                for panel in panels:
                    for key, expected in old[panel]["metrics"].items():
                        assert math.isclose(panels[panel]["metrics"][key], expected, rel_tol=1e-4, abs_tol=1e-5), (panel, key)
                print("UNRESTRICTED REPLAY MATCHED", flush=True)
            r.core.atomic_json(evaluation, dict(step=step, panels=panels, examples_seen=step * cfg.accumulation,
                epochs=step * cfg.accumulation / len(data["removal"]), tokens_seen=sum(h["processed_tokens"] for h in history)))
            print(job["name"], "step", step, "refusal", panels["test"]["metrics"]["blocked_refusal"],
                "correct", panels["test"]["metrics"]["blocked_correct"], flush=True)
        if step == steps:
            break
        history.append(r.train_update(model, tok, encoded, cfg, opt, cfg.seeds[0] + 9000, step))
        if (step + 1) % 4 == 0 or step + 1 == steps:
            r.checkpoint(out / "latest.pt", model, opt, step + 1, history, fingerprint)
    assert frozen_hash(model) == unchanged, "Frozen parameters changed"
    r.core.atomic_json(out / "training.json", history)
    r.core.atomic_torch(out / "final.pt", dict(parameters=r.parameter_state(model), fingerprint=fingerprint,
        parent=str(directory / "learned.pt"), scope=job["scope"]))
    r.core.atomic_json(out / "complete.json", dict(fingerprint=fingerprint, final_step=steps,
        frozen_parameters_unchanged=True, peak_allocated_bytes=[r.torch.cuda.max_memory_allocated(i) for i in range(2)]))
    if not reference:
        (out / "latest.pt").unlink(missing_ok=True)
    del model, opt
    r.cleanup(None)


def analyse(root, parent, study="bypass", reference=None):
    records = []
    for source in sorted((parent / "runs").glob("layer*")):
        for path in source.glob("seed_*/channels/removal_lr_*/eval_*.json"):
            value = json.loads(path.read_text())
            if value["step"] <= 8:
                for panel, report in value["panels"].items():
                    records.append(dict(source=source.name, scope="all", lr=value["lr"], step=value["step"],
                        panel=panel, **report["metrics"]))
    for job in jobs(parent, study, reference):
        for path in (root / "runs" / job["name"]).glob("eval_*.json"):
            value = json.loads(path.read_text())
            for panel, report in value["panels"].items():
                records.append(dict(source=job["source"], scope=job["scope"], lr=job["lr"], step=value["step"],
                    panel=panel, **report["metrics"]))
    r.core.atomic_json(root / "comparison.json", records)


def launch(root, parent, study="bypass", reference=None):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((root / "smoke" / "complete.json").read_text())["final_step"] == 1
    if study.startswith("position"):
        for scope in ("upstream", "downstream"):
            completed = json.loads((root / ("smoke_" + scope) / "complete.json").read_text())
            assert completed["final_step"] == 1 and completed["frozen_parameters_unchanged"]
    queue = jobs(parent, study, reference)
    total, steps = (6, 64) if study == "position_long" else (32, 8)
    assert len(queue) == total
    plan = dict(parent=str(parent), study=study, jobs=queue, steps=steps, evaluation_steps=list(range(steps + 1)),
        reference="Existing unrestricted trajectories, with first-step numerical replay validated",
        question=("Can two preceding versus two following blocks bypass the frozen refusal block?"
            if study.startswith("position") else "Can other layers bypass refusal while the entire trained block remains bitwise frozen?"),
        comparison=("Equal trainable scalar counts and update/data budgets; runtime not matched"
            if study.startswith("position") else "Mechanism intervention; not parameter-count or compute matched"))
    if (root / "plan.json").exists():
        assert json.loads((root / "plan.json").read_text()) == plan
    r.core.atomic_json(root / "plan.json", plan)
    (root / "logs").mkdir(exist_ok=True)
    done = [j["name"] for j in queue if (root / "runs" / j["name"] / "complete.json").exists()]
    queue = [j for j in queue if j["name"] not in done]
    failed, active = [], {}
    pairs = [(0, 1), (2, 3), (4, 5)]
    def stop(*_):
        for proc, _, _ in active.values():
            proc.terminate()
        for proc, _, _ in active.values():
            proc.wait()
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while queue or active:
        for slot, (proc, job, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                log.close()
                (done if code == 0 else failed).append(job["name"])
                del active[slot]
                print("FINISHED", job["name"], code, flush=True)
        for slot, pair in enumerate(pairs):
            if slot in active or not queue:
                continue
            if shutil.disk_usage(root).free < 60 * 1024**3:
                stop()
            job = queue.pop(0)
            log = (root / "logs" / (job["name"] + ".log")).open("a")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, pair)))
            command = [sys.executable, "-u", __file__, "worker", "--root", str(root),
                "--parent", str(parent), "--study", study, "--job", job["name"]]
            if reference:
                command += ["--reference", str(reference)]
            proc = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            active[slot] = proc, job, log
            print("STARTED", job["name"], proc.pid, pair, flush=True)
        r.core.atomic_json(root / "progress.json", dict(total=total, finished=done, failed=failed,
            pending=[j["name"] for j in queue], running=[dict(job=j["name"], pid=p.pid, gpus=pairs[s])
                for s, (p,j,_) in active.items()], updated=time.time()))
        if active:
            time.sleep(10)
    analyse(root, parent, study, reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "launch", "worker", "analyse"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--study", choices=("bypass", "position", "position_long"), default="bypass")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.mode == "smoke":
        job = dict(name="unrestricted_replay", source="layer03_s31", scope="all", lr=1e-6)
        run_job(args.parent, args.root / "smoke", job, steps=1, replay=True)
        if args.study.startswith("position"):
            for scope in ("upstream", "downstream"):
                candidate = dict(name="smoke_" + scope, source="layer03_s31", scope=scope, lr=1e-6)
                if args.study == "position_long":
                    candidate = next(j for j in jobs(args.parent, args.study, args.reference)
                                     if j["source"] == "layer22_s31" and j["scope"] == scope)
                run_job(args.parent, args.root / ("smoke_" + scope), candidate, steps=1)
    elif args.mode == "worker":
        job = next(j for j in jobs(args.parent, args.study, args.reference) if j["name"] == args.job)
        run_job(args.parent, args.root / "runs" / job["name"], job,
                steps=64 if args.study == "position_long" else 8)
    elif args.mode == "launch":
        launch(args.root, args.parent, args.study, args.reference)
    else:
        analyse(args.root, args.parent, args.study, args.reference)
