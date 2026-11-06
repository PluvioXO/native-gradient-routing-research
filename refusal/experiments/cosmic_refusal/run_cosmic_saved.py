#!/usr/bin/env python3
"""Run the authors' COSMIC direction selection on saved all-on CASR banks."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

COSMIC_COMMIT = "29085039d79cf428fb2106e580e8f9a7121040a1"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODES = ("independent", "aware", "preserve")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def assignments(cells, seed, gpu):
    jobs = [dict(id=f"train_{support}_{budget}_{mode}_s{seed}", support=support,
                 budget=budget, width=width, mode=mode, seed=seed)
            for support, budget, width in cells for mode in MODES]
    assert len(jobs) == 12 and len({j["id"] for j in jobs}) == 12
    return jobs[gpu::6]


def configure_paths(args):
    os.environ.setdefault("HF_HOME", "/var/tmp/casr-mn866-composition/cache/huggingface")
    os.environ.setdefault("XDG_CACHE_HOME", "/var/tmp/casr-mn866-composition/cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    sys.path[:0] = [str(args.site_packages), str(args.cosmic_root),
                    str(args.study_root), str(args.study_root / "reference"),
                    str(args.study_root / "reference" / "src")]


def cosmic_model(model, tokenizer):
    from pipeline.model_utils.qwen_model import QwenModel

    base = object.__new__(QwenModel)
    base.model_name_or_path = MODEL
    base.model = model
    base.tokenizer = tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.pad_token_id = 151643
    base.tokenize_instructions_fn = base._get_tokenize_instructions_fn()
    base.eoi_toks = base._get_eoi_toks()
    base.refusal_toks = base._get_refusal_toks()
    base.model_block_modules = base._get_model_block_modules()
    base.model_attn_modules = base._get_attn_modules()
    base.model_mlp_modules = base._get_mlp_modules()
    base.model_post_attn_modules = base._get_post_attn_modules()
    assert len(base.model_block_modules) == model.config.num_hidden_layers
    return base


def rank_layers(model, instructions, tokenize_fn, artifact_dir, batch_size=8, fraction=0.10):
    """COSMIC's layer ranking: lowest cosine of harmful/harmless mean activations."""
    import numpy as np
    import torch

    n_layers = model.config.num_hidden_layers
    outputs = [torch.zeros((len(rows), n_layers, model.config.hidden_size), device=model.device)
               for rows in instructions]
    for rows, saved in zip(instructions, outputs):
        for start in range(0, len(rows), batch_size):
            tokens = tokenize_fn(instructions=rows[start:start + batch_size])
            result = model(input_ids=tokens.input_ids.to(model.device),
                           attention_mask=tokens.attention_mask.to(model.device),
                           output_hidden_states=True)
            for layer, state in enumerate(result.hidden_states[1:]):
                saved[start:start + batch_size, layer] = state[:, -1, :]
    means = [saved.mean(dim=0) for saved in outputs]
    normalized = [value / torch.norm(value, dim=-1, keepdim=True) for value in means]
    similarities = torch.sum(normalized[0] * normalized[1], dim=-1).cpu().numpy()
    ranked = np.argsort(similarities).tolist()
    atomic_json(Path(artifact_dir) / "select_direction" / "COSMIC_evaluation_layers.json",
                {"ranked_layers": ranked})
    return ranked[:int(fraction * n_layers)]


def run_one(args, spec, include_base=False):
    import torch
    from torch.nn.utils.parametrize import cached
    from dataset.load_dataset import load_dataset_split
    from pipeline.submodules.generate_directions_cosmic import generate_directions
    from pipeline.submodules.select_direction_cosmic import select_direction
    import run_aware
    import nr5_rl as rl

    run_id = spec["id"]
    checkpoint = None
    if not include_base:
        checkpoint = args.model_output / "fits" / run_id / "bank_final.pt"
        completion = args.model_output / "fits" / run_id / "completed.json"
        if not checkpoint.is_file() or not completion.is_file():
            raise FileNotFoundError(f"Missing completed saved bank: {checkpoint}")
        bank_hash = sha256(checkpoint)
        complete_record = json.loads(completion.read_text())
        if not complete_record.get("status", {}).get("complete") or complete_record["bank_sha256"] != bank_hash:
            raise ValueError(f"Saved bank is incomplete or fails its recorded SHA-256: {run_id}")
        if complete_record["spec"]["id"] != run_id:
            raise ValueError(f"Saved bank completion record names another fit: {run_id}")
        spec = dict(spec, bank_sha256=bank_hash)
    final = args.results / "runs" / run_id
    if final.exists():
        record_path = final / "completed.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("status") == "complete":
                if spec.get("bank_sha256") and record["bank_sha256"] != spec["bank_sha256"]:
                    raise ValueError(f"Completed result has a different checkpoint hash: {run_id}")
                print(f"SKIP {run_id}: already complete", flush=True)
                return True
        shutil.rmtree(final)

    work = args.results / ".work" / f"{run_id}.{os.getpid()}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    model = tokenizer = module = model_base = None
    started = time.time()
    try:
        if include_base:
            model, tokenizer = run_aware.previous.load_model()
            parameterization = "frozen pinned base model"
            base_hash = None
            bank_hash = None
            delta_hash = None
        else:
            bank = rl._load(checkpoint)
            if bank["spec"] != complete_record["spec"] or bank["spec"]["id"] != run_id:
                raise ValueError(f"Saved bank metadata does not match its completion record: {run_id}")
            module, tokenizer = run_aware.previous.build_module(bank["spec"])
            if module.manifest() != bank["manifest"]:
                raise ValueError(f"Rebuilt parameter layout does not match saved bank: {run_id}")
            module.load_state_dict(bank["deltas"])
            module.independent_task = None
            module.assert_frozen_base()
            base_hash = module.base_hash()
            if base_hash != bank["manifest"]["base_hash"]:
                raise ValueError(f"Pinned base model hash mismatch: {run_id}")
            delta_hash = module.delta_hash()
            model = module.model
            parameterization = bank["manifest"]["architecture"]
            spec = dict(bank["spec"], bank_sha256=bank_hash, delta_sha256=delta_hash,
                        base_sha256=base_hash)
            del bank

        model_base = cosmic_model(model, tokenizer)
        sampling_seed, n_train, n_val, batch_size = 42, 180, 100, 8
        random.seed(sampling_seed)
        harmful_train = random.sample(load_dataset_split("harmful", "train", instructions_only=True), n_train)
        harmless_train = random.sample(load_dataset_split("harmless", "train", instructions_only=True), n_train)
        harmful_val = random.sample(load_dataset_split("harmful", "val", instructions_only=True), n_val)
        harmless_val = random.sample(load_dataset_split("harmless", "val", instructions_only=True), n_val)
        data_hash = hashlib.sha256(json.dumps(
            {"harmful_train": harmful_train, "harmless_train": harmless_train,
             "harmful_val": harmful_val, "harmless_val": harmless_val},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest = dict(run_id=run_id, spec=spec, model=MODEL, revision=REVISION,
                        base_sha256=base_hash, bank_sha256=bank_hash, delta_sha256=delta_hash,
                        parameterization=parameterization, active_slots=8 if module else 0,
                        cosmic_commit=COSMIC_COMMIT, method="COSMIC author implementation / ACE affine selection",
                        train_per_class=n_train, validation_per_class=n_val,
                        sampling_seed=sampling_seed, dataset_sha256=data_hash,
                        candidate_positions=len(model_base.eoi_toks),
                        evaluated_layer_fraction=0.10, batch_size=batch_size,
                        torch_version=torch.__version__, device=str(model.device),
                        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)))
        atomic_json(work / "run_manifest.json", manifest)

        gate = module.gate((True,) * 8) if module else nullcontext()
        with torch.no_grad(), gate, cached():
            candidate_directions, harmless_mean = generate_directions(
                model_base, harmful_train, harmless_train,
                artifact_dir=str(work / "generate_directions"), batch_size=batch_size)
            torch.save(harmless_mean.cpu(), work / "generate_directions" / "harmless_mean.pt")

            layers = rank_layers(model, (harmful_train, harmless_train),
                                 model_base.tokenize_instructions_fn, work,
                                 batch_size=batch_size, fraction=0.10)
            pos, layer, direction, reference = select_direction(
                model_base, harmful_val, harmless_val, candidate_directions,
                harmless_mean, layers, str(work / "select_direction"),
                affine_steering=True, batch_size=batch_size)

        torch.save(direction.detach().cpu(), work / "direction.pt")
        torch.save(reference.detach().cpu(), work / "harmless_reference.pt")
        evaluations = json.loads((work / "select_direction" / "direction_evaluations.json").read_text())
        selected = next(r for r in evaluations if r["position"] == pos and r["layer"] == layer)
        files = {str(p.relative_to(work)): sha256(p) for p in sorted(work.rglob("*")) if p.is_file()}
        completed = dict(status="complete", selected_position=pos, selected_layer=layer,
                         selected_scores=selected, ranked_layers=layers,
                         elapsed_seconds=time.time() - started, files_sha256=files,
                         completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(work / "completed.json", completed)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(work, final)
        print(f"COMPLETE {run_id}: layer={layer}, position={pos}, score={selected}", flush=True)
        return True
    except Exception as exc:
        atomic_json(work / "failure.json", dict(run_id=run_id, error=repr(exc),
                                                  failed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        failed = args.results / "failures" / run_id
        failed.parent.mkdir(parents=True, exist_ok=True)
        if failed.exists():
            shutil.rmtree(failed)
        os.replace(work, failed)
        print(f"FAILED {run_id}: {exc!r}", flush=True)
        return False
    finally:
        if module is not None and hasattr(module, "close"):
            module.close()
        del model, tokenizer, module
        if model_base is not None:
            del model_base
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--cosmic-root", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(31, 47, 71), required=True)
    parser.add_argument("--gpu", type=int, choices=range(6), required=True)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--only-job")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_paths(args)
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import run_aware
    if run_aware.previous.CELLS != (("native", "small", 4), ("lora", "small", 1),
                                    ("native", "medium", 26), ("lora", "medium", 4)):
        raise RuntimeError("The saved-bank sweep no longer matches the expected COSMIC input plan")
    if run_aware.previous.MODEL != MODEL or run_aware.previous.REVISION != REVISION:
        raise RuntimeError("The saved-bank base model or revision changed")
    if args.cosmic_root.joinpath(".git").exists():
        import subprocess
        actual = subprocess.check_output(["git", "-C", str(args.cosmic_root), "rev-parse", "HEAD"], text=True).strip()
        if actual != COSMIC_COMMIT:
            raise RuntimeError(f"Unexpected COSMIC source revision: {actual}")
    jobs = assignments(run_aware.previous.CELLS, args.seed, args.gpu)
    if args.seed == 31 and args.gpu == 0 and args.include_base:
        jobs.insert(0, dict(id="qwen25_05b_base", seed=31, support="base", budget="base", width=0, mode="base"))
    if args.only_job:
        jobs = [job for job in jobs if job["id"] == args.only_job]
        if len(jobs) != 1:
            raise ValueError(f"Job {args.only_job!r} is not assigned to seed {args.seed}, GPU {args.gpu}")
    if args.dry_run:
        print(json.dumps(dict(seed=args.seed, gpu=args.gpu, jobs=[j["id"] for j in jobs]), indent=2))
        return
    args.results.mkdir(parents=True, exist_ok=True)
    failed = []
    for spec in jobs:
        if not run_one(args, spec, include_base=spec["mode"] == "base"):
            failed.append(spec["id"])
    if failed:
        raise SystemExit(f"COSMIC worker failed jobs: {failed}")


if __name__ == "__main__":
    main()
