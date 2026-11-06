#!/usr/bin/env python3
"""Independent native/LoRA RL fits, then frozen-module composition evaluation."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
SEEDS = (31, 47, 71)
LRS = (3e-5, 1e-4, 3e-4)
CELLS = (("native", "small", 4), ("lora", "small", 1), ("native", "medium", 26), ("lora", "medium", 4))
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execution_revision(output, plan, current_sources=None):
    path = output / "execution_revision.json"
    if not path.exists():
        return dict(id="original", source_hashes=plan["source_hashes"])
    record = json.loads(path.read_text())
    if record["scientific_plan_sha256"] != digest(plan):
        raise ValueError("Execution revision belongs to another scientific plan")
    expected_id = digest({k: v for k, v in record.items() if k != "id"})
    if record["id"] != expected_id:
        raise ValueError("Execution revision checksum mismatch")
    if current_sources is not None and record["source_hashes"] != current_sources:
        raise ValueError("Execution revision does not match current source files")
    return record


def validate_completed(output, job, plan):
    path = output / "fits" / job["id"] / "completed.json"
    if not path.exists(): return False
    try:
        record = json.loads(path.read_text())
        assert all(record["spec"][key] == value for key, value in job.items())
        if job["kind"] == "baseline":
            baseline = json.loads((output / "baseline.json").read_text())
            assert record["complete"] and baseline["complete"] and baseline["base_hash"]
            assert baseline["identity"]["benchmarks"] == digest(json.loads((ROOT / "benchmarks.json").read_text()))
            assert baseline["identity"]["scorer_sha256"] == hash_file(ROOT / "src/retention.py")
        else:
            assert record["plan_sha256"] == digest(plan)
            if job["kind"] == "merge":
                assert record["composition"] and record["retention"] and record["artifact_hashes"]
                for name, expected in record["artifact_hashes"].items():
                    rel = Path(name)
                    assert not rel.is_absolute() and ".." not in rel.parts
                    assert hash_file(path.parent / rel) == expected
            else:
                assert record["final_step"] == job["steps"] and record["status"]["complete"]
                assert hash_file(path.parent / "owner_final.pt") == record["owner_sha256"]
        return True
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        raise ValueError(f"Invalid completion record for {job['id']}; repair outputs before resuming") from error


def make_plan(smoke=False):
    sources = [*sorted(ROOT.glob("*.py")), *sorted((ROOT / "src").glob("*.py")),
               *sorted((ROOT / "tests").glob("*.py")), ROOT / "dataset.json", ROOT / "benchmarks.json", ROOT / "requirements.txt"]
    jobs = []
    if smoke:
        for support, budget, width in CELLS:
            jobs.append(dict(id=f"smoke_{support}_{budget}", kind="smoke", support=support,
                             budget=budget, width=width, task=7, seed=907, lr=1e-4, steps=2))
    else:
        jobs.append(dict(id="baseline", kind="baseline"))
        for support, budget, width in CELLS:
            for lr in LRS:
                for task in (0, 1):
                    jobs.append(dict(id=f"pilot_{support}_{budget}_lr{lr:g}_t{task}", kind="pilot",
                        support=support, budget=budget, width=width, task=task, seed=907, lr=lr, steps=160))
        main = [dict(id=f"main_{support}_{budget}_s{seed}_t{task}", kind="main", support=support,
                     budget=budget, width=width, task=task, seed=seed, steps=320)
                for support, budget, width in CELLS for seed in SEEDS for task in range(8)]
        random.Random(20261021).shuffle(main)
        jobs.extend(main)
        for support, budget, width in CELLS:
            for seed in SEEDS:
                jobs.append(dict(id=f"merge_{support}_{budget}_s{seed}", kind="merge", support=support,
                    budget=budget, width=width, seed=seed,
                    dependencies=[f"main_{support}_{budget}_s{seed}_t{t}" for t in range(8)]))
    return dict(study="independent-native-lora-composition-v1", model=MODEL, revision=REVISION, smoke=smoke,
        tasks=["arithmetic", "pointer", "boolean", "brackets", "sorting", "stack", "shortest_path", "sudoku"],
        seeds=list(SEEDS), pilot_seed=907, learning_rates=list(LRS), precision="float32",
        learning_rate_selection="Mean final-step development accuracy on independent arithmetic/pointer pilots; one LR per support/budget; smaller LR breaks ties. No test or retention scores used.",
        primary="Final step 320. Taskwise all-eight minus standalone accuracy; paired lost/rescued answers, macro and worst task; singleton/2/4/8 curves. Report all tasks, flag standalone gain below 5 percentage points.",
        independent_training="Only one task's train/dev examples; only its slot receives updates; all other slots stay at initial values with empty optimizer histories. No foreign/composition preservation and no combined training.",
        comparison="Approximately parameter matched; not FLOP matched. Report rollout tokens, RL seconds, peak memory and measured evaluation seconds. Total allocation grows with skill count.",
        evaluation="All 28 pairs; eight cyclic sets of four (each task appears four times); all eight; each task alone; all eight minus the evaluated owner. Frozen original test bank is exploratory, already inspected historically.",
        calibration="Secondary per-owner coefficient selection in one coordinate pass over {0,0.5,1,1.5}, maximizing macro development accuracy. Identical procedure for native and LoRA, no weight learning. Applied to the full-eight mixture; unit coefficients remain primary.",
        compounds="Fresh combinations of held-out primitive inputs: pointer then arithmetic, stack top-history then sorting. Base, each component, both, all eight, calibrated all eight; gold-intermediate and generated-intermediate pipeline controls. Compound demos use training inputs only. No compound tuning.",
        retention="Frozen 996-question MC and 16384-token WikiText panel from pointer sweep; baseline, each singleton, all-eight and calibrated all-eight. Reused panel, descriptive only.",
        checkpointing="Main disk outside home quota; atomic local/durable continuation copies, raw rollouts, 40-step snapshots, final owner tensors and exact source/data hashes. Free-space guard 60 GiB. Local disk is not a second-machine backup.",
        jobs=jobs, source_hashes={str(p.relative_to(ROOT)): hash_file(p) for p in sources})


def init_runtime():
    import multi_run
    import nr5_rl as rl
    import nr6_train as train
    import multi_tasks as tasks
    import compound_tasks
    multi_run.setup()
    data = json.loads((ROOT / "dataset.json").read_text())
    compound = compound_tasks.build(data)
    primitive_prompt = tasks.format_prompt
    def format_prompt(e):
        return compound_tasks.prompt(e, compound["demos"]) if "family" in e else primitive_prompt(e)
    rl.format_prompt = train.format_prompt = format_prompt
    train.RECIPES["independent_routed"] = ("routed", "all_ab")
    return data, compound


def build_module(spec):
    from multi_run import load_model
    from budget_model import BudgetNative, BudgetLoRA
    model, tok = load_model()
    kwargs = dict(seed=spec["seed"])
    module = (BudgetNative(model, neurons=spec["width"], **kwargs) if spec["support"] == "native"
              else BudgetLoRA(model, rank=spec["width"], **kwargs))
    return module, tok


def config(spec, plan):
    from multi_run import Config
    from nr6_io import digest
    smoke = spec["kind"] == "smoke"
    return Config(max_steps=spec.get("steps", 320), lr=spec.get("lr", 1e-4),
        prompts_per_update=1 if smoke else 4, group_size=2 if smoke else 4,
        generation_batch=2, eval_batch=2 if smoke else 8, max_new_tokens=96,
        max_prompt_tokens=1536, eval_every=2 if smoke else 40, save_every=2 if smoke else 20,
        snapshot_every=2 if smoke else 40, head_chunk=8, kl_head_chunk=8,
        max_mean_kl=10. if smoke else .02, max_sequence_kl=10. if smoke else .1,
        preservation_mode="all_ab", audit_steps=[], seed=spec.get("seed", 31),
        arm_name=spec["id"], task_count=1, cluster_plan_sha256=digest(plan)).validate()


def checked_space(path):
    if shutil.disk_usage(path).free < 60 * 2**30:
        raise OSError("Fewer than 60 GiB free on experiment disk; refusing another checkpoint/job")


def train_fit(spec, plan, output):
    import torch
    import nr5_rl as rl
    import nr6_train as train
    from nr6_io import Workspace, digest
    from multi_run import witnesses
    data, compound = init_runtime()
    task = spec["task"]
    cfg = config(spec, plan)
    subset = {s: [e for e in data[s] if e["task_id"] == task] for s in ("train", "dev")}
    if spec["kind"] == "smoke":
        subset = {s: rows[:1] for s, rows in subset.items()}
        from dai_rl import update
        def diagnostic(*args):
            args = list(args); args[3] = [1.] * len(args[2])
            result = update(*args); result["scripted_smoke_advantages"] = True
            return result
        train.extended_update = diagnostic
    fit = output / "fits" / spec["id"]
    ws = Workspace(fit / "local", fit / "durable",
        dict(plan_sha256=digest(plan), spec=spec, data_hash=digest(subset), config=asdict(cfg)))
    ws.restore_training("fit", cfg)
    module, tok = build_module(spec)
    module.independent_task = task
    module.execution_revision = execution_revision(output, plan)["id"]
    try:
        initial = {n: v.cpu() for n, v in module.state_dict().items()}
        example = subset["dev"][:1]
        with module.gate((False,) * 8): original = witnesses(module.model, tok, example)
        if ws.path("fit/latest.pt").exists():
            restored = rl._load(ws.path("fit/latest.pt"))
            assert all(torch.equal(initial[n], restored["deltas"][n]) for n, _, t in module._named_deltas if t != task)
            assert all(not state["state"] for i, state in enumerate(restored["optimizers"]) if i != task)
        retained = set()
        def save():
            checked_space(output)
            ws.save_training("fit", cfg)
            for folder, pattern in (("rollouts", "*.json.gz"), ("snapshots", "*.pt")):
                for path in (ws.path("fit") / folder).glob(pattern):
                    rel = str(path.relative_to(ws.local))
                    if rel not in retained:
                        ws.publish(rel, evidence=False); retained.add(rel)
        started = time.monotonic()
        status = train.train_recipe(module, tok, subset, cfg, ws.path("fit"), "independent_routed", spec["seed"], save)
        latest = rl._load(ws.path("fit/latest.pt"))
        module.load_state_dict(latest["deltas"])
        assert latest["step"] == cfg.max_steps
        assert {r["task_id"] for r in latest["history"]} == {task}
        assert all(not state["state"] for i, state in enumerate(latest["optimizers"]) if i != task)
        assert all(torch.equal(initial[n], p.detach().cpu()) for n, p, t in module._named_deltas if t != task)
        moved = {region: any(not torch.equal(initial[n], p.detach().cpu())
                 for n, p, t in module._named_deltas if t == task and region in n)
                 for region in (".mlp.", ".self_attn.")}
        if spec["kind"] == "smoke":
            module.independent_task = None
            assert all(moved.values()) and status["updates_with_movement"] > 0
            from torch.nn.utils.parametrize import cached
            candidates = data["train"] + compound["test"]
            longest = max(candidates, key=lambda e: len(tok(rl.format_prompt(e), add_special_tokens=False).input_ids))
            enc = tok(rl.format_prompt(longest), add_special_tokens=False, return_tensors="pt",
                      return_token_type_ids=False).to("cuda")
            with torch.no_grad(), module.gate((True,) * 8), cached():
                generated = module.model.generate(**enc, min_new_tokens=96, max_new_tokens=96,
                    do_sample=False, eos_token_id=None, pad_token_id=tok.pad_token_id,
                    temperature=None, top_p=None, top_k=None)
            assert enc.input_ids.shape[1] <= cfg.max_prompt_tokens
            assert generated.shape[1] - enc.input_ids.shape[1] == 96
            del generated, enc
            mini_compound = dict(compound, test=[next(e for e in compound["test"] if e["family"] == family)
                                                for family in ("pointer_arithmetic", "stack_sorting")])
            compound_evaluation(module, tok, mini_compound, cfg, fit / "smoke_compounds")
            import retention
            benchmark = json.loads((ROOT / "benchmarks.json").read_text())
            benchmark = dict(benchmark, examples=[next(e for e in benchmark["examples"] if e["benchmark"] == name)
                              for name in benchmark["protocol"]["primary_metrics"]], wiki_chunks=benchmark["wiki_chunks"][:1])
            retention.evaluate(module, benchmark, fit / "smoke_retention.json", (True,) * 8, ws.runtime)
            assert torch.cuda.max_memory_allocated() / 2**30 < 7.25
        with module.gate((False,) * 8): restored = witnesses(module.model, tok, example)
        assert all(torch.equal(a, b) for a, b in zip(original, restored)) and module.assert_frozen_base()
        payload = dict(spec=spec, plan_sha256=digest(plan), execution_revision=module.execution_revision, manifest=module.manifest(),
            deltas={n: p.detach().cpu() for n, p, t in module._named_deltas if t == task},
            step=latest["step"], base_hash=module.base_hash())
        rl.atomic_torch(fit / "owner_final.pt", payload)
        accuracy = latest["dev_history"][-1]["accuracy"]
        completion = dict(spec=spec, plan_sha256=digest(plan), execution_revision=module.execution_revision,
            timing_by_revision={revision: dict(updates=sum(r.get("execution_revision", "original") == revision for r in latest["history"]),
                training_seconds=sum(r["seconds"] for r in latest["history"] if r.get("execution_revision", "original") == revision))
                for revision in sorted({r.get("execution_revision", "original") for r in latest["history"]})}, final_step=latest["step"],
            owner_sha256=hash_file(fit / "owner_final.pt"), status=status,
            dev_accuracy=accuracy[f"{task}/owner"], base_dev_accuracy=accuracy[f"{task}/none"],
            scalars_per_task=module.manifest()["trainable_scalars_per_slot"][task],
            training_seconds=sum(r["seconds"] for r in latest["history"]),
            response_tokens=sum(r["response_tokens"] for r in latest["history"]),
            peak_cuda_gib=torch.cuda.max_memory_allocated() / 2**30,
            both_regions_moved=moved, unused_slots_unchanged=True, all_off_equal=True,
            elapsed_this_session=time.monotonic() - started)
        atomic(fit / "completed.json", completion)
        print("COMPLETE", spec["id"], completion["dev_accuracy"], flush=True)
    finally: module.close()


def masks_for_task(task):
    """Every pair; balanced cyclic quadruples; all-on and owner-removal."""
    groups = [(), (task,)]
    groups += [tuple(pair) for pair in combinations(range(8), 2) if task in pair]
    groups += sorted({tuple(sorted((start + j) % 8 for j in range(4))) for start in range(8)
                      if task in {(start + j) % 8 for j in range(4)}})
    groups += [tuple(range(8)), tuple(t for t in range(8) if t != task)]
    return groups


def mask_name(indices):
    return "set_" + ("_".join(map(str, indices)) or "none")


def predictions(module, tok, examples, indices, cfg, path, split):
    import nr5_rl as rl
    from nr6_io import digest
    name = mask_name(indices)
    rl.MASKS[name] = tuple(t in indices for t in range(8))
    identity = dict(delta=module.delta_hash(), data=digest(examples), indices=list(indices),
                    cfg=asdict(cfg), split=split)
    record = json.loads(path.read_text()) if path.exists() else dict(identity=identity, rows=[])
    assert record["identity"] == identity
    seen = {r["id"] for r in record["rows"]}
    assert len(seen) == len(record["rows"]) and seen <= {e["id"] for e in examples}
    for task in sorted({e["task_id"] for e in examples}):
        missing = [e for e in examples if e["task_id"] == task and e["id"] not in seen]
        for start in range(0, len(missing), cfg.eval_batch):
            started = time.monotonic()
            record["rows"].extend(rl.predict(module, tok, missing[start:start + cfg.eval_batch],
                name, cfg, split, cfg.arm_name, cfg.seed))
            record["generation_seconds"] = record.get("generation_seconds", 0.) + time.monotonic() - started
            atomic(path, record)
    assert len(record["rows"]) == len(examples)
    return record["rows"]


def combine_owners(module, payloads, plan_hash):
    import torch
    state = module.state_dict()
    assert len(payloads) == 8 and {p["spec"]["task"] for p in payloads} == set(range(8))
    for payload in payloads:
        task = payload["spec"]["task"]
        assert payload["plan_sha256"] == plan_hash and payload["manifest"] == module.manifest()
        expected = {n for n, _, t in module._named_deltas if t == task}
        assert set(payload["deltas"]) == expected
        state.update(payload["deltas"])
    module.load_state_dict(state)
    assert all(torch.equal(p.detach().cpu(), state[n].cpu()) for n, p, _ in module._named_deltas)
    return {n: v.detach().cpu().clone() for n, v in state.items()}


def load_scaled(module, original, coefficients):
    # Scale B only: scaling both LoRA factors would square the intended coefficient.
    state = {n: v * coefficients[t] if not n.endswith(".A") else v
             for n, _, t in module._named_deltas for v in [original[n]]}
    module.load_state_dict(state)


def calibrate(module, tok, data, cfg, folder, original):
    selected = folder / "calibration.json"
    if selected.exists(): return json.loads(selected.read_text())["coefficients"]
    coefficients = [1.] * 8
    trials = []
    for task in range(8):
        scores = []
        for value in (1., .5, 1.5, 0.):
            candidate = list(coefficients); candidate[task] = value
            load_scaled(module, original, candidate)
            rows = predictions(module, tok, data["dev"], tuple(range(8)), cfg,
                folder / "calibration" / f"task_{task}_value_{value:g}.json", "calibration_dev")
            score = sum(r["correct"] for r in rows) / len(rows)
            scores.append((score, value)); trials.append(dict(task=task, coefficients=candidate, macro_accuracy=score))
        best = max(score for score, _ in scores)
        coefficients[task] = next(v for s, v in scores if s == best)
    atomic(selected, dict(coefficients=coefficients, trials=trials,
                         scope="One coordinate pass on standard development questions only; no compound/test/retention tuning"))
    return coefficients


def compound_evaluation(module, tok, compound, cfg, folder):
    import compound_tasks
    import nr5_rl as rl
    from nr6_io import digest
    result = []
    for family, components in compound_tasks.FAMILIES:
        examples = [e for e in compound["test"] if e["family"] == family]
        for indices in [(), (components[0],), (components[1],), tuple(sorted(components)), tuple(range(8))]:
            rows = predictions(module, tok, examples, indices, cfg,
                               folder / f"{family}_{mask_name(indices)}.json", "compound_test")
            result.append(dict(family=family, mask=list(indices), n=len(rows), correct=sum(r["correct"] for r in rows)))
        # Gold intermediate is supplied only to this explicitly assisted control.
        rows = predictions(module, tok, [dict(e["second"], id=e["id"]) for e in examples],
            (components[1],), cfg, folder / f"{family}_gold_intermediate.json", "assisted_second_stage")
        result.append(dict(family=family, control="gold_intermediate", n=len(rows), correct=sum(r["correct"] for r in rows)))
        path = folder / f"{family}_pipeline.json"
        identity = dict(delta=module.delta_hash(), data=digest(examples), config=asdict(cfg))
        saved = json.loads(path.read_text()) if path.exists() else dict(identity=identity, rows=[])
        assert saved["identity"] == identity
        seen = {r["id"] for r in saved["rows"]}
        missing = [e for e in examples if e["id"] not in seen]
        for start in range(0, len(missing), cfg.eval_batch):
            batch = missing[start:start + cfg.eval_batch]
            stage1 = rl._generate(module, tok, [e["first"] for e in batch], cfg, f"slot_{components[0]}")
            stage2_queries = [compound_tasks.intermediate_query(e, item["completion"]) for e, item in zip(batch, stage1)]
            valid = [q for q in stage2_queries if q is not None]
            stage2 = iter(rl._generate(module, tok, valid, cfg, f"slot_{components[1]}") if valid else [])
            for e, first, query in zip(batch, stage1, stage2_queries):
                second = next(stage2) if query is not None else None
                saved["rows"].append(dict(id=e["id"], first=first, second=second,
                    gold=e["answer"], correct=bool(second and second["answer"] == e["answer"])))
            atomic(path, saved)
        result.append(dict(family=family, control="generated_intermediate_pipeline", n=len(saved["rows"]),
                           correct=sum(r["correct"] for r in saved["rows"])))
    return result


def merge_fit(spec, plan, output):
    import torch
    from multi_run import witnesses
    import nr5_rl as rl
    from nr6_io import digest, runtime_info
    import retention
    data, compound = init_runtime()
    module, tok = build_module(spec)
    folder = output / "fits" / spec["id"]; folder.mkdir(parents=True, exist_ok=True)
    cfg = config(spec, plan)
    try:
        examples = [next(e for e in data["dev"] if e["task_id"] == t) for t in range(8)]
        with module.gate((False,) * 8): before_logits = witnesses(module.model, tok, examples)
        payloads = []
        sources = {}
        if "bank_checkpoint" in spec:
            fit = output / "fits" / spec["bank_checkpoint"]
            completion = json.loads((fit / "completed.json").read_text())
            assert hash_file(fit / "bank_final.pt") == completion["bank_sha256"]
            bank = rl._load(fit / "bank_final.pt")
            assert bank["plan_sha256"] == digest(plan) and bank["manifest"] == module.manifest()
            module.load_state_dict(bank["deltas"])
            original = {name: value.cpu() for name, value in module.state_dict().items()}
            sources[spec["bank_checkpoint"]] = completion["bank_sha256"]
            del bank
        else:
            for task, job in enumerate(spec["dependencies"]):
                fit = output / "fits" / job
                completion = json.loads((fit / "completed.json").read_text())
                assert hash_file(fit / "owner_final.pt") == completion["owner_sha256"]
                payloads.append(rl._load(fit / "owner_final.pt")); sources[job] = completion["owner_sha256"]
            original = combine_owners(module, payloads, digest(plan))
        del payloads
        with module.gate((False,) * 8): after_logits = witnesses(module.model, tok, examples)
        assert all(torch.equal(a, b) for a, b in zip(before_logits, after_logits)) and module.assert_frozen_base()
        atomic(folder / "composition_manifest.json", dict(source_owners=sources, manifest=module.manifest(),
            plan_sha256=digest(plan), compound_hash=digest(compound), all_off_equal=True))
        benchmarks = json.loads((ROOT / "benchmarks.json").read_text())
        baseline = retention.evaluate(module, benchmarks, folder / "retention/base.json", (False,) * 8, runtime_info())
        all_rows, timings = [], []
        for task in range(8):
            examples = [e for e in data["test"] if e["task_id"] == task]
            for indices in masks_for_task(task):
                evaluation_path = folder / "evaluation" / f"task_{task}_{mask_name(indices)}.json"
                rows = predictions(module, tok, examples, indices, cfg, evaluation_path, "test")
                all_rows.extend(dict(r, active_slots=list(indices), active_count=len(indices)) for r in rows)
                timings.append(dict(task=task, mask=list(indices),
                    seconds=json.loads(evaluation_path.read_text())["generation_seconds"], n=len(rows)))
                print("EVALUATION", spec["id"], task, indices, flush=True)
        from collections import defaultdict
        grouped = defaultdict(list)
        for row in all_rows: grouped[row["task_id"], tuple(row["active_slots"])].append(row)
        summary = []
        for (task, indices), rows in sorted(grouped.items()):
            solo = {r["id"]: r for r in grouped[task, (task,)]}
            base = {r["id"]: r for r in grouped[task, ()]}
            summary.append(dict(task=task, active_slots=list(indices), active_count=len(indices), n=len(rows),
                accuracy=sum(r["correct"] for r in rows) / len(rows),
                standalone_accuracy=sum(r["correct"] for r in solo.values()) / len(solo),
                base_accuracy=sum(r["correct"] for r in base.values()) / len(base),
                standalone_gain_pp=100 * (sum(r["correct"] for r in solo.values()) - sum(r["correct"] for r in base.values())) / len(rows),
                standalone_gain_below_5pp=(sum(r["correct"] for r in solo.values()) - sum(r["correct"] for r in base.values())) / len(rows) < .05,
                combined_minus_standalone_pp=100 * (sum(r["correct"] for r in rows) - sum(r["correct"] for r in solo.values())) / len(rows),
                lost=sum(solo[r["id"]]["correct"] and not r["correct"] for r in rows),
                rescued=sum(not solo[r["id"]]["correct"] and r["correct"] for r in rows),
                changed=sum(solo[r["id"]]["answer"] != r["answer"] for r in rows)))
        atomic(folder / "composition_summary.json", summary)
        atomic(folder / "evaluation_timing.json", timings)
        retention_results = {}
        for indices in [(t,) for t in range(8)] + [tuple(range(8))]:
            result = retention.evaluate(module, benchmarks, folder / f"retention/{mask_name(indices)}.json",
                                        tuple(t in indices for t in range(8)), runtime_info())
            retention_results[mask_name(indices)] = retention.changes(baseline, result, benchmarks)
        compounds = compound_evaluation(module, tok, compound, cfg, folder / "compounds")
        coefficients = calibrate(module, tok, data, cfg, folder, original)
        load_scaled(module, original, coefficients)
        calibrated_rows = predictions(module, tok, data["test"], tuple(range(8)), cfg,
            folder / "evaluation/calibrated_all.json", "calibrated_test")
        calibrated_compounds = predictions(module, tok, compound["test"], tuple(range(8)), cfg,
            folder / "compounds/calibrated_all.json", "calibrated_compound_test")
        calibrated_retention = retention.evaluate(module, benchmarks, folder / "retention/calibrated_all.json", (True,) * 8, runtime_info())
        assert module.assert_frozen_base()
        atomic(folder / "completed.json", dict(spec=spec, plan_sha256=digest(plan), execution_revision=execution_revision(output, plan)["id"],
            artifact_hashes={str(p.relative_to(folder)): hash_file(p) for p in sorted(folder.rglob("*.json")) if p.name != "completed.json"}, composition=summary,
            retention=retention_results, compounds=compounds, coefficients=coefficients,
            calibrated_accuracy=sum(r["correct"] for r in calibrated_rows) / len(calibrated_rows),
            calibrated_compound_accuracy=sum(r["correct"] for r in calibrated_compounds) / len(calibrated_compounds),
            calibrated_retention=retention.changes(baseline, calibrated_retention, benchmarks), all_off_equal=True))
    finally: module.close()


def select_lrs(output, plan):
    path = output / "selected_learning_rates.json"
    if path.exists(): return json.loads(path.read_text())
    selection = {}
    for support, budget, width in CELLS:
        scores = []
        for lr in LRS:
            jobs = [j for j in plan["jobs"] if j["kind"] == "pilot" and
                    (j["support"], j["budget"], j["lr"]) == (support, budget, lr)]
            assert len(jobs) == 2
            records = [json.loads((output / "fits" / j["id"] / "completed.json").read_text()) for j in jobs]
            scores.append(dict(lr=lr, score=sum(r["dev_accuracy"] for r in records) / len(records)))
        best = max(row["score"] for row in scores)
        selection[f"{support}_{budget}"] = dict(lr=next(row["lr"] for row in scores if row["score"] == best), candidates=scores)
    atomic(path, selection)
    return selection


def worker(job, plan, output):
    if job["kind"] in ("main", "merge"):
        selected = select_lrs(output, plan)
        job = dict(job, lr=selected[f"{job['support']}_{job['budget']}"]["lr"])
    if job["kind"] == "baseline":
        import multi_run
        init_runtime()
        multi_run.baseline(output)
        atomic(output / "fits/baseline/completed.json", dict(complete=True, spec=job))
    elif job["kind"] == "merge": merge_fit(job, plan, output)
    else: train_fit(job, plan, output)


def status(output):
    path = output / "status.json"
    if path.exists(): print(path.read_text())
    else: print("No launcher status yet")


def supervise(plan, output, devices):
    pending = list(plan["jobs"]); processes = {}; attempts = {}; failed = []
    available = list(devices)
    validated = set()
    jobs_by_id = {job["id"]: job for job in plan["jobs"]}
    def done(job):
        if job["id"] in validated: return True
        if validate_completed(output, jobs_by_id[job["id"]], plan):
            validated.add(job["id"]); return True
        return False
    pending = [j for j in pending if not done(j)]
    for job in pending:
        path = output / "fits" / job["id"] / "worker.lock"
        if path.exists():
            try:
                with exclusive(path): pass
            except BlockingIOError as error:
                raise RuntimeError(f"Surviving worker owns {job['id']}; refusing duplicate launch") from error
    pilots = [j for j in plan["jobs"] if j["kind"] == "pilot"]
    baseline_jobs = [j for j in plan["jobs"] if j["kind"] == "baseline"]
    stop_error, report = None, {}
    try:
        while True:
            if pending or processes: checked_space(output)
            ready = [j for j in pending if (j["kind"] not in ("main", "merge") or all(done(p) for p in pilots))
                     and (j["kind"] != "merge" or all(done(dict(id=d)) for d in j["dependencies"]))
                     and (j["kind"] == "baseline" or all(done(b) for b in baseline_jobs))]
            for job in ready[:len(available)]:
                pending.remove(job); device = available.pop(0)
                log = output / "logs" / f"{job['id']}.log"; log.parent.mkdir(exist_ok=True)
                with log.open("a") as stream:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                        "--output", str(output), "--job", job["id"]], stdout=stream, stderr=subprocess.STDOUT,
                        env=env, stdin=subprocess.DEVNULL)
                processes[job["id"]] = (process, device, job)
                attempts[job["id"]] = attempts.get(job["id"], 0) + 1
                print("START", job["id"], "GPU", device, "PID", process.pid, flush=True)
            for key, (process, device, job) in list(processes.items()):
                code = process.poll()
                if code is None: continue
                del processes[key]; available.append(device)
                if code == 75:
                    raise RuntimeError(f"Another worker owns {key}; refusing concurrent writes")
                if code == 0 and done(job): print("DONE", key, flush=True)
                elif attempts[key] < 3:
                    print("RETRY", key, "exit", code, flush=True); pending.append(job)
                else: failed.append(dict(id=key, exit=code)); print("FAILED", key, code, flush=True)
            complete = [j for j in plan["jobs"] if done(j)]
            active = []
            for process, device, job in processes.values():
                resume = output / "fits" / job["id"] / "durable/fit/resume.json"
                state = json.loads(resume.read_text()) if resume.exists() else {}
                active.append(dict(id=job["id"], gpu=device, pid=process.pid, step=state.get("step", 0), total=job.get("steps")))
            report = dict(updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), planned=len(plan["jobs"]),
                completed=len(complete), by_kind={kind:dict(done=sum(j["kind"] == kind for j in complete),
                    total=sum(j["kind"] == kind for j in plan["jobs"])) for kind in ("baseline", "pilot", "main", "merge", "smoke")},
                active=active, pending=len(pending), failed=failed, launcher_state="running", execution_revision=execution_revision(output, plan)["id"],
                free_gib=shutil.disk_usage(output).free / 2**30)
            atomic(output / "status.json", report)
            if not pending and not processes: break
            if not processes and pending and not ready:
                raise RuntimeError(f"No runnable jobs; failed dependencies: {failed}")
            time.sleep(10 if processes else .1)
        if failed: raise RuntimeError(f"Jobs failed: {failed}")
        report["launcher_state"] = "complete"
        atomic(output / "status.json", report)
        atomic(output / "ALL_COMPLETE.json", report)
        print("ALL COMPLETE", flush=True)
    except BaseException as error:
        stop_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        for process, _, _ in processes.values(): process.terminate()
        for process, _, _ in processes.values():
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        if stop_error is not None:
            report.update(launcher_state="stopped", stop_reason=stop_error, active=[],
                          updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            atomic(output / "status.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "smoke", "worker", "status"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--job")
    args = parser.parse_args(); output = args.output.resolve()
    if args.action == "status": status(output); return
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "worker":
        plan = json.loads((output / "experiment_plan.json").read_text())
        revision = execution_revision(output, plan)
        assert all(hash_file(ROOT / path) == value for path, value in revision["source_hashes"].items()), "Source changed during study"
        job = next(j for j in plan["jobs"] if j["id"] == args.job)
        try:
            with exclusive(output / "fits" / job["id"] / "worker.lock"):
                worker(job, plan, output)
        except BlockingIOError:
            print("Worker already owns", job["id"], flush=True); raise SystemExit(75)
        return
    with exclusive(output / "launcher.lock"):
        current = make_plan(args.action == "smoke")
        old = output / "experiment_plan.json"
        plan = json.loads(old.read_text()) if old.exists() else current
        if {k: v for k, v in plan.items() if k != "source_hashes"} != {k: v for k, v in current.items() if k != "source_hashes"}:
            raise ValueError("Scientific settings differ: choose a fresh output directory")
        revision = execution_revision(output, plan, current["source_hashes"])
        if revision["source_hashes"] != current["source_hashes"]:
            raise ValueError("Plan/source differs without an approved execution revision")
        if not old.exists(): atomic(old, plan)
        devices = args.gpus.split(",")
        assert devices and len(set(devices)) == len(devices) and all(d.isdigit() for d in devices)
        def terminate(signum, frame): raise SystemExit(128 + signum)
        previous = signal.signal(signal.SIGTERM, terminate)
        try: supervise(plan, output, devices)
        finally: signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__": main()
