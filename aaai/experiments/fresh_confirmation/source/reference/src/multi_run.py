"""Pointer-only worker; joint MLP/attention, preserved raw data and reusable slots."""
from dataclasses import asdict, dataclass
import hashlib
import json
import shutil
from pathlib import Path
import time
import torch
from torch.nn.utils.parametrize import cached
from transformers import AutoModelForCausalLM, AutoTokenizer
import nr5_rl as rl
import nr6_train as train
import nr6_io as io
from nr6_io import Workspace, digest
from multi_model import MultiNative, MultiLoRA
from budget_model import BudgetNative, BudgetLoRA
from budget_config import ACTIVE_TASKS
import multi_tasks as tasks
from dai_drive import DriveBackup
from dai_rl import update
import retention

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def setup():
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = io.runtime_info
    io.runtime_info = lambda: dict(original(), dtype="float32", microbatch=1)
    rl.MASKS.clear()
    rl.MASKS.update({name: None for name in ("none", "owner", "foreign", "all")})
    rl.format_prompt = tasks.format_prompt
    train.format_prompt = tasks.format_prompt
    train.RECIPES["unrouted_preserve_all"] = ("unrouted_preserve", "all_ab")
    train.extended_update = update


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left",
                                        local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        use_safetensors=True, attn_implementation="sdpa", local_files_only=True,
        trust_remote_code=False).to("cuda").eval()
    return model, tok


def baseline(output):
    setup()
    benchmarks = json.loads((ROOT / "benchmarks.json").read_text())
    model, _ = load_model()
    module = MultiNative(model)
    try:
        record = retention.evaluate(module, benchmarks, output / "baseline.json", (False,) * 8, io.runtime_info())
        record["base_hash"] = module.base_hash()
        record["complete"] = True
        rl.atomic_json(output / "baseline.json", record)
        print("BASELINE COMPLETE", record["summary"], record["wiki_summary"], flush=True)
    finally: module.close()


@torch.no_grad()
def witnesses(model, tok, examples):
    result = []
    for e in examples:
        enc = tok(tasks.format_prompt(e), add_special_tokens=False, return_tensors="pt",
                  return_token_type_ids=False).to("cuda")
        hidden = model.model(**enc, use_cache=False).last_hidden_state
        result.append(model.lm_head(hidden[:, -1]).float().cpu())
    return result


def evaluate_tasks(module, tok, data, cfg, ws):
    identity = dict(delta=module.delta_hash(), runtime=ws.runtime, data=digest(data), endpoint="final")
    rel = "fit/evaluation/final.json"
    old = ws.pull(rel)
    cache = json.loads(old.read_text()) if old else dict(identity=identity, rows=[])
    assert cache["identity"] == identity
    rows = cache["rows"]
    seen = {(r["split"], r["mask"], r["id"]) for r in rows}
    expected = set()
    for split in ("test", "depth_probe"):
        masks = ["none", "all", "foreign", *[f"slot_{i}" for i in range(8)]] if split == "test" else list(rl.MASKS)
        for mask in masks:
            expected.update((split, mask, e["id"]) for e in data[split])
            missing = [e for e in data[split] if (split, mask, e["id"]) not in seen]
            for task in range(8):
                group = [e for e in missing if e["task_id"] == task]
                for start in range(0, len(group), cfg.eval_batch):
                    fresh = rl.predict(module, tok, group[start:start + cfg.eval_batch], mask, cfg,
                                       split, cfg.arm_name, cfg.seed)
                    for row in fresh: row.update(task_name=tasks.NAMES[task], endpoint="final", checkpoint_step=cfg.max_steps)
                    rows.extend(fresh)
                    if (start // cfg.eval_batch + 1) % 10 == 0 or start + cfg.eval_batch >= len(group):
                        ws.write_json(rel, dict(identity=identity, rows=rows))
                print(f"Final evaluation: {split}/{mask}/task_{task} {len(group)} new examples", flush=True)
    assert len(rows) == len(expected) and {(r['split'], r['mask'], r['id']) for r in rows} == expected
    ws.write_json("fit/predictions_final.json", dict(identity=identity, rows=rows))
    return rows


def metrics(rows):
    output, matrix = [], []
    for split, task, depth in sorted({(r["split"], r["task_id"], r["difficulty"]) for r in rows}):
        group = [r for r in rows if (r["split"], r["task_id"], r["difficulty"]) == (split, task, depth)]
        masks = {name: {r["id"]: r for r in group if r["mask"] == name} for name in {r["mask"] for r in group}}
        ids = set(masks["none"])
        assert all(set(m) == ids for m in masks.values())
        owner = f"slot_{task}" if split == "test" else "owner"
        def compare(a, b):
            return dict(lost=sum(masks[a][i]["correct"] and not masks[b][i]["correct"] for i in ids),
                rescued=sum(not masks[a][i]["correct"] and masks[b][i]["correct"] for i in ids),
                changed=sum(masks[a][i]["answer"] != masks[b][i]["answer"] for i in ids))
        row = dict(split=split, task_id=task, task=tasks.NAMES[task], depth=depth, n=len(ids),
            base=sum(r["correct"] for r in masks["none"].values()),
            owner=sum(r["correct"] for r in masks[owner].values()),
            all_on=sum(r["correct"] for r in masks["all"].values()),
            foreign=sum(r["correct"] for r in masks["foreign"].values()),
            owner_exact_trace=sum(r["trace_correct"] for r in masks[owner].values()),
            all_exact_trace=sum(r["trace_correct"] for r in masks["all"].values()),
            all_truncated=sum(r["truncated"] for r in masks["all"].values()),
            **{"composition_" + k: v for k, v in compare(owner, "all").items()},
            **{"foreign_" + k: v for k, v in compare("none", "foreign").items()})
        output.append(row)
        if split == "test":
            for slot in range(8):
                selected = f"slot_{slot}"
                matrix.append(dict(task_id=task, task=tasks.NAMES[task], slot=slot, depth=depth,
                    n=len(ids), base=row["base"], correct=sum(r["correct"] for r in masks[selected].values()),
                    **compare("none", selected)))
    return dict(performance=output, interference=matrix)


@dataclass
class Config(train.Config):
    cluster_plan_sha256: str = ""
    task_count: int = 1
    seed: int = 31
    arm_name: str = ""
    base_dtype: str = "float32"
    trajectory_microbatch: int = 1
    record_rollouts: bool = True


def worker(args, plan):
    setup()
    spec = next(j for j in plan["jobs"] if j["id"] == args._worker)
    fit = args.output / "fits" / spec["id"]
    cfg = Config(max_steps=plan["steps"], lr=plan["lr"], prompts_per_update=1 if args.smoke else 4,
        group_size=2 if args.smoke else 4, generation_batch=2, eval_batch=2 if args.smoke else 8, max_new_tokens=96,
        max_mean_kl=10. if args.smoke else .02, max_sequence_kl=10. if args.smoke else .1,
        max_prompt_tokens=1024, eval_every=plan["dev_every"],
        save_every=plan["save_every"], snapshot_every=plan["snapshot_every"], head_chunk=8, kl_head_chunk=8,
        preservation_mode="all_ab", audit_steps=[], cluster_plan_sha256=digest(plan),
        seed=spec["seed"], arm_name=spec["id"]).validate()
    data = json.loads((ROOT / "dataset.json").read_text())
    tasks.audit(data)
    data = {**data, **{s: [e for e in data[s] if e["task_id"] in ACTIVE_TASKS]
                      for s in ("train", "dev", "test", "depth_probe")}}
    if args.smoke:
        data = {**data, **{s: [next(e for e in data[s] if e["task_id"] == t) for t in ACTIVE_TASKS]
                          for s in ("train", "dev", "test", "depth_probe")}}
        def smoke_update(module, tok, items, advantages, config, optimizers, task, routing, pitems):
            result = update(module, tok, items, [1.] * len(items), config, optimizers, task, routing, pitems)
            result["scripted_smoke_advantages"] = True
            return result
        train.extended_update = smoke_update
    identity = dict(study=plan["study"], job=spec, config=asdict(cfg),
                    data_hash=digest(data), plan_sha256=digest(plan), source_hashes=plan["source_hashes"])
    ws = Workspace(fit / "local", fit / "durable", identity)
    for filename in plan["source_hashes"]:
        p = ws.path("source/" + filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / filename).read_bytes())
        ws.publish("source/" + filename)
    backup = DriveBackup(args.drive_remote)
    backup.restore(spec["id"], ws.durable, "fit/continuation.pt", ws.run_hash)
    ws.restore_training("fit", cfg)
    original_save = ws.save_training
    inventory = {}
    index_path = ws.durable / "fit/retained_files.json"
    if index_path.exists(): inventory = json.loads(index_path.read_text())
    def save():
        if shutil.disk_usage(ws.local).free < 40 * 2**30:
            raise OSError("Less than 40 GiB free; stopping before the next large checkpoint write")
        original_save("fit", cfg)
        for folder, pattern in (("rollouts", "*.json.gz"), ("snapshots", "*.pt")):
            for path in (ws.path("fit") / folder).glob(pattern):
                rel = str(path.relative_to(ws.local))
                if rel not in inventory:
                    ws.publish(rel, evidence=False)
                    inventory[rel] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        ws.write_json("fit/retained_files.json", inventory)
        resume = json.loads(ws.path("fit/resume.json").read_text())
        backup.mirror_evidence(spec["id"], ws.durable)
        backup.commit(spec["id"], ws.durable / "fit/continuation.pt", resume["step"], ws.run_hash)
    started = time.monotonic()
    model, tok = load_model()
    witness_examples = [next(e for e in data["dev"] if e["task_id"] == task) for task in ACTIVE_TASKS]
    raw = witnesses(model, tok, witness_examples)
    module = (BudgetNative(model, seed=spec["seed"], neurons=spec["channels"]) if spec["support"] == "native"
              else BudgetLoRA(model, seed=spec["seed"], rank=spec["rank"]))
    assert module.manifest()["trainable_scalars_per_slot"] == [spec["scalars_per_slot"]] * 8
    torch.cuda.reset_peak_memory_stats()
    try:
        ws.write_json("fit/slot_manifest.json", module.manifest())
        initial = {name: value.cpu() for name, value in module.state_dict().items()}
        if not (ws.durable / "fit/snapshots/initial.pt").exists():
            rl.atomic_torch(ws.path("fit/snapshots/initial.pt"), dict(step=0, deltas=module.state_dict(), manifest=module.manifest()))
        status = train.train_recipe(module, tok, data, cfg, ws.path("fit"), spec["recipe"], spec["seed"], save)
        if args.smoke: assert status["updates_with_movement"] > 0
        latest = rl._load(ws.path("fit/latest.pt"))
        assert latest["step"] == cfg.max_steps
        module.load_state_dict(latest["deltas"])
        if args.smoke:
            initialized_roles = sum(bool(state["state"]) for state in latest["optimizers"])
            assert initialized_roles == 1 and latest["optimizers"][1]["state"], initialized_roles
            full_data = json.loads((ROOT / "dataset.json").read_text())
            longest = max((e for e in full_data["train"] if e["task_id"] in ACTIVE_TASKS),
                          key=lambda e: len(tok(tasks.format_prompt(e), add_special_tokens=False).input_ids))
            enc = tok(tasks.format_prompt(longest), add_special_tokens=False, return_tensors="pt",
                      return_token_type_ids=False).to("cuda")
            with torch.no_grad(), module.gate((True,) * 8), cached():
                generated = module.model.generate(**enc, min_new_tokens=96, max_new_tokens=96,
                    do_sample=False, eos_token_id=None, pad_token_id=tok.pad_token_id,
                    temperature=None, top_p=None, top_k=None)
            assert generated.shape[1] - enc.input_ids.shape[1] == 96
            peak = torch.cuda.max_memory_allocated() / 2**30
            ws.write_json("fit/gpu_calibration.json", dict(prompt_tokens=enc.input_ids.shape[1],
                forced_new_tokens=96, peak_allocated_gib=peak, optimizer_roles_initialized=initialized_roles,
                budget=spec, scripted_smoke=True))
            assert peak < 7.25, f"Insufficient GPU headroom at this budget: {peak:.2f} GiB"
            del generated, enc
        rl.atomic_torch(ws.path("fit/final.pt"), dict(deltas=latest["deltas"], step=latest["step"], signature=latest["signature"]))
        ws.publish("fit/final.pt", evidence=False)
        ws.publish("fit/best.pt", evidence=False)
        assert {r["task_id"] for r in latest["history"]} == set(ACTIVE_TASKS)
        assert all(not state["state"] for i, state in enumerate(latest["optimizers"]) if i != 1)
        assert all(torch.equal(initial[name], p.detach().cpu())
                   for name, p, task in module._named_deltas if task not in ACTIVE_TASKS)
        moved_regions = {region: any(not torch.equal(initial[name], p.detach().cpu())
                        for name, p, task in module._named_deltas if task == 1 and region in name)
                        for region in (".mlp.", ".self_attn.")}
        if args.smoke: assert all(moved_regions.values()), moved_regions
        ws.write_json("fit/reuse_manifest.json", dict(model=MODEL, revision=REVISION,
            base_hash=module.base_hash(), job=spec, slot_manifest=module.manifest(),
            completed_task_slots=list(ACTIVE_TASKS), unused_slots=[t for t in range(8) if t not in ACTIVE_TASKS],
            unused_slots_unchanged=True, final_delta_file="final.pt",
            final_delta_sha256=hashlib.sha256(ws.path("fit/final.pt").read_bytes()).hexdigest(),
            optimizer_rng_file="continuation.pt", plan_sha256=digest(plan),
            loading="Instantiate BudgetNative/BudgetLoRA with this job's seed and width/rank on the pinned base, then load final.pt['deltas']. Keep all eight slots and this allocation for sequential A+B."))
        with module.gate((False,) * 8): off = witnesses(model, tok, witness_examples)
        error = max(float((a - b).abs().max()) for a, b in zip(raw, off))
        assert error == 0 and module.assert_frozen_base()
        measured = metrics(evaluate_tasks(module, tok, data, cfg, ws))
        ws.write_json("fit/metrics.json", measured)
        benchmarks = json.loads((ROOT / "benchmarks.json").read_text())
        if args.smoke:
            benchmarks = dict(benchmarks, examples=[next(e for e in benchmarks["examples"] if e["benchmark"] == name)
                              for name in benchmarks["protocol"]["primary_metrics"]], wiki_chunks=benchmarks["wiki_chunks"][:1])
            before = retention.evaluate(module, benchmarks, ws.path("fit/retention/base.json"), (False,) * 8, ws.runtime)
        else:
            before = json.loads((args.output / "baseline.json").read_text())
            assert before["complete"] and before["base_hash"] == module.base_hash()
            assert {k: before["identity"][k] for k in retention.identity(benchmarks, ws.runtime)} == retention.identity(benchmarks, ws.runtime)
        retention_results = {}
        for name in ["all"]:
            mask = rl.resolve_mask(module, name, 0)
            after = retention.evaluate(module, benchmarks, ws.path(f"fit/retention/{name}.json"), mask, ws.runtime)
            ws.publish(f"fit/retention/{name}.json")
            retention_results[name] = retention.changes(before, after, benchmarks)
        assert module.assert_frozen_base()
        audits = dict(all_off_max_logit_diff=error, witness_count=len(witness_examples), base_hash=module.base_hash(),
                      delta_hash=module.delta_hash(), precision="float32", unused_slots_unchanged=True,
                      active_tasks=list(ACTIVE_TASKS), both_regions_moved=all(moved_regions.values()),
                      region_movement=moved_regions)
        ws.write_json("fit/audits.json", audits)
        learning = [dict(task_id=t, task=tasks.NAMES[t],
                        updates=sum(r["task_id"] == t for r in latest["history"]),
                        reward_diverse_updates=sum(r["task_id"] == t and r["reward_diverse_groups"] > 0 for r in latest["history"]),
                        owner_moving_updates=sum(r["task_id"] == t and r["slot_max_move"][t] > 0 for r in latest["history"])) for t in ACTIVE_TASKS]
        completion = dict(spec, plan_sha256=digest(plan), metrics=measured, retention=retention_results,
            audits=audits, learning=learning, status=status, final_step=latest["step"],
            training_seconds=sum(r["seconds"] for r in latest["history"]),
            sampled_response_tokens=sum(r["response_tokens"] for r in latest["history"]),
            retained_file_count=len(inventory),
            seconds_this_session=time.monotonic() - started,
            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        ws.write_json("fit/completed.json", completion)
        ws.export()
        backup.mirror_evidence(spec["id"], ws.durable)
        marker = fit / "completed.pending.json"
        rl.atomic_json(marker, completion)
        backup.upload_file(marker, spec["id"] + "/completed.json")
        marker.replace(fit / "completed.json")
        print("COMPLETE AND LOCALLY BACKED UP:", spec["id"], flush=True)
    finally:
        module.close()
