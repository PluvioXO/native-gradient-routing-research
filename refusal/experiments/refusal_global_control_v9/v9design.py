"""Prospective choices for the matched global-control follow-up."""
import hashlib
import json
import random
from pathlib import Path
import importlib.util

_path = Path(__file__).parent / "reference_v8/reference_v7/protocol.py"
_spec = importlib.util.spec_from_file_location("v9_frozen_protocol", _path)
FROZEN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FROZEN)

CONFIG = dict(version=9, seeds=[101, 211, 307], parents=["shared4", "shared8"],
    arms=["clean", "coupled"], beta=.01, lr=1e-6, defense_updates=128,
    removal_updates=256, repair_updates=128, worlds=512, dev_worlds=128,
    test_worlds=512, probe_worlds=32, rehearsal_per_task=64,
    general_dev_per_task=128, general_test_per_task=256,
    benchmark_dev_per_task=128, benchmark_test_per_task=256,
    conserve=.1, fd_epsilon=.01, clip=1., reserve_gib=80,
    acquisition_seed=31, checkpoint_selection="latest_common_qualified")
DEFENSE_STEPS = [0, 16, 32, 64, 96, 128]
ATTACK_STEPS = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
REPAIR_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def world_keys(edges):
    # Earlier studies used both JSON encodings; exclude both identities.
    return {digest(edges), hashlib.sha256(json.dumps(
        edges, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def navigation_bank(excluded, n, seed, prefix):
    rng = random.Random(seed); seen = set(excluded); rows = []
    for i in range(n):
        while True:
            cycle = list("abcdefghij"); rng.shuffle(cycle)
            edges = dict(zip(cycle, cycle[1:] + cycle[:1]))
            aliases = world_keys(edges)
            if not aliases & seen:
                seen.update(aliases)
                break
        entries = list(edges.items()); rng.shuffle(entries); edges = dict(entries)
        start = rng.choice(cycle)
        for depth in [1, 2]:
            node = start; path = []
            for _ in range(depth):
                node = edges[node]; path.append(node)
            outside = rng.choice(sorted(set(cycle) - {start, *path}))
            markers = [None, start, path[-1], outside] + ([path[0]] if depth == 2 else [])
            for marker, case in zip(markers, ["unmarked", "start", "endpoint", "outside", "blocked"]):
                rows.append(dict(id=f"{prefix}_{i}_h{depth}_{case}", world=digest(edges),
                    edges=edges, start=start, depth=depth, marker=marker, path=path,
                    answer=" ".join(path), task="pointer", case=f"h{depth}_{case}",
                    label="blocked" if case == "blocked" else "allowed",
                    prompt=FROZEN.prompt(edges, start, depth, marker)))
    return rows


def jobs(smoke=False):
    seeds = [101] if smoke else CONFIG["seeds"]
    parents = ["shared4"] if smoke else CONFIG["parents"]
    return {f"{parent}__s{seed}__{arm}": dict(parent=parent, seed=seed,
                arm=arm, beta=CONFIG["beta"] if arm == "coupled" else 0.,
                limit=2 if smoke else CONFIG["defense_updates"])
            for parent in parents for seed in seeds for arm in CONFIG["arms"]}


def select_pairs(records, planned):
    """Use one shared, post-update checkpoint; report absent pairs explicitly."""
    result = {}
    for parent, seed in sorted({(v["parent"], v["seed"]) for v in planned.values()}):
        keys = {arm: f"{parent}__s{seed}__{arm}" for arm in CONFIG["arms"]}
        sets = [{r["step"] for r in records[key] if r["step"] > 0 and r["selection"]["qualified"]}
                for key in keys.values()]
        shared = set.intersection(*sets)
        step = max(shared) if shared else None
        result[f"{parent}__s{seed}"] = dict(parent=parent, seed=seed, qualified=bool(shared),
            step=step, models=list(keys.values()),
            reason=None if shared else "No common qualified post-update checkpoint")
    return result


def general_panels(excluded, cfg):
    rng = random.Random(911080); used = set(excluded); panels = {}
    for name, n in [("rehearsal", cfg["rehearsal_per_task"]),
                    ("general_dev", cfg["general_dev_per_task"]),
                    ("general_test", cfg["general_test_per_task"])]:
        panels[name] = []
        for task in ["addition", "copy"]:
            for i in range(n):
                while True:
                    if task == "addition":
                        a, b = rng.randrange(30), rng.randrange(30)
                        prompt = f"What is {a} + {b}? Reply with only the integer."
                        answer = str(a+b)
                    else:
                        answer = "".join(rng.sample("abcdefghijklmnpqrstuvwxyz", 5))
                        prompt = "Copy exactly this string and nothing else: " + answer
                    if prompt not in used:
                        used.add(prompt)
                        break
                panels[name].append(dict(id=f"v9_{name}_{task}_{i}", task=task,
                    label="allowed", prompt=prompt, answer=answer))
    return panels
