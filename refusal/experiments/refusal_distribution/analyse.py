"""Recount the fixed-budget refusal study from archived response records."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np


def main():
    if not (Path(__file__).resolve().parent/'results').exists():raise SystemExit('Raw measurements were removed during the 5 November cleanup. See refusal/RAW_DATA.md to restore them; analysed tables and plots remain in this study.')
    here = Path(__file__).resolve().parent
    root = here / "results/casr-refusal-distribution-20261030"
    out = here / "analysis"
    out.mkdir(exist_ok=True)
    read = lambda p: json.loads(p.read_text())
    digest = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
    tree = ast.parse((root / "source/removal.py").read_text())
    names = {"classify", "metrics", "baseline_ok", "qualifies"}
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                and n.name in names], type_ignores=[])
    scoring = dict(re=re, np=np)
    exec(compile(functions, "archived_scoring", "exec"), scoring)
    acquisitions, summaries, records = [], [], []
    audited_rows, evaluations = 0, 0
    plan = read(root / "plan.json")
    assert len(plan["jobs"]) == 9
    for job in plan["jobs"]:
        run = root / "runs" / job["name"]
        manifest, data = read(run / "manifest.json"), read(run / "data.json")
        assert manifest["config"] == plan["configurations"][job["name"]]
        assert digest(data) == manifest["data_sha256"]
        for name, sha in manifest["sources"].items():
            assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == sha
            assert (run / "source" / name).read_bytes() == (root / "source" / name).read_bytes()
        cfg = SimpleNamespace(**manifest["config"])

        def audit(report, panel):
            nonlocal audited_rows
            rows = report["rows"]
            expected = {r["id"]: r for r in data["benchmarks" if panel == "general" else panel]}
            assert len(rows) == len(expected) and {r["id"] for r in rows} == set(expected)
            for row in rows:
                assert row["prompt"] == expected[row["id"]]["prompt"]
                if panel == "general":
                    assert row["predicted"] == int(np.argmax(row["choice_logprobs"]))
                    assert row["correct"] == (row["predicted"] == row["gold"])
                    assert math.isclose(row["gold_nll"], -row["choice_logprobs"][row["gold"]], abs_tol=1e-6)
                else:
                    assert row["answer"] == expected[row["id"]]["answer"]
                    assert all(row[k] == v for k, v in scoring["classify"](row["response"], row["answer"]).items())
            if panel == "general":
                measures = {task + "_" + key: float(np.mean([r[key] for r in rows if r["task"] == task]))
                            for task in {r["task"] for r in rows} for key in ("correct", "gold_nll", "choice_mass")}
            else:
                measures = scoring["metrics"](rows)
            assert measures.keys() == report["metrics"].keys()
            assert all(math.isclose(v, report["metrics"][k], abs_tol=1e-8) for k, v in measures.items())
            audited_rows += len(rows)

        baseline = read(run / "baseline.json")
        for panel, report in baseline.items():
            audit(report, panel)
        assert scoring["baseline_ok"](baseline["dev"], cfg)
        directory = run / f"seed_{job['seed']}" / "channels"
        acquisition, allocation = read(directory / "acquisition.json"), read(directory / "allocation.json")
        assert allocation["trainable"] == sum(m["count"] for m in allocation["matrices"]) == 10752
        assert [x["layer"] for x in allocation["channels"]] == cfg.layer_indices
        passed = []
        for p in sorted(directory.glob("acquire_*/dev_*.json")):
            report = read(p)
            audit(report, "dev")
            if scoring["qualifies"](report, baseline["dev"], cfg):
                passed.append(p)
        assert bool(passed) == acquisition["qualified"]
        assert acquisition["seed"] == job["seed"]
        acquisitions.append(dict(job=job["name"], condition=job["condition"], **acquisition))
        if not acquisition["qualified"]:
            assert read(run / "status.json")["status"] == "inconclusive"
            continue
        assert read(run / "status.json")["status"] == "complete"
        before = read(directory / "before_removal.json")
        for panel, report in before.items():
            audit(report, panel)
        for lr in cfg.removal_lrs:
            removal = directory / f"removal_lr_{lr:g}"
            complete = read(removal / "complete.json")
            assert complete["fingerprint"] == digest(manifest) and complete["final_step"] == 64
            paths = sorted(removal.glob("eval_*.json"))
            assert len(paths) == len(cfg.removal_steps)
            curve = []
            for step, path in zip(cfg.removal_steps, paths):
                value = read(path)
                assert value["step"] == step and value["examples_seen"] == step * 4
                assert value["epochs"] == step / 128
                for panel, report in value["panels"].items():
                    audit(report, panel)
                    if step == 0:
                        assert report == before[panel]
                    records.append(dict(condition=job["condition"], seed=job["seed"], lr=lr,
                                        step=step, panel=panel, **report["metrics"]))
                curve.append(value)
                evaluations += 1
            def first(key, predicate):
                return next((v["step"] for v in curve if predicate(v["panels"]["test"]["metrics"][key])), None)
            summary = dict(condition=job["condition"], seed=job["seed"], lr=lr,
                first_zero=first("blocked_refusal", lambda x: x == 0),
                first_90_correct=first("blocked_correct", lambda x: x >= .9),
                initial=before["test"]["metrics"], final=curve[-1]["panels"]["test"]["metrics"])
            summary["controls"] = {key: dict(base=baseline[panel]["metrics"][key],
                before=before[panel]["metrics"][key], after=curve[-1]["panels"][panel]["metrics"][key])
                for panel, key in (("utility", "utility_correct"), ("general", "arc_easy_correct"),
                                   ("general", "winogrande_correct"))}
            summaries.append(summary)
    assert len(acquisitions) == 9 and len(summaries) == 14 and evaluations == 406
    audit_record = dict(acquisitions=9, qualified=7, removal_trajectories=14,
        removal_evaluations=evaluations, audited_rows=audited_rows, metrics_recomputed=True,
        archive_sha256=hashlib.sha256((here / "results/casr-refusal-distribution-measurements-20261031.tar.gz").read_bytes()).hexdigest())
    for name, value in (("audit", audit_record), ("acquisition", acquisitions), ("summary", summaries), ("records", records)):
        (out / (name + ".json")).write_text(json.dumps(value, indent=2))
    with (out / "thresholds.csv").open("w") as stream:
        fields = ["condition", "seed", "lr", "first_zero", "first_90_correct"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(audit_record, indent=2))
    for condition in ("single12", "dual08_16", "four03_09_15_21"):
        selected = [x for x in summaries if x["condition"] == condition]
        print(condition)
        for lr in (1e-6, 1e-5):
            print(lr, [(x["seed"], x["first_zero"], x["first_90_correct"]) for x in selected if x["lr"] == lr])
        for key in ("utility_correct", "arc_easy_correct", "winogrande_correct"):
            changes = [100 * (x["controls"][key]["after"] - x["controls"][key]["before"]) for x in selected]
            print(key, "change pp: mean/min/max", np.mean(changes), min(changes), max(changes))


if __name__ == "__main__":
    main()
