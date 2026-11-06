"""Audit archived final measurements and produce one record per continuation bank."""
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE / "results/casr-adaptive-ablation-20261030"
OUT = HERE / "analysis"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-9), (a, b)


def main():
    if not ROOT.exists():raise SystemExit('Raw measurements were removed during the 5 November cleanup. See aaai/RAW_DATA.md to restore them; analysed tables and plots remain in this study.')
    OUT.mkdir(exist_ok=True)
    inventory = read(ROOT / "export_inventory.json")
    for name, expected in inventory.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert sha(ROOT / name) == expected, name
    print("Verified exported files:", len(inventory), flush=True)
    folder = ROOT / "outputs/main"
    plan = read(folder / "experiment_plan.json")
    complete = read(folder / "SHARD_COMPLETE.json")
    assert complete["completed"] == 48 and not complete["failed"]
    for name, expected in plan["source_hashes"].items():
        assert sha(ROOT / "source" / name) == expected, name
    sys.path[:0] = [str(ROOT / "source/reference"), str(ROOT / "source/reference/src")]
    import compound_tasks
    from rlq_tasks import extract_answer
    data = read(ROOT / "source/reference/dataset.json")
    compound = compound_tasks.build(data)
    gold = {e["id"]: str(e["answer"]) for split in ("dev", "test") for e in data[split] + compound[split]}
    benchmarks = read(ROOT / "source/reference/benchmarks.json")
    questions = {e["id"]: e for e in benchmarks["examples"]}
    counts = Counter()
    results, task_results, control_results = [], [], []
    base_signature = None

    def predictions(path):
        record = read(path)
        rows = record["rows"]
        assert len({r["id"] for r in rows}) == len(rows)
        for row in rows:
            assert str(row["gold"]) == gold[row["id"]]
            if "completion" in row:
                answer = extract_answer(row["completion"])
                assert answer == row["answer"]
                assert row["correct"] == (answer == row["gold"])
                assert row["valid"] == (answer is not None)
            else:
                assert "first" in row and "second" in row
                for stage in (row["first"], row["second"]):
                    if stage is not None:
                        assert stage["answer"] == extract_answer(stage["completion"])
                assert row["correct"] == bool(row["second"] and row["second"]["answer"] == row["gold"])
            counts["prediction_rows"] += 1
        return rows

    def retention(path):
        record = read(path)
        assert record["identity"]["benchmarks"] == digest(benchmarks)
        assert record["identity"]["scorer_sha256"] == sha(ROOT / "source/reference/src/retention.py")
        assert len(record["rows"]) == len(questions) == 996
        assert {r["id"] for r in record["rows"]} == set(questions)
        for row in record["rows"]:
            e = questions[row["id"]]
            assert row["gold"] == e["gold"]
            scores = row["scores"]
            norm = [s / len(c) for s, c in zip(scores, e["choices"])]
            for a, b in zip(norm, row["scores_norm"]):
                close(a, b)
            for values, pred, key in ((scores, "prediction", "correct"), (norm, "prediction_norm", "correct_norm")):
                assert row[pred] == max(range(len(values)), key=values.__getitem__)
                assert row[key] == (row[pred] == row["gold"])
            counts["retention_rows"] += 1
        for row in record["summary"]:
            group = [r for r in record["rows"] if r["benchmark"] == row["benchmark"]]
            assert row["n"] == len(group)
            close(row["acc"], statistics.mean(r["correct"] for r in group))
            close(row["acc_norm"], statistics.mean(r["correct_norm"] for r in group))
        assert len(record["wiki"]) == len(benchmarks["wiki_chunks"])
        for i, row in enumerate(record["wiki"]):
            assert row["index"] == i and row["tokens"] == len(benchmarks["wiki_chunks"][i]) - 1
            assert math.isfinite(row["nll"])
        nll, tokens = sum(r["nll"] for r in record["wiki"]), sum(r["tokens"] for r in record["wiki"])
        close(nll, record["wiki_summary"]["nll"])
        assert tokens == record["wiki_summary"]["tokens"] == 16384
        close(math.exp(nll / tokens), record["wiki_summary"]["perplexity"])
        counts["wiki_target_tokens"] += tokens
        return record

    for job in plan["jobs"]:
        fit = folder / "fits" / job["id"]
        done = read(fit / "completed.json")
        assert done["spec"] == job and done["plan_sha256"] == digest(plan)
        assert done["all_off_equal"]
        if job["kind"] == "main":
            assert done["final_step"] == 640 and done["status"]["complete"]
            assert sha(fit / "bank_final.pt") == done["bank_sha256"]
            history = [json.loads(line) for line in (fit / "local/fit/history.jsonl").read_text().splitlines()]
            assert [r["step"] for r in history] == list(range(1, 641))
            assert Counter(r["task_id"] for r in history) == {i: 80 for i in range(8)}
            assert all(r["foreign_optimizers_unchanged"] for r in history)
            for row in history:
                path = fit / "local/fit" / row["rollout_file"]
                assert sha(path) == row["rollout_sha256"]
            assert sum(r["accepted"] for r in history) == done["accepted_updates"]
            counts["training_updates"] += len(history)
            for path in (fit / "local/fit/dev_predictions").glob("*.json"):
                predictions(path)
            continue
        for name, expected in done["artifact_hashes"].items():
            assert sha(fit / name) == expected
        train = read(folder / "fits" / job["bank_checkpoint"] / "completed.json")
        manifest = read(fit / "composition_manifest.json")
        assert manifest["source_owners"] == {job["bank_checkpoint"]: train["bank_sha256"]}
        assert manifest["all_off_equal"]
        for path in sorted(fit.rglob("*.json")):
            if path.parent.name in ("evaluation", "calibration", "compounds"):
                predictions(path)
        for item in done["composition"]:
            suffix = "_".join(map(str, item["active_slots"])) or "none"
            rows = read(fit / f"evaluation/task_{item['task']}_set_{suffix}.json")["rows"]
            assert item["n"] == len(rows) == 120
            close(item["accuracy"], statistics.mean(r["correct"] for r in rows))
            solo = {r["id"]: r for r in read(fit / f"evaluation/task_{item['task']}_set_{item['task']}.json")["rows"]}
            base_rows = {r["id"]: r for r in read(fit / f"evaluation/task_{item['task']}_set_none.json")["rows"]}
            assert set(solo) == set(base_rows) == {r["id"] for r in rows}
            close(item["standalone_accuracy"], statistics.mean(r["correct"] for r in solo.values()))
            close(item["base_accuracy"], statistics.mean(r["correct"] for r in base_rows.values()))
            for key, count in (
                ("lost", sum(solo[r["id"]]["correct"] and not r["correct"] for r in rows)),
                ("rescued", sum(not solo[r["id"]]["correct"] and r["correct"] for r in rows)),
                ("changed", sum(solo[r["id"]]["answer"] != r["answer"] for r in rows))):
                assert item[key] == count
            task_results.append(dict(support=job["support"], mode=job["mode"], seed=job["seed"], **item))
        assert len(done["composition"]) == 120
        curves = {k: statistics.mean(r["accuracy"] for r in done["composition"] if r["active_count"] == k)
                  for k in (0, 1, 2, 4, 7, 8)}
        base = retention(fit / "retention/base.json")
        signature = digest(dict(rows=base["rows"], wiki=base["wiki"]))
        if base_signature is None:
            base_signature = signature
        assert signature == base_signature, "Frozen baseline changed across evaluations"
        for path in sorted((fit / "retention").glob("*.json")):
            if path.name == "base.json":
                continue
            after = retention(path)
            reported = done["calibrated_retention"] if path.stem == "calibrated_all" else done["retention"][path.stem]
            changes = []
            before = {r["id"]: r for r in base["rows"]}
            for b in reported["accuracy"]:
                key = "correct_norm" if b["metric"] == "acc_norm" else "correct"
                rows = [r for r in after["rows"] if r["benchmark"] == b["benchmark"]]
                change = 100 * statistics.mean(int(r[key]) - int(before[r["id"]][key]) for r in rows)
                close(change, b["change_pp"])
                close(statistics.mean(r[key] for r in rows), b["after"])
                close(statistics.mean(before[r["id"]][key] for r in rows), b["base"])
                pred = "prediction_norm" if key == "correct_norm" else "prediction"
                assert b["lost"] == sum(before[r["id"]][key] and not r[key] for r in rows)
                assert b["rescued"] == sum(not before[r["id"]][key] and r[key] for r in rows)
                assert b["changed"] == sum(before[r["id"]][pred] != r[pred] for r in rows)
                changes.append((change, len(rows)))
                control_results.append(dict(support=job["support"], mode=job["mode"], seed=job["seed"], mask=path.stem, **b))
            ppl = 100 * (after["wiki_summary"]["perplexity"] / base["wiki_summary"]["perplexity"] - 1)
            close(ppl, reported["language"]["perplexity_change_percent"])
            if path.stem == "set_0_1_2_3_4_5_6_7":
                primary_ppl, primary_mc = ppl, sum(v * n for v, n in changes) / sum(n for _, n in changes)
        coefficients = [1.] * 8
        calibration = read(fit / "calibration.json")
        for task in range(8):
            scores = []
            for value in (1., .5, 1.5, 0.):
                rows = read(fit / f"calibration/task_{task}_value_{value:g}.json")["rows"]
                score = statistics.mean(r["correct"] for r in rows)
                trial = calibration["trials"][task * 4 + len(scores)]
                candidate = list(coefficients); candidate[task] = value
                assert trial["coefficients"] == candidate
                close(score, trial["macro_accuracy"])
                scores.append((score, value))
            coefficients[task] = next(v for s, v in scores if s == max(x[0] for x in scores))
        assert coefficients == calibration["coefficients"] == done["coefficients"]
        calibrated = statistics.mean(r["correct"] for r in read(fit / "evaluation/calibrated_all.json")["rows"])
        close(calibrated, done["calibrated_accuracy"])
        close(statistics.mean(r["correct"] for r in read(fit / "compounds/calibrated_all.json")["rows"]), done["calibrated_compound_accuracy"])
        for item in done["compounds"]:
            suffix = item.get("control")
            if suffix == "generated_intermediate_pipeline":
                suffix = "pipeline"
            if suffix is None:
                suffix = "set_" + ("_".join(map(str, item["mask"])) or "none")
            rows = read(fit / f"compounds/{item['family']}_{suffix}.json")["rows"]
            assert item["n"] == len(rows) and item["correct"] == sum(r["correct"] for r in rows)
        results.append(dict(support=job["support"], mode=job["mode"], seed=job["seed"],
            **{f"accuracy_{k}": 100 * v for k, v in curves.items()}, wiki_change=primary_ppl,
            mc_change=primary_mc, calibrated_accuracy=100 * calibrated,
            training_seconds=train["training_seconds"], response_tokens=train["response_tokens"],
            accepted_updates=train["accepted_updates"], peak_cuda_gib=train["peak_cuda_gib"]))
        print("Audited", job["id"], flush=True)
    assert len(results) == 24 and counts["training_updates"] == 15360
    for name, value in (("per_bank", results), ("per_task", task_results), ("controls", control_results)):
        (OUT / (name + ".json")).write_text(json.dumps(value, indent=2))
    with (OUT / "per_bank.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    audit = dict(complete=True, training_banks=24, evaluated_banks=24, verified_export_files=len(inventory),
        plan_sha256=digest(plan), archive_inventory_sha256=sha(ROOT / "export_inventory.json"),
        counts=dict(counts), frozen_baseline_identical=True, source_hashes_verified=True,
        limitation="Recomputed answers, aggregate likelihoods and counts; did not rerun model forwards or optimization.")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
