"""Paired general-capability retention, with resumable per-item likelihoods."""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.nn.utils.parametrize import cached
from nr6_io import digest
from nr5_rl import atomic_json


@torch.no_grad()
def loglikelihood(model, ids, prefix, chunk=8):
    assert 0 < prefix < len(ids)
    tensor = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    hidden = model.model(input_ids=tensor[:, :-1], use_cache=False).last_hidden_state[0, prefix - 1:]
    targets = tensor[0, prefix:]
    total = 0.
    for start in range(0, len(targets), chunk):
        logits = model.lm_head(hidden[start:start + chunk]).float()
        total -= float(F.cross_entropy(logits, targets[start:start + chunk], reduction="sum"))
    return total


def identity(benchmarks, runtime):
    return dict(benchmarks=digest(benchmarks), runtime=runtime,
                scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def evaluate(module, benchmarks, path, mask, runtime):
    path = Path(path)
    expected = dict(identity(benchmarks, runtime), mask=list(mask), delta=module.delta_hash())
    record = json.loads(path.read_text()) if path.exists() else dict(identity=expected, rows=[], wiki=[])
    assert record["identity"] == expected
    seen = {row["id"] for row in record["rows"]}
    assert len(seen) == len(record["rows"])
    module.model.eval()
    with torch.no_grad(), module.gate(mask), cached():
        for e in benchmarks["examples"]:
            if e["id"] in seen:
                continue
            scores = [loglikelihood(module.model, e["context_ids"] + choice, len(e["context_ids"])) for choice in e["choice_ids"]]
            normed = [score / len(choice) for score, choice in zip(scores, e["choices"])]
            raw, normalized = max(range(len(scores)), key=scores.__getitem__), max(range(len(scores)), key=normed.__getitem__)
            record["rows"].append(dict(id=e["id"], benchmark=e["benchmark"], gold=e["gold"],
                prediction=raw, prediction_norm=normalized, correct=raw == e["gold"],
                correct_norm=normalized == e["gold"], scores=scores, scores_norm=normed))
            if len(record["rows"]) % 32 == 0:
                atomic_json(path, record)
                print(f"Retention {path.stem}: {len(record['rows'])}/{len(benchmarks['examples'])} questions", flush=True)
        for index in range(len(record["wiki"]), len(benchmarks["wiki_chunks"])):
            ids = benchmarks["wiki_chunks"][index]
            score = loglikelihood(module.model, ids, 1)
            record["wiki"].append(dict(index=index, nll=-score, tokens=len(ids) - 1))
            if (index + 1) % 8 == 0: atomic_json(path, record)
    assert len(record["rows"]) == len(benchmarks["examples"])
    assert module.delta_hash() == expected["delta"]
    results = []
    for name in sorted({e["benchmark"] for e in benchmarks["examples"]}):
        group = [r for r in record["rows"] if r["benchmark"] == name]
        results.append(dict(benchmark=name, n=len(group),
            acc=sum(r["correct"] for r in group) / len(group),
            acc_norm=sum(r["correct_norm"] for r in group) / len(group)))
    nll, tokens = sum(r["nll"] for r in record["wiki"]), sum(r["tokens"] for r in record["wiki"])
    record.update(summary=results, wiki_summary=dict(nll=nll, tokens=tokens, mean_nll=nll / tokens,
                                                   perplexity=math.exp(nll / tokens)))
    atomic_json(path, record)
    return record


def changes(before, after, benchmarks):
    assert before["identity"]["benchmarks"] == after["identity"]["benchmarks"] == digest(benchmarks)
    base = {r["id"]: r for r in before["rows"]}
    assert set(base) == {r["id"] for r in after["rows"]}
    result = []
    for name, metric in benchmarks["protocol"]["primary_metrics"].items():
        key = "correct_norm" if metric == "acc_norm" else "correct"
        pred = "prediction_norm" if metric == "acc_norm" else "prediction"
        group = [r for r in after["rows"] if r["benchmark"] == name]
        b, a = sum(base[r["id"]][key] for r in group), sum(r[key] for r in group)
        result.append(dict(benchmark=name, metric=metric, n=len(group), base=b / len(group), after=a / len(group),
            change_pp=100 * (a - b) / len(group),
            lost=sum(base[r["id"]][key] and not r[key] for r in group),
            rescued=sum(not base[r["id"]][key] and r[key] for r in group),
            changed=sum(base[r["id"]][pred] != r[pred] for r in group)))
    b, a = before["wiki_summary"], after["wiki_summary"]
    return dict(accuracy=result, language=dict(base_perplexity=b["perplexity"], after_perplexity=a["perplexity"],
        perplexity_change_percent=100 * (a["perplexity"] / b["perplexity"] - 1), mean_nll_change=a["mean_nll"] - b["mean_nll"]))
