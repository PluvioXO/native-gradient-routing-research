"""Held-out compositions of existing primitives; no compound examples enter RL."""
import hashlib
import random
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import multi_tasks as tasks
import nr5_tasks
import rlq_tasks

FAMILIES = (("pointer_arithmetic", (1, 0)), ("stack_sorting", (5, 4)))


def make_row(family, first, second, index):
    if family == "pointer_arithmetic":
        operations = re.search(r"Operations: (.+)\.\n", second["problem"]).group(1)
        intermediate = rlq_tasks.solve(1, first["problem"])
        problem2 = re.sub(r"Start at [0-9]\.", f"Start at {intermediate['answer']}.", second["problem"])
        solved = rlq_tasks.solve(0, problem2)
        query = (f"Stage 1: {first['problem']}\nStage 2: Use the final digit from stage 1 as the starting digit. "
                 f"Apply these operations modulo 10: {operations}.\nReturn the final digit after both stages.")
        second_query = dict(second, problem=problem2, **solved)
        trace = intermediate["gold_trace"] + solved["gold_trace"][1:]
    else:
        intermediate = tasks.solve(5, first["payload"])
        digits = intermediate["gold_trace"]
        payload = dict(digits=digits, k=1 + index % len(digits), direction=second["payload"]["direction"])
        solved = tasks.solve(4, payload)
        query = (f"Stage 1: {tasks.render(5, first['payload'])}\nStage 2: Collect the initial top digit and the top digit "
                 f"after EVERY instruction, including duplicates. Return the {payload['k']}th {payload['direction']} "
                 "digit in that list, counting duplicates.")
        second_query = dict(second, payload=payload, problem=tasks.render(4, payload), **solved)
        trace = digits + solved["gold_trace"]
    identity = hashlib.sha256(f"{family}|{first['id']}|{second['id']}|{index}".encode()).hexdigest()[:24]
    return dict(id="compound_" + identity, task_id=first["task_id"], difficulty=first["difficulty"],
        family=family, components=list(dict(FAMILIES)[family]), problem=query, answer=solved["answer"],
        gold_trace=trace, first=dict(first), second=second_query)


def build(data):
    result = {"dev": [], "test": []}
    demos = {}
    for family, (a, b) in FAMILIES:
        for split in ("train", "dev", "test"):
            left = [e for e in data[split] if e["task_id"] == a]
            right = [e for e in data[split] if e["task_id"] == b]
            random.Random(f"composition-v1:{family}:{split}").shuffle(right)
            rows = [make_row(family, first, second, i) for i, (first, second) in enumerate(zip(left, right))]
            if split == "train": demos[family] = rows[:2]
            else: result[split].extend(rows)
    result["demos"] = demos
    return result


def prompt(example, demos):
    rules = ("Follow both stages in order. Pointer maps give successor digits; arithmetic uses remainder modulo 10. "
             "For stack operations, ADD/MUL use modulo 10; the stack is shown bottom to top. "
             "For sorting, duplicates count separately. Show your work. End with a line: Answer: d")
    parts = [rules]
    for e in demos[example["family"]]:
        parts.append(f"Problem: {e['problem']}\nWork: " + " -> ".join(map(str, e["gold_trace"])) + f"\nAnswer: {e['answer']}")
    parts.append(f"Problem: {example['problem']}\nWork:")
    return "\n\n".join(parts)


def intermediate_query(example, first_completion):
    """Construct a standard second-stage prompt using only the generated intermediate."""
    if example["family"] == "pointer_arithmetic":
        answer = rlq_tasks.extract_answer(first_completion)
        if answer is None: return None
        second = dict(example["second"])
        second["problem"] = re.sub(r"Start at [0-9]\.", f"Start at {answer}.", second["problem"])
        return second
    match = re.search(r"Work:\s*(.*?)\n\s*Answer:", "Work:" + first_completion, re.S)
    if match is None: return None
    text = match.group(1).strip()
    if text.startswith("Work:"): text = text[5:].strip()
    if re.fullmatch(r"[0-9](?:\s*(?:->|→)\s*[0-9])*", text) is None: return None
    digits = [int(x) for x in re.findall(r"[0-9]", text)]
    payload = dict(example["second"]["payload"], digits=digits)
    if len(digits) < payload["k"]: return None
    return dict(example["second"], payload=payload, problem=tasks.render(4, payload))
