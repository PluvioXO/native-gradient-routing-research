"""Eight deterministic reasoning families, public inputs, and independent solvers."""
from collections import Counter, deque
import hashlib
import json
import random
import nr5_data
import nr5_tasks
import rlq_tasks

NAMES = ("arithmetic", "pointer", "boolean", "brackets", "sorting", "stack", "shortest_path", "sudoku")
RULES = {
    2: "Evaluate AND, OR, XOR on bits. Work lists values in postorder: left subtree, right subtree, then operator, including leaves.",
    3: "Check that (), [], {} are correctly matched and nested. Answer 1 if valid, else 0. Work starts at depth 0, then gives depth after each character; use 9 from the first mismatched closing onward. Unclosed openings also invalidate the string.",
    4: "Return the requested kth smallest or largest digit, counting duplicates. Work lists all digits sorted in the requested order.",
    5: "Run the stack program, bottom to top. PUSH d pushes d; DUP copies top; SWAP exchanges top two; POP removes top. ADD or MUL replaces the top two by their sum or product modulo 10. Work gives initial top, then top after each instruction. Answer is final top.",
    6: "Find the shortest directed path from start to target. Answer is its number of edges. Work is the path; break ties lexicographically by node digits. Empty adjacency lists have no outgoing edges.",
    7: "Solve the 4x4 Sudoku: each row, column and 2x2 box contains 1,2,3,4 once. Dots are blanks. Answer the target cell. Work lists all missing values in row-major order.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def shortest(adj, start, target):
    queue, seen = deque([[start]]), {start}
    while queue:
        path = queue.popleft()
        if path[-1] == target:
            return path
        for node in sorted(adj[str(path[-1])]):
            if node not in seen:
                seen.add(node)
                queue.append(path + [node])
    return None


def sudoku_solutions(grid, limit=2):
    if len(grid) != 16 or any(v not in range(5) for v in grid):
        raise ValueError("Expected 16 Sudoku cells in 0..4")
    groups = [grid[r * 4:r * 4 + 4] for r in range(4)] + [[grid[r * 4 + c] for r in range(4)] for c in range(4)]
    groups += [[grid[r * 4 + c] for r in range(a, a + 2) for c in range(b, b + 2)] for a in (0, 2) for b in (0, 2)]
    if any(len([v for v in group if v]) != len({v for v in group if v}) for group in groups):
        return []
    results, board = [], list(grid)
    def search():
        if len(results) >= limit:
            return
        choices = []
        for index, value in enumerate(board):
            if value:
                continue
            row, col = divmod(index, 4)
            used = set(board[row * 4:row * 4 + 4]) | {board[r * 4 + col] for r in range(4)}
            used |= {board[r * 4 + c] for r in range(row // 2 * 2, row // 2 * 2 + 2)
                     for c in range(col // 2 * 2, col // 2 * 2 + 2)}
            available = sorted({1, 2, 3, 4} - used)
            if not available:
                return
            choices.append((len(available), index, available))
        if not choices:
            results.append(board.copy())
            return
        _, index, available = min(choices)
        for value in available:
            board[index] = value
            search()
        board[index] = 0
    search()
    return results


def solve(task, payload):
    if task == 2:
        trace = []
        def visit(node):
            if isinstance(node, int):
                result = node
            else:
                op, left, right = node
                a, b = visit(left), visit(right)
                result = {"AND": a & b, "OR": a | b, "XOR": a ^ b}[op]
            trace.append(result)
            return result
        answer = visit(payload["tree"])
    elif task == 3:
        stack, trace, bad = [], [0], False
        closes = {")": "(", "]": "[", "}": "{"}
        for char in payload["text"]:
            if not bad:
                if char in "([{":
                    stack.append(char)
                elif not stack or stack.pop() != closes[char]:
                    bad = True
            trace.append(9 if bad else len(stack))
        answer = int(not bad and not stack)
    elif task == 4:
        trace = sorted(payload["digits"], reverse=payload["direction"] == "largest")
        answer = trace[payload["k"] - 1]
    elif task == 5:
        stack = payload["initial"].copy()
        trace = [stack[-1]]
        for instruction in payload["ops"]:
            op = instruction[0]
            if op == "PUSH": stack.append(instruction[1])
            elif op == "DUP": stack.append(stack[-1])
            elif op == "SWAP": stack[-2:] = stack[-2:][::-1]
            elif op == "POP": stack.pop()
            else:
                b, a = stack.pop(), stack.pop()
                stack.append((a + b if op == "ADD" else a * b) % 10)
            trace.append(stack[-1])
        answer = stack[-1]
    elif task == 6:
        trace = shortest(payload["adj"], payload["start"], payload["target"])
        assert trace is not None
        answer = len(trace) - 1
    elif task == 7:
        solutions = sudoku_solutions(payload["grid"])
        assert len(solutions) == 1, "Sudoku must have exactly one solution"
        solution = solutions[0]
        trace = [solution[i] for i, value in enumerate(payload["grid"]) if not value]
        answer = solution[payload["target"]]
    else:
        raise ValueError(task)
    return dict(answer=str(answer), gold_trace=trace)


def render(task, p):
    if task == 2:
        def text(node):
            return str(node) if isinstance(node, int) else f"({text(node[1])} {node[0]} {text(node[2])})"
        return text(p["tree"])
    if task == 3: return p["text"]
    if task == 4: return f"Digits: {','.join(map(str, p['digits']))}. k={p['k']}; {p['direction']}."
    if task == 5:
        ops = "; ".join(" ".join(map(str, op)) for op in p["ops"])
        return f"Initial stack: {','.join(map(str, p['initial']))}. Program: {ops}."
    if task == 6:
        adj = "; ".join(f"{node}:{','.join(map(str, p['adj'][str(node)])) or '-'}" for node in range(10))
        return f"Edges: {adj}. Start {p['start']}; target {p['target']}."
    if task == 7:
        rows = ["".join(str(v) if v else "." for v in p["grid"][r * 4:r * 4 + 4]) for r in range(4)]
        row, col = divmod(p["target"], 4)
        return f"Grid: {' / '.join(rows)}. Target: row {row + 1}, column {col + 1}."
    raise ValueError(task)


def candidate(task, depth, rng):
    if task == 2:
        def tree(n):
            if not n: return rng.randrange(2)
            left = rng.randrange(n)
            return [rng.choice(("AND", "OR", "XOR")), tree(left), tree(n - 1 - left)]
        p = dict(tree=tree(depth + 2))
        family = p
    elif task == 3:
        stack, chars, opened = [], [], 0
        for _ in range(2 * (depth + 2)):
            if opened < depth + 2 and (not stack or rng.random() < .55):
                left, right = rng.choice((('(', ')'), ('[', ']'), ('{', '}')))
                chars.append(left); stack.append(right); opened += 1
            else: chars.append(stack.pop())
        family = "".join(chars)
        if rng.randrange(2):
            pairs = [(i, j) for i in range(len(chars)) for j in range(i + 1, len(chars))
                     if chars[i] in ")]}" and chars[j] in ")]}" and chars[i] != chars[j]]
            if not pairs: return None
            a, b = rng.choice(pairs)
            # Keep every character count and the open/close depth trajectory.
            chars[a], chars[b] = chars[b], chars[a]
        p = dict(text="".join(chars))
    elif task == 4:
        p = dict(digits=[rng.randrange(10) for _ in range(depth + 4)],
                 k=rng.randrange(1, depth + 5), direction=rng.choice(("smallest", "largest")))
        family = sorted(p["digits"])
    elif task == 5:
        initial, ops, size = [rng.randrange(10) for _ in range(2)], [], 2
        for _ in range(depth + 3):
            choices = (["PUSH", "DUP"] if size < 5 else []) + (["SWAP", "POP", "ADD", "MUL"] if size > 1 else [])
            op = rng.choice(choices)
            ops.append([op, rng.randrange(10)] if op == "PUSH" else [op])
            size += 1 if op in ("PUSH", "DUP") else (-1 if op in ("POP", "ADD", "MUL") else 0)
        p = dict(initial=initial, ops=ops); family = p
    elif task == 6:
        nodes = list(range(10)); rng.shuffle(nodes)
        adj = {str(i): [] for i in range(10)}
        for a, b in zip(nodes[:depth], nodes[1:depth + 1]): adj[str(a)].append(b)
        start, target = nodes[0], nodes[depth]
        pairs = [(a, b) for a in range(10) for b in range(10) if a != b and b not in adj[str(a)]]
        rng.shuffle(pairs)
        for a, b in pairs[:25]:
            adj[str(a)].append(b)
            if len(shortest(adj, start, target)) - 1 != depth: adj[str(a)].remove(b)
        p = dict(adj={k: sorted(v) for k, v in adj.items()}, start=start, target=target)
        family = p["adj"]
    elif task == 7:
        rows = [0, 1, 2, 3]; cols = [0, 1, 2, 3]
        for order in (rows, cols):
            if rng.randrange(2): order[:2] = order[:2][::-1]
            if rng.randrange(2): order[2:] = order[2:][::-1]
            if rng.randrange(2): order[:] = order[2:] + order[:2]
        digits = list(range(1, 5)); rng.shuffle(digits)
        grid = [digits[(r * 2 + r // 2 + c) % 4] for r in rows for c in cols]
        blanks = rng.sample(range(16), depth + 3)
        for i in blanks: grid[i] = 0
        if len(sudoku_solutions(grid)) != 1: return None
        p = dict(grid=grid, target=rng.choice(blanks)); family = grid
    else: raise ValueError(task)
    row = dict(task_id=task, difficulty=depth, payload=p, problem=render(task, p),
               id=digest([task, p])[:24], world_key=digest([task, family])[:24])
    return dict(row, **solve(task, p))


DEMOS = {}
for _task in range(2, 8):
    _rng = random.Random(2026101700 + _task)
    DEMOS[_task] = []
    for _depth in (1, 2):
        _row = None
        while _row is None: _row = candidate(_task, _depth, _rng)
        DEMOS[_task].append(_row)


def format_prompt(example):
    task = example["task_id"]
    if task < 2:
        return nr5_tasks.format_prompt(example)
    parts = [f"Task: {NAMES[task]}. {RULES[task]} End with a line: Answer: d"]
    for demo in DEMOS[task]:
        work = " -> ".join(map(str, demo["gold_trace"]))
        parts.append(f"Problem: {demo['problem']}\nWork: {work}\nAnswer: {demo['answer']}")
    parts.append(f"Problem: {render(task, example['payload'])}\nWork:")
    return "\n\n".join(parts)


def make_data():
    old = nr5_data.make_data("mixed", "pilot")
    result = {name: [] for name in ("train", "dev", "test", "depth_probe")}
    seen = {e["id"] for group in DEMOS.values() for e in group}
    worlds = {(e["task_id"], e["world_key"]): "demo" for group in DEMOS.values() for e in group}
    for split, count in (("train", 80), ("dev", 20), ("test", 40), ("depth_probe", 40)):
        for task in range(8):
            for depth in ((4, 5) if split == "depth_probe" else (1, 2, 3)):
                if task < 2:
                    pool = [e for e in old[split] if e["task_id"] == task and e["difficulty"] == depth]
                    result[split].extend(nr5_data._take_balanced(pool, count))
                    continue
                answers = list("01") if task in (2, 3) else (list("1234") if task == 7 else ([str(depth)] if task == 6 else list("0123456789")))
                needed = {a: count // len(answers) for a in answers}
                rng = random.Random(int(digest(["eight-task-v1", split, task, depth])[:16], 16))
                attempts = 0
                while sum(needed.values()):
                    attempts += 1
                    if attempts > 100000: raise RuntimeError(("Cannot allocate stratum", split, task, depth, needed))
                    row = candidate(task, depth, rng)
                    if row is None or not needed.get(row["answer"], 0) or row["id"] in seen: continue
                    key = task, row["world_key"]
                    if key in worlds and worlds[key] != split: continue
                    seen.add(row["id"]); worlds[key] = split
                    needed[row["answer"]] -= 1
                    result[split].append(row)
        result[split].sort(key=lambda e: (e["task_id"], e["difficulty"], e["id"]))
    result["provenance"] = dict(study="eight-task-v1", tasks=list(NAMES), data_seed=20261017,
        train_per_task_per_depth=80, dev_per_task_per_depth=20, test_per_task_per_depth=40,
        train_depths=[1, 2, 3], probe_depths=[4, 5],
        scope="Synthetic algorithmic families. Difficulty units differ by task. Primary comparisons are taskwise.",
        separation="Exact inputs and underlying graph, Sudoku puzzle, digit multiset, or bracket source family are split-disjoint; algorithm primitives are shared.")
    audit(result)
    return result


def audit(data):
    ids, worlds = set(), {}
    for group in DEMOS.values():
        for e in group: ids.add(e["id"]); worlds[e["task_id"], e["world_key"]] = "demo"
    for split, count in (("train", 80), ("dev", 20), ("test", 40), ("depth_probe", 40)):
        counts = Counter((e["task_id"], e["difficulty"]) for e in data[split])
        assert counts == Counter({(t, d): count for t in range(8) for d in ((4, 5) if split == "depth_probe" else (1, 2, 3))})
        for e in data[split]:
            assert e["id"] not in ids; ids.add(e["id"])
            assert format_prompt(e) == format_prompt(dict(e, answer="hidden", gold_trace=[]))
            if e["task_id"] < 2:
                expected = rlq_tasks.solve(e["task_id"], e["problem"])
                world = nr5_tasks.pointer_map_key(e) if e["task_id"] == 1 else e["id"]
            else:
                expected = solve(e["task_id"], e["payload"])
                assert e["problem"] == render(e["task_id"], e["payload"])
                world = e["world_key"]
            assert expected == {k: e[k] for k in ("answer", "gold_trace")}
            if e["task_id"] == 3:
                text = e["payload"]["text"]
                assert all(text.count(a) == text.count(b) for a, b in (("(", ")"), ("[", "]"), ("{", "}")))
                depth = 0
                for char in text:
                    depth += 1 if char in '([{' else -1
                    assert depth >= 0
                assert depth == 0
            key = e["task_id"], world
            assert key not in worlds or worlds[key] == split
            worlds[key] = split
