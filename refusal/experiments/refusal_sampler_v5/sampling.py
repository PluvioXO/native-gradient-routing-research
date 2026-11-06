"""Only the positive refusal example changes between the two schedules."""
from collections import Counter


def blocked_groups(depths):
    return [(depth, position) for depth in depths for position in range(1, depth)]


def blocked_example(pools, allowed, depths, step, micro, mode):
    if mode == 'original':
        options = pools[(allowed['problem'], max(2, allowed['depth']))]
        return options[(step // (len(depths) // 2) + micro) % len(options)]
    if mode != 'balanced':
        raise ValueError(f'Unknown sampler: {mode}')
    groups = blocked_groups(depths)
    depth, position = groups[(2 * step + micro) % len(groups)]
    row = pools[(allowed['problem'], depth)][position - 1]
    assert row['case'] == f'h{depth}_blocked_{position}' and row['label'] == 'blocked'
    return row


def schedule(data, cfg, mode, pick):
    allowed = [r for r in data['policy_train'] if r['label'] != 'blocked']
    blocked = {}
    for row in data['policy_train']:
        if row['label'] == 'blocked':
            blocked.setdefault((row['problem'], row['depth']), []).append(row)
    indices = {d: [i for i, r in enumerate(allowed) if r['depth'] == d] for d in cfg['train_depths']}
    preserved = {d: [r for r in data['preserve_train'] if r['depth'] == d] for d in cfg['train_depths']}
    orders = {d: {} for d in cfg['train_depths']}
    half = len(cfg['train_depths']) // 2
    for step in range(cfg['policy_steps']):
        depths = [cfg['train_depths'][step % half], cfg['train_depths'][step % half + half]]
        examples = []
        for micro, depth in enumerate(depths):
            pool = indices[depth]
            a = allowed[pool[pick(len(pool), step, 0, cfg['seed'], orders[depth], per_step=1)]]
            b = blocked_example(blocked, a, cfg['train_depths'], step, micro, mode)
            t = preserved[depth][(step // half) % len(preserved[depth])]
            examples.append(dict(allowed=a['id'], blocked=b['id'], teacher=t['id'], depth=depth,
                                 blocked_case=b['case'], problem=a['problem']))
        n = len(data['rehearsal']) // 2
        c = data['rehearsal'][(step // 2) % n + (step % 2) * n]
        yield dict(step=step + 1, examples=examples, conservation=c['id'])


def audit(data, cfg, pick):
    original = list(schedule(data, cfg, 'original', pick))
    balanced = list(schedule(data, cfg, 'balanced', pick))
    for a, b in zip(original, balanced):
        assert a['conservation'] == b['conservation']
        assert [(r['allowed'], r['teacher'], r['depth']) for r in a['examples']] == [
            (r['allowed'], r['teacher'], r['depth']) for r in b['examples']]
    counts = {name: Counter(r['blocked_case'] for s in rows for r in s['examples'])
              for name, rows in [('original', original), ('balanced', balanced)]}
    assert max(counts['balanced'].values()) - min(counts['balanced'].values()) <= 1
    assert set(counts['balanced']) == {f'h{d}_blocked_{p}' for d, p in blocked_groups(cfg['train_depths'])}
    return dict(steps=cfg['policy_steps'], positive_examples=2 * cfg['policy_steps'],
                negative_examples=2 * cfg['policy_steps'], counts={k: dict(v) for k, v in counts.items()},
                identical_allowed_teacher_conservation=True, matching='updates/examples; not exact tokens or FLOPs')
