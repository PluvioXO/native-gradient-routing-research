"""Validate imported graph truth, exact schedules and token accounting without a model."""
import argparse
from functools import lru_cache
from pathlib import Path
import sys
import study
import sampling


def check(root):
    manifest = study.engine.read(root / 'manifest.json')
    assert manifest['sources'] == study.source_hashes()
    data = study.engine.read(root / 'data.json')
    cfg = manifest['protocol']
    assert study.engine.core.digest(data) == manifest['data_sha256']
    rows_checked = 0
    for rows in data.values():
        assert len(rows) == len({r['id'] for r in rows})
        for row in rows:
            if row['task'] != 'pointer':
                continue
            node, path = row['start'], []
            for command in row['commands']:
                node = row['edges'][node][command]
                path.append(node)
            assert path == row['path'] and row['answer'] == ' '.join(path)
            assert (row['label'] == 'blocked') == (row['marker'] in path[:-1])
            assert len(set([row['start'], *path])) == len(path) + 1
            rows_checked += 1
    banks = {name: {r['world'] for r in data['policy_' + name]} for name in ['train', 'dev', 'test']}
    assert not banks['train'] & banks['dev'] and not banks['train'] & banks['test'] and not banks['dev'] & banks['test']
    tokenizer = study.engine.AutoTokenizer.from_pretrained(cfg['model'], revision=cfg['revision'])
    @lru_cache(None)
    def tokens(prompt, answer=None):
        return len(study.engine.legacy.ids_for(tokenizer, dict(prompt=prompt), answer))
    def size(row, answer=None):
        return tokens(row['prompt'], answer)
    lookup = {r['id']: r for r in data['policy_train']}
    teachers = {r['id']: r for r in data['preserve_train']}
    conservation = {r['id']: r for r in data['rehearsal']}
    counts = {}
    for mode in cfg['samplers']:
        total, target, positive = 0, 0, 0
        for step in sampling.schedule(data, cfg, mode, study.engine.pick):
            for sample in step['examples']:
                a, b, t = lookup[sample['allowed']], lookup[sample['blocked']], teachers[sample['teacher']]
                total += size(a, a['answer']) + size(a) + size(b) + size(b, 'REFUSE') + size(t, t['answer'])
                target += size(a, a['answer']) - size(a) + size(b, 'REFUSE') - size(b) - 1
                positive += size(b)
            c = conservation[step['conservation']]
            total += size(c, c['answer'])
        counts[mode] = dict(student_forward_input_tokens=total, positive_prompt_tokens=positive,
                            supervised_target_tokens=target, student_forward_calls=11 * cfg['policy_steps'])
    longest = max(size(r) for rows in data.values() for r in rows)
    assert longest <= 512
    value = dict(checks_passed=True, pointer_rows_checked=rows_checked,
                 train_dev_test_worlds={k: len(v) for k, v in banks.items()}, max_prefix_tokens=longest,
                 tokens=counts, token_scope='Student forward inputs; excludes cached-teacher construction and gradient-checkpoint recomputation',
                 flops_measured=False, python=sys.version, torch=study.engine.torch.__version__,
                 transformers=study.engine.core.version('transformers'), sources=manifest['sources'])
    study.engine.core.atomic_json(root / 'preflight.json', value)
    print(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    check(parser.parse_args().root.resolve())
