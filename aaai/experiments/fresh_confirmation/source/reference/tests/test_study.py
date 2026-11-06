"""Numerical ownership, composition, scaling, and interrupted-resume checks."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from contextlib import redirect_stdout
import io

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch
import nr5_rl as rl
import nr6_train as train
from nr5_rl_checks import fixture
from nr6_checks import equal
from budget_model import BudgetNative, BudgetLoRA
from dai_rl import update
import multi_tasks as tasks
import compound_tasks
import run_study as study


def test_plan_data():
    plan = study.make_plan()
    assert {kind: sum(j["kind"] == kind for j in plan["jobs"]) for kind in ("baseline", "pilot", "main", "merge")} == dict(baseline=1, pilot=24, main=96, merge=12)
    assert len({j["id"] for j in plan["jobs"]}) == len(plan["jobs"])
    for task in range(8):
        masks = study.masks_for_task(task)
        assert len(masks) == len(set(masks)) == 15
        assert sum(len(m) == 2 for m in masks) == 7
        assert sum(len(m) == 4 for m in masks) == 4
        assert all(task in m for m in masks if len(m) in (1, 2, 4, 8))
    data = json.loads((ROOT / "dataset.json").read_text()); tasks.audit(data)
    compounds = compound_tasks.build(data)
    assert compounds == compound_tasks.build(data)
    assert len(compounds["test"]) == 240 and len(compounds["dev"]) == 120
    assert not {e['id'] for e in compounds['test']} & {e['id'] for e in compounds['dev']}
    for e in compounds['test'] + compounds['dev']:
        assert compound_tasks.prompt(e, compounds['demos']) == compound_tasks.prompt(dict(e, answer='hidden', gold_trace=[]), compounds['demos'])
        expected = (tasks.solve(4, e['second']['payload']) if e['family'] == 'stack_sorting'
                    else tasks.rlq_tasks.solve(0, e['second']['problem']))
        assert expected['answer'] == e['answer']
        first = e['first']
        completion = ' ' + ' -> '.join(map(str, first['gold_trace'])) + '\nAnswer: ' + first['answer'] + '\n'
        query = compound_tasks.intermediate_query(e, completion)
        assert query is not None and query['problem'] == e['second']['problem']
        assert compound_tasks.intermediate_query(e, 'invalid') is None
        assert compound_tasks.intermediate_query(e, completion.replace('->', '→')) == query
    print('PASS plan counts, balanced masks, target-blind compound prompts and independent primitive solvers', flush=True)
    return data


def test_numerics(data):
    torch.set_num_threads(1)
    base, tok, _ = fixture()
    rl.format_prompt = train.format_prompt = tasks.format_prompt
    rl.MASKS.clear(); rl.MASKS.update({name: None for name in ('none', 'owner', 'foreign', 'all')})
    train.RECIPES['independent_routed'] = ('routed', 'all_ab')
    cfg = train.Config(max_steps=3, prompts_per_update=1, group_size=2, generation_batch=2,
        max_new_tokens=4, max_prompt_tokens=2048, eval_every=3, save_every=1, eval_batch=2,
        lr=1e-3, min_lr=1e-5, max_mean_kl=100, max_sequence_kl=100,
        kl_positions=2, head_chunk=8, kl_head_chunk=7, clip_grad=1e6,
        preservation_mode='all_ab', audit_steps=[])
    def scripted(examples):
        return [dict(example=e, prompt_ids=tok(tasks.format_prompt(e), add_special_tokens=False).input_ids,
            generated_ids=tok(' 1 -> 2\nAnswer: ' + (e['answer'] if i % 2 == 0 else str((int(e['answer'])+1) % 10)) + '\n', add_special_tokens=False).input_ids,
            completion=' 1 -> 2\nAnswer: ' + e['answer'] + '\n', answer=e['answer'], valid=True, correct=i % 2 == 0, truncated=False)
            for i, e in enumerate(examples)]
    for cls in (BudgetNative, BudgetLoRA):
        kwargs = dict(seed=31, n_layers=2, allocation_neurons=2)
        kwargs.update(neurons=2) if cls is BudgetNative else kwargs.update(rank=2)
        module = cls(copy.deepcopy(base), **kwargs)
        initial = module.state_dict(); payloads = []
        for task in range(8):
            module.load_state_dict(initial)
            example = next(e for e in data['train'] if e['task_id'] == task)
            items = scripted([example] * 2)
            opts = [torch.optim.AdamW(module.parameters(t), lr=cfg.lr, weight_decay=0) for t in list(range(8))*2]
            result = update(module, tok, items, [1., -1.], cfg, opts, task, 'routed', items)
            assert result['accepted'] and result['active_optimizer_indices'] == [task]
            assert all(torch.equal(initial[n], p) for n, p, t in module._named_deltas if t != task)
            assert all(not o.state for i, o in enumerate(opts) if i != task)
            assert all(any(not torch.equal(initial[n], p) for n,p,t in module._named_deltas if t == task and region in n)
                       for region in ('.mlp.', '.self_attn.'))
            payloads.append(dict(spec=dict(task=task), plan_sha256='test', manifest=module.manifest(),
                deltas={n:p.detach().clone() for n,p,t in module._named_deltas if t == task}))
        original = study.combine_owners(module, payloads, 'test')
        expected_weights = []
        for linear, adapter in module._attachments:
            weight = linear.parametrizations.weight.original
            with torch.no_grad(), module.gate((True,)*8): actual = linear.weight.clone()
            if cls is BudgetLoRA:
                b = torch.cat(list(adapter.left), dim=1)
                a = torch.cat(list(adapter.right), dim=0)
                expected = weight + b @ a
                assert torch.allclose(actual, expected, atol=2e-6, rtol=1e-5)
            else:
                expected = weight.clone()
                for task in range(8):
                    index = getattr(adapter, f'index_{task}')
                    expected.index_add_(adapter.axis, index, adapter.deltas[task])
                assert torch.equal(actual, expected)
            expected_weights.append(actual)
        study.load_scaled(module, original, [.5]*8)
        for (linear, _), full in zip(module._attachments, expected_weights):
            weight = linear.parametrizations.weight.original
            with torch.no_grad(), module.gate((True,)*8): half = linear.weight.clone()
            assert torch.allclose(half-weight, (full-weight)*.5, atol=2e-6, rtol=1e-5)
        assert module.assert_frozen_base(); module.close()
        # Exercise the real trainer's save/resume path on a task beyond A/B.
        example = next(e for e in data['train'] if e['task_id'] == 7)
        subset = dict(train=[example], dev=[example])
        modules = [cls(copy.deepcopy(base), **kwargs) for _ in range(3)]
        for module in modules: module.independent_task = 7
        def generate(module, tokenizer, examples, config, mask='AB', sample=False):
            
            if sample: torch.rand(3)
            return scripted(examples)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.object(train, 'extended_update', side_effect=update), patch.object(rl, '_generate', side_effect=generate):
                train.train_recipe(modules[0], tok, subset, cfg, folder/'full', 'independent_routed', 31)
                def interrupt():
                    if rl._load(folder/'partial/latest.pt')['step'] == 1: raise ConnectionError('test interruption')
                try: train.train_recipe(modules[1], tok, subset, cfg, folder/'partial', 'independent_routed', 31, interrupt)
                except ConnectionError: pass
                else: raise AssertionError('Expected interruption')
                train.train_recipe(modules[2], tok, subset, cfg, folder/'partial', 'independent_routed', 31)
            full, resumed = [rl._load(folder/name/'latest.pt') for name in ('full','partial')]
            for key in ('deltas','optimizers','rng'): assert equal(full[key], resumed[key]), key
        for module in modules: module.close()
    print('PASS all-eight-task gradient ownership, unchanged nonowners, exact native composition, LoRA concatenation, coefficient scaling, and interrupted resume', flush=True)


def test_atomic_processes():
    import multiprocessing
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'selection.json'
        value = {'selection': [1e-4]*4, 'payload': 'x'*100000}
        context = multiprocessing.get_context('fork')
        processes = [context.Process(target=study.atomic, args=(path, value)) for _ in range(12)]
        for process in processes: process.start()
        for process in processes:
            process.join(15); assert process.exitcode == 0
        assert json.loads(path.read_text()) == value and not list(path.parent.glob('*.tmp'))
    print('PASS concurrent identical selection writes use separate temporary files', flush=True)


def test_queue():
    plan = study.make_plan()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp); starts = []
        class Process:
            def __init__(self, command, **kwargs):
                job_id = command[command.index('--job') + 1]
                job = next(j for j in plan['jobs'] if j['id'] == job_id)
                completed = {j['id'] for j in plan['jobs'] if (output/'fits'/j['id']/'completed.json').exists()}
                if job['kind'] != 'baseline': assert 'baseline' in completed
                if job['kind'] in ('main', 'merge'):
                    assert all(j['id'] in completed for j in plan['jobs'] if j['kind'] == 'pilot')
                if job['kind'] == 'merge': assert set(job['dependencies']) <= completed
                starts.append(job_id); self.pid = len(starts)
                study.atomic(output/'fits'/job_id/'completed.json', {'complete': True})
            def poll(self): return 0
        with patch.object(study.subprocess, 'Popen', Process), patch.object(study.time, 'sleep'), patch.object(study, 'checked_space'), patch.object(study, 'validate_completed', side_effect=lambda output, job, plan: (output/'fits'/job['id']/'completed.json').exists()), redirect_stdout(io.StringIO()):
            study.supervise(plan, output, ['0','1','2','3','4','5'])
            assert len(starts) == 133 and json.loads((output/'ALL_COMPLETE.json').read_text())['completed'] == 133
            starts.clear(); study.supervise(plan, output, ['0'])
            assert not starts and json.loads((output/'status.json').read_text())['completed'] == 133
    print('PASS baseline/pilot/owner prerequisites, full queue completion and completed-run restart', flush=True)


if __name__ == '__main__':
    data = test_plan_data(); test_numerics(data); test_atomic_processes(); test_queue()
    print('ALL OFFLINE CHECKS PASSED', flush=True)
