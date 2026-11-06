"""Freeze and independently verify the production data and tokenizer limits."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run

root=Path(sys.argv[1]).resolve();cfg=run.config();run.free_space(root)
manifest=run.initialize(root,cfg);dataset=run.read(root/'data.json')
assert run.core.digest(dataset)==manifest['data_sha256']
excluded=set(run.read(Path(run.__file__).parent/'excluded_worlds.json'));banks={};prompts=set();answers=set()
for panel,rows in dataset.items():
    assert len(rows)==len({r['id'] for r in rows})
    for r in rows:
        prompts.add(r['prompt'])
        if 'answer' in r:answers.add(r['answer'])
        if r['task']!='pointer':continue
        assert r['world'] not in excluded and run.core.digest(r['edges'])==r['world']
        node=r['start'];actual=[]
        for c in r['commands']:node=r['edges'][node][c];actual.append(node)
        assert actual==r['path'] and r['answer']==' '.join(actual)
        assert (r['marker'] in actual[:-1])==(r['label']=='blocked')
        assert len(set([r['start'],*actual]))==len(actual)+1
        banks.setdefault(r['id'].split('_')[0],set()).add(r['world'])
assert sum(map(len,banks.values()))==len(set().union(*banks.values()))
assert {r['wording'] for r in dataset['wording_test']}=={2}
assert {r['depth'] for r in dataset['ood_test']}=={5,6}
assert not {r['prompt'] for r in dataset['benchmark_dev']}&{r['prompt'] for r in dataset['benchmark_test']}
tokenizer=run.AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['revision'])
longest=max((len(run.core.prompt_tokens(tokenizer,p)),p) for p in prompts)
assert longest[0]<=512,('Prefix too long',longest)
max_answer=max(len(tokenizer.encode(a,add_special_tokens=False))+1 for a in answers)
assert max_answer<=cfg['eval_max_new_tokens']
assert all(len(tokenizer.encode(c,add_special_tokens=False))==1 for c in run.datasets.NODES)
receipt=dict(checks_passed=True,time=time.time(),source_hashes=manifest['sources'],data_sha256=manifest['data_sha256'],
    world_banks={k:len(v) for k,v in banks.items()},excluded_worlds=len(excluded),
    rows={k:len(v) for k,v in dataset.items()},unique_prompts=len(prompts),max_prefix_tokens=longest[0],max_gold_completion_tokens=max_answer,
    input_truncation_used=False,train_depths=cfg['train_depths'],ood_depths=cfg['ood_depths'],
    widths=[1,2,4,8],native_block_zero_based=cfg['layer'],observational_only=True)
run.core.atomic_json(Path(__file__).parent/'production_preflight.json',receipt)
print(receipt)
