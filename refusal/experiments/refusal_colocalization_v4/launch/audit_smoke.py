"""Read-only verification of the completed five-budget hardware smoke."""
import json
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run,campaign

root=Path(sys.argv[1]).resolve();manifest=run.read(root/'manifest.json');cfg=manifest['protocol'];stamp=run.core.digest(manifest)
assert cfg==run.config(True) and manifest['sources']==run.source_hashes()
status=run.read(root/'status.json');assert status['training_complete'] and not status['running'] and not status['failures']
jobs=run.read(root/'jobs.json');assert len(jobs)==21 and {j['seed'] for j in jobs.values()}=={31}
assert run.read(root/'recipes.json')==campaign.select_recipes(root,jobs,cfg)
assert run.read(root/'recipes.json')['validation_passed']
baseline=run.read(root/'base.json');clones=0;counts={c:0 for c in campaign.CONDITIONS};frozen=set();masks={}
for key,spec in jobs.items():
    directory=root/'jobs'/key;done=run.read(directory/'DONE.json')
    assert done['job']==spec and done['run']==stamp and done['qualified']
    if spec['kind']=='base':continue
    layout=run.read(directory/'layout.json')
    if layout['frozen_hash']:frozen.add(layout['frozen_hash'])
    assert (directory/'final.pt').stat().st_ino==(directory/done['identity']['stage']/'final.pt').stat().st_ino
    if spec['geometry']=='native':
        assert layout['unique_trainable']==spec['width']*2688
        assert len(layout['layout'])==3 and all(x['name'].startswith('model.layers.11.mlp.') for x in layout['layout'])
        mask=layout['layout'][0]['indices'][0]
        assert len(mask)==spec['width'] and all(x['indices'][0]==mask for x in layout['layout'])
        masks[spec['width']]=set(mask)
    if spec['kind']=='acquire':
        for stage in ['basic','hard']:assert run.read(directory/stage/'DONE.json')['step']==2
        assert (directory/'capability.pt').stat().st_ino==(directory/'hard/best.pt').stat().st_ino
        best=run.read(directory/'hard/best.json')
        assert best['selection']==run.capability_selection(best['report'],baseline,cfg,best['step'])
    else:
        assert run.read(directory/'policy/DONE.json')['step']==2
        best=run.read(directory/'policy/best.json')
        assert best['selection']==run.policy_selection(best['report'],baseline,cfg,best['step'])
        counts[spec['condition']]+=1
        parent=run.read(root/'jobs'/spec['teacher']/'acquisition.json')['dev_mixed']['rows']
        initial=run.read(directory/'policy/dev_0000.json')['task']['rows']
        assert [r['response'] for r in parent]==[r['response'] for r in initial];clones+=1
        selected=run.read(root/'jobs'/spec['teacher']/'hard/best.json')
        assert best['identity']['parent']==run.checkpoint_owner(selected)
assert set(counts.values())=={3} and len(frozen)==1
assert masks[1]<masks[2]<masks[4]<masks[8]
endpoints=sorted(root.glob('jobs/*/test_*.json'));assert len(endpoints)==21
assert min(p.stat().st_mtime_ns for p in endpoints)>=max(p.stat().st_mtime_ns for p in root.glob('jobs/*/DONE.json'))
for p in endpoints:
    report=run.read(p);assert report['run']==stamp
    assert set(report['panels'])=={'task','wording','ood','general','benchmark'}
observations=sorted(root.glob('observations/*/*/activations.pt'));assert len(observations)==21
for p in observations:
    a=run.core.load_pt(p);assert a['parameter_hash']==a['parameter_hash_after'] and a['observational_only']
    j=run.read(p.parent/'alignment.json')
    assert j['observational_only'] and not j['direction_penalty_used'] and not j['causal_ablation_performed']
    assert len(j['layers'])==24 and len(a['new_behavior'])==4
receipt=dict(checks_passed=True,time=time.time(),hardware_smoke_only=True,completed_jobs=21,endpoint_reports=21,
    initial_parent_clones_verified=clones,candidates_per_condition=counts,observational_hash_pairs_verified=21,
    source_hashes=run.source_hashes(),research_thresholds_not_tested=True)
run.core.atomic_json(Path(__file__).parent/'hardware_validation.json',receipt)
print(json.dumps(receipt,indent=2))
