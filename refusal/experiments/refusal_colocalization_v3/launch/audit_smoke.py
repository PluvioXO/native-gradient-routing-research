"""Verify the completed hardware smoke before launching real thresholds."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run
import campaign

root=Path(sys.argv[1]).resolve()
manifest=run.read(root/'manifest.json'); cfg=manifest['protocol']; stamp=run.core.digest(manifest)
assert cfg==run.config(True) and manifest['sources']==run.source_hashes()
status=run.read(root/'status.json')
assert status['training_complete'] and not status['running'] and not status['failures']
jobs=run.read(root/'jobs.json')
assert len(jobs)==27 and {j['seed'] for j in jobs.values()}=={31,47}
data=run.read(root/'data.json'); assert run.core.digest(data)==manifest['data_sha256']
assert run.read(root/'recipes.json')==campaign.select_recipes(root,jobs,cfg)
assert run.read(root/'recipes.json')['validation_passed']
baseline=run.read(root/'base.json')
clones=0; candidates={c:0 for c in campaign.CONDITIONS}; frozen=set()
for key,spec in jobs.items():
    directory=root/'jobs'/key; done=run.read(directory/'DONE.json')
    assert done['job']==spec and done['run']==stamp and done['qualified']
    if spec['kind']=='base':continue
    layout=run.read(directory/'layout.json')
    if layout['frozen_hash']:frozen.add(layout['frozen_hash'])
    final=directory/done['identity']['stage']/'final.pt'
    assert final.stat().st_ino==(directory/'final.pt').stat().st_ino
    if spec['kind']=='policy':
        assert run.read(directory/'policy/history.json')[-1]['step']==2
        best=run.read(directory/'policy/best.json')
        assert best['selection']==run.policy_selection(best['report'],baseline,cfg,best['step'])
        if spec['calibration']:candidates[spec['condition']]+=1
        parent=run.read(root/'jobs'/spec['teacher']/'acquisition.json')['dev_mixed']['rows']
        current=run.read(directory/'policy/dev_0000.json')['task']['rows']
        assert [r['response'] for r in parent]==[r['response'] for r in current]
        clones+=1
assert set(candidates.values())=={3} and len(frozen)==1
endpoints=sorted(root.glob('jobs/*/test_*.json')); assert len(endpoints)==27
assert min(p.stat().st_mtime_ns for p in endpoints)>=max(p.stat().st_mtime_ns for p in root.glob('jobs/*/DONE.json'))
observations=sorted(root.glob('observations/*/*/activations.pt')); assert len(observations)==27
for path in observations:
    artifact=run.core.load_pt(path)
    assert artifact['parameter_hash']==artifact['parameter_hash_after'] and artifact['observational_only']
    alignment=run.read(path.parent/'alignment.json')
    assert alignment['observational_only'] and not alignment['direction_penalty_used'] and not alignment['causal_ablation_performed']
    assert len(alignment['layers'])==24 and len(artifact['new_behavior'])==4
report=dict(time=time.time(),checks_passed=True,hardware_smoke_only=True,training_jobs=27,
    calibration_candidates_per_condition=candidates,initial_parent_response_clones_verified=clones,
    final_endpoint_reports=27,observational_hash_pairs_verified=27,confirmation_seeds_exercised=[31,47],
    source_hashes=run.source_hashes(),research_thresholds_not_tested=True)
run.core.atomic_json(Path(__file__).resolve().parent/'hardware_validation.json',report)
print(json.dumps(report,indent=2))
