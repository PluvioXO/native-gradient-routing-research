"""CPU checks for freshness, dependency barriers, scientific parity and statistics."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import fresh_data
import confirm_study as study


def check():
    p=study.protocol();data=json.loads((ROOT/'reference/dataset.json').read_text())
    assert fresh_data.audit(data,p)['passed']
    assert fresh_data.make_data(p)==data,'Generation must be deterministic without model selection'
    bad=copy.deepcopy(data);bad['test'][0]=copy.deepcopy(data['train'][0])
    try:fresh_data.audit(bad,p)
    except AssertionError:pass
    else:raise AssertionError('Cross-split leak was accepted')
    bad=copy.deepcopy(data);bad['test'][0]['gold_trace']=[]
    try:fresh_data.audit(bad,p)
    except AssertionError:pass
    else:raise AssertionError('Incorrect solver label was accepted')
    for smoke,expected in [(False,dict(independent=96,bank=12,main=36,merge=48)),(True,dict(independent=2,main=6,merge=2))]:
        plan=study.make_plan(smoke);jobs={j['id']:j for j in plan['jobs']};assert len(jobs)==len(plan['jobs'])
        from collections import Counter
        assert Counter(j['kind'] for j in jobs.values())==expected
        training={j['id'] for j in jobs.values() if j['kind']!='merge'}
        for job in jobs.values():
            assert set(job.get('dependencies',[]))<=set(jobs)
            if job['kind']=='merge':assert set(job['dependencies'])==training,'Fresh test can run before training finishes'
        assert plan==study.make_plan(smoke)
    lineage=json.loads((ROOT/'engine_lineage.json').read_text())
    for name,expected in lineage['unchanged'].items():
        assert study.hash_file(ROOT/name)==expected,name
    text=(ROOT/'run_aware.py').read_text().replace('import independent_engine as previous','import run_study as previous')
    import hashlib
    assert hashlib.sha256(text.encode()).hexdigest()==lineage['run_aware_original_sha256']
    # Real CPU fixtures cover all eight owners, teacher restoration and optimizer rollback.
    import test_aware
    test_aware.check()
    from unittest.mock import patch
    import multi_run
    with patch.object(multi_run,'setup'):
        study.configure_runtime(True)
        smoke,_=study.previous.init_runtime()
        train_ids={e['id'] for e in data['train']}
        assert all(e['id'] in train_ids for split in ('train','dev','test','depth_probe') for e in smoke[split])
        study.configure_runtime(False)
        production,_=study.previous.init_runtime();assert production==data
    from analyse_results import paired,holm
    result=[paired([1,2,3,4,5,6]),paired([-1,-2,-3,-4,-5,-6])];holm(result)
    assert result[0]['meets_prespecified_success'] and not result[1]['meets_prespecified_success']
    assert paired([0]*6)['p_two_sided']==1
    assert paired([1]*6)['p_two_sided']==0
    print('PASS: fresh balanced world-disjoint bank; leak/label rejection; deterministic generation; 192-job production graph; global test barrier; unchanged training kernels; eight-owner CPU checks; train-only smoke; paired statistics',flush=True)


if __name__=='__main__':check()
