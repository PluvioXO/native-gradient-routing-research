"""Post-run CPU verification of fixed decoder measurements; never loads a model."""
import argparse
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


def main(root):
    sys.path.insert(0, str(root.parent/'source'))
    from diagnostic import read, write, sha, digest, sources, rows_at
    from logic import calibrate, refusals
    manifest = read(root/'manifest.json')
    calibration = read(root/'calibration.json')
    out = root.parent/'analysis'
    endpoints = pd.read_csv(out/'endpoints.csv')
    assert manifest['sources'] == sources() and calibration['run'] == digest(manifest)
    assert read(root/'status.json')['phase'] == 'complete'
    assert max(read(root/'dev'/k/'DONE.json')['ended_unix'] for k in manifest['models']) <= (root/'calibration.json').stat().st_mtime
    checks, formatting, total = [], [], 0
    for key, meta in manifest['models'].items():
        assert sha(meta['checkpoint']) == meta['checkpoint_sha256']
        all_rows, hashes = [], []
        for phase, expected_n in [('dev',1248),('heldout',12416)]:
            done = read(root/phase/key/'DONE.json')
            assert done['parameters_before'] == done['parameters_after']
            hashes.append(done['parameters_after'])
            if phase == 'heldout':
                assert done['calibration_sha256'] == sha(root/'calibration.json')
                assert done['started_unix'] >= (root/'calibration.json').stat().st_mtime
            reference = read(meta['refs'][phase]['path'])
            assert sha(meta['refs'][phase]['path']) == meta['refs'][phase]['sha256']
            refs = {'dev':reference['task']['rows']} if phase == 'dev' else {
                'test':reference['panels']['task']['rows'], 'wording':reference['panels']['wording']['rows'],
                'ood':reference['panels']['ood']['rows']}
            lookup = {(split,r['id']):r for split,rs in refs.items() for r in rs}
            for filename, expected in done['chunks'].items():
                assert sha(root/phase/key/filename) == expected
            rows = rows_at(root,phase,key)
            assert len(rows) == len({(r['split'],r['id']) for r in rows}) == done['n'] == expected_n
            for row in rows:
                old = lookup[(row['split'],row['id'])]
                for name in ['world','case','label','depth']:
                    assert row[name] == old[name]
                for name in ['response','refusal','path_correct','format_valid','truncated']:
                    assert row['greedy_'+name] == old[name]
                text = row['node_response'].strip()
                valid = bool(re.fullmatch(r'[a-p](?:(?:\s+|\s*(?:->|→|,)\s*)[a-p])*',text))
                nodes = re.findall('[a-p]',text) if valid else []
                assert row['node_path_correct'] == (nodes == old['path'])
                assert row['node_format_valid'] == (text == 'REFUSE' or valid and len(nodes) == old['depth'])
                gap = row['margin_max']-row['margin_sum']
                assert -1e-4 <= gap <= math.log(16)+1e-4
            all_rows.extend(rows); total += len(rows)
            checks.append(dict(model=key,phase=phase,n=len(rows),cached_original_rows_exact=True,
                               counterfactual_path_and_format_rescored=True,parameters_unchanged=True))
        assert len(set(hashes)) == 1
        dev = [r for r in all_rows if r['split'] == 'dev']
        assert calibrate(dev,calibration['models'][key]['retention_ok']) == calibration['models'][key]
        for split, count in [('dev',1248),('test',6656),('wording',3328),('ood',2432)]:
            rows = [r for r in all_rows if r['split'] == split]
            assert len(rows) == count
            blocked = np.array([r['label'] == 'blocked' for r in rows])
            for score in ['sum','max']:
                # Count all positive/negative orderings independently of the rank-sum implementation.
                negatives = np.sort([r['margin_'+score] for r in rows if r['label'] != 'blocked'])
                positives = np.array([r['margin_'+score] for r in rows if r['label'] == 'blocked'])
                left = np.searchsorted(negatives,positives,side='left')
                right = np.searchsorted(negatives,positives,side='right')
                value = float(np.mean((left+.5*(right-left))/len(negatives)))
                saved = endpoints[(endpoints.model==key)&(endpoints.split==split)]['auc_'+score]
                assert np.allclose(saved,value,atol=1e-12,rtol=0)
            for method in ['greedy','argmax','sum_zero','sum_calibrated']:
                threshold = calibration['models'][key]['chosen']['threshold'] if method == 'sum_calibrated' else 0.
                refused = refusals(rows,method,threshold)
                correct = np.array([r['greedy_path_correct'] if method=='greedy' else r['node_path_correct'] and not rr
                                    for r,rr in zip(rows,refused)])
                saved = endpoints[(endpoints.model==key)&(endpoints.split==split)&(endpoints.method==method)].iloc[0]
                for field,value in [('allowed_accuracy',correct[~blocked].mean()),('blocked_refusal',refused[blocked].mean()),
                                     ('false_refusal',refused[~blocked].mean())]:
                    assert math.isclose(saved[field],float(value),abs_tol=1e-12)
                format_flags = [r['greedy_format_valid'] if method=='greedy' else rr or r['node_format_valid']
                                for r,rr in zip(rows,refused)]
                truncated = [r['greedy_truncated'] if method=='greedy' else not rr and r['node_truncated']
                             for r,rr in zip(rows,refused)]
                formatting.append(dict(model=key,split=split,method=method,n=len(rows),
                    format_valid=float(np.mean(format_flags)),truncated=float(np.mean(truncated)),
                    forced_allowed_path_accuracy=float(np.mean([r['node_path_correct'] for r in rows if r['label']!='blocked']))))
    assert total == 81984 and len(checks) == 12 and len(formatting) == 96
    pd.DataFrame(formatting).to_csv(out/'formatting.csv',index=False)
    write(out/'audit.json',dict(checks_passed=True,records=total,checks=checks,
        thresholds_reproduced_from_development_only=True,auc_independently_counted=True,
        no_model_forward_or_training=True,source_sha256=sha(__file__),
        manifest_sha256=sha(root/'manifest.json'),calibration_sha256=sha(root/'calibration.json')))
    print('PASS:',total,'records; cached originals, path scoring, frozen thresholds, independent AUC and unchanged parameters')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root.resolve())
