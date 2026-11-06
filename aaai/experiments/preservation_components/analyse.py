"""Audit the component extension with the existing saved-prediction auditor."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent/'adaptive_ablation'
spec = importlib.util.spec_from_file_location('saved_measurement_audit', LEGACY/'analyse.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    if not (HERE/'results').exists():raise SystemExit('Raw measurements were removed during the 5 November cleanup. See aaai/RAW_DATA.md to restore them; analysed tables and plots remain in this study.')
    audit.ROOT = HERE/'results/casr-preservation-components-20261031'
    audit.OUT = HERE/'analysis'
    audit.main()
    old_root = LEGACY/'results/casr-adaptive-ablation-20261030'
    controls = audit.read(audit.ROOT/'source/controls_manifest.json')
    assert audit.sha(old_root/'outputs/main/experiment_plan.json') == controls['plan_sha256']
    old_plan = audit.read(old_root/'outputs/main/experiment_plan.json')
    new_plan = audit.read(audit.ROOT/'outputs/main/experiment_plan.json')
    assert old_plan['inputs'] == new_plan['inputs']
    for record in controls['records']:
        p=old_root/record['path']
        assert audit.sha(p)==record['sha256']
        if record['bank_sha256']:
            assert audit.sha(p.parent/'bank_final.pt')==record['bank_sha256']
    # Reuse the old audit only if its entire exported evidence and summary are unchanged.
    old_audit=audit.read(LEGACY/'analysis/audit.json')
    assert old_audit['complete']
    assert audit.sha(old_root/'export_inventory.json')==old_audit['archive_inventory_sha256']
    for name,expected in audit.read(old_root/'export_inventory.json').items():
        assert audit.sha(old_root/name)==expected,name
    provenance=audit.read(HERE.parents[1]/'figures/adaptive_ablation/provenance.json')
    assert audit.sha(LEGACY/'analysis/per_bank.json')==provenance['inputs']['per_bank.json']
    for support in ('native','lora'):
        for seed in (31,47,71):
            left=old_root/f'outputs/main/fits/eval_{support}_small_aware_s{seed}/retention/base.json'
            right=audit.ROOT/f'outputs/main/fits/eval_{support}_small_self_only_s{seed}/retention/base.json'
            a,b=audit.read(left),audit.read(right)
            assert a['rows']==b['rows'] and a['wiki']==b['wiki']
    record=audit.read(audit.OUT/'audit.json')
    record.update(archived_control_banks=24,control_completions_verified=len(controls['records']),
                  archived_controls_unchanged=True,cross_study_baseline_identical=True,
                  auditor_sha256=audit.sha(LEGACY/'analyse.py'),extension_auditor_sha256=audit.sha(Path(__file__)))
    (audit.OUT/'audit.json').write_text(__import__('json').dumps(record,indent=2))
    print('PASS archived control lineage, evidence hashes, summary identity, and cross-study baseline')


if __name__=='__main__':main()
