"""Replay smoke updates 2/3 from the committed update-1 optimizer/RNG state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import study


def probe(root, job):
    assert study.engine.read(root / 'manifest.json')['protocol']['hardware_smoke']
    assert study.engine.read(root / 'status.json')['phase'] == 'complete'
    directory = root / 'jobs' / job
    phase = directory / 'policy'
    resources = study.engine.read(directory / 'resources.json')
    for path in [directory / 'DONE.json', phase / 'DONE.json', phase / 'final.pt', phase / 'checkpoint.pt']:
        assert not path.with_suffix(path.suffix + '.first').exists()
        path.rename(path.with_suffix(path.suffix + '.first'))
    os.link(phase / 'resume_probe.pt', phase / 'checkpoint.pt')
    command = [sys.executable, '-u', str(study.HERE / 'study.py'), 'worker', '--root', str(root), '--job', job, '--smoke']
    subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=resources['visible_gpus']), check=True)
    first = study.engine.core.load_pt(phase / 'checkpoint.pt.first')
    resumed = study.engine.core.load_pt(phase / 'checkpoint.pt')
    assert first['step'] == resumed['step'] == 3 and first['identity'] == resumed['identity']
    maximum = 0.
    for name, tensor in first['parameters'].items():
        other = resumed['parameters'][name]
        maximum = max(maximum, float((tensor - other).abs().max()))
        assert study.engine.torch.allclose(tensor, other, rtol=1e-5, atol=1e-8), name
    for key, values in first['optimizer']['state'].items():
        for name, tensor in values.items():
            assert study.engine.torch.allclose(tensor, resumed['optimizer']['state'][key][name], rtol=1e-5, atol=1e-8), (key, name)
    assert first['rng']['python'] == resumed['rng']['python']
    assert first['rng']['numpy'] == resumed['rng']['numpy']
    assert study.engine.torch.equal(first['rng']['torch'], resumed['rng']['torch'])
    assert all(study.engine.torch.equal(a, b) for a, b in zip(first['rng']['cuda'], resumed['rng']['cuda']))
    result = dict(job=job, restored_from_update=1, final_update=3, max_parameter_difference=maximum,
                  optimizer_and_rng_verified=True, rtol=1e-5, atol=1e-8)
    study.engine.core.atomic_json(directory / 'gpu_resume_verified.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    probe(args.root.resolve(), args.job)
