"""Materialise the audited V4 engine and apply narrowly scoped V5 changes."""
import hashlib
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent


def prepare():
    manifest = json.loads((HERE / 'launch/v4_manifest.json').read_text())
    destination = HERE / 'reference_v4'
    destination.mkdir(exist_ok=True)
    with tarfile.open(HERE / 'launch/v4_source.tar.gz') as archive:
        for name, expected in manifest['sources'].items():
            member = archive.getmember(name)
            assert member.isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts
            payload = archive.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest() == expected, name
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    target = destination / 'run.py'
    source = target.read_text()
    patches = [
        ('import observe\n', 'import observe\nimport sampling\n'),
        ("endpoints=['final'] if spec['kind']=='base' else ['best','final']",
         "endpoints=['final'] if spec['kind']=='base' else ['best'] if spec['kind']=='acquire' and cfg.get('imported_acquisitions') else ['best','final']"),
        ('optimizer=opt.state_dict(),step=step,history=history,extra=extra))',
         'optimizer=opt.state_dict(),step=step,history=history,extra=extra,rng=rng_state()))'),
        ("opt.load_state_dict(saved['optimizer']); start=saved['step']; history=saved['history']",
         "opt.load_state_dict(saved['optimizer']); start=saved['step']; history=saved['history']; restore_rng(saved['rng'])"),
        ("model.train(); opt.zero_grad(set_to_none=True); tick=time.monotonic(); losses={}",
         "model.train(); opt.zero_grad(set_to_none=True); tick=time.monotonic(); losses={}; examples=[]"),
        ("a=allowed[i];options=blocked[(a['problem'],max(2,depth))]\n                b=options[(step//half_depths+micro)%len(options)]",
         "a=allowed[i]\n                b=sampling.blocked_example(blocked,a,cfg['train_depths'],step,micro,job.get('sampler','original'))\n                examples.append(dict(allowed=a['id'],blocked=b['id'],depth=depth))"),
        ("options=preserve_indices[depth];pi=options[(step//half_depths)%len(options)]",
         "options=preserve_indices[depth];pi=options[(step//half_depths)%len(options)]\n                examples[-1]['teacher']=preserve_rows[pi]['id']"),
        ("train_seconds=time.monotonic()-tick,**losses))",
         "train_seconds=time.monotonic()-tick,examples=examples,conservation=data['rehearsal'][i]['id'],**losses))"),
        ("save_checkpoint(checkpoint,route,opt,step+1,history,ident,{})",
         "save_checkpoint(checkpoint,route,opt,step+1,history,ident,{})\n            if cfg.get('hardware_smoke') and step==0:link_checkpoint(checkpoint,directory/'resume_probe.pt')"),
    ]
    for old, new in patches:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    rng_helpers = '''def rng_state():
    n=core.np.random.get_state()
    return dict(python=random.getstate(),torch=torch.get_rng_state(),
        numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']);torch.set_rng_state(state['torch'])
    n=state['numpy'];core.np.random.set_state((n[0],core.np.array(n[1],dtype='uint32'),n[2],n[3],n[4]))
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])


'''
    assert source.count('def save_checkpoint(') == 1
    source = source.replace('def save_checkpoint(', rng_helpers + 'def save_checkpoint(')
    target.write_text(source)
    changed = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
               for name in manifest['sources']}
    assert {name for name, value in changed.items() if value != manifest['sources'][name]} == {'run.py'}
    (HERE / 'launch/engine_lineage.json').write_text(json.dumps(dict(
        original_sources=manifest['sources'], runtime_sources=changed,
        changes=['positive-example sampler hook', 'actual training example IDs',
                 'Python/NumPy/Torch/CUDA RNG resume', 'evaluate the imported capability parent only']), indent=2) + '\n')


if __name__ == '__main__':
    prepare()
    print('Prepared pinned V4 reference; only declared run.py changes applied')
