"""Final native-routing, baseline-evaluation and exact-resume checks on one GPU."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import run

root=Path(sys.argv[1]).resolve()
source=Path(__file__).resolve().parent
cfg=copy.deepcopy(run.PROTOCOL)
cfg.update(widths=[4],capability_steps=2,joint_steps=2,checkpoint_every=1,dev_every=2,
           train_pairs=4,dev_pairs=2,test_pairs=2,wording_pairs=2,
           rehearsal_per_task=2,general_dev_per_task=2,general_test_per_task=2)
run.initialize(root,cfg)
cache=source/'cache'
env=dict(os.environ,HF_HOME=str(cache/'huggingface'),XDG_CACHE_HOME=str(cache),
         CUDA_CACHE_PATH=str(cache/'cuda'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
         TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2')


def check(*flags,expect_error=None):
    command=[sys.executable,str(source/'run.py'),'--worker','--smoke','--root',str(root),*flags]
    result=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    with (root/'validation.log').open('a') as stream:
        stream.write(result.stdout)
    if expect_error:
        assert result.returncode!=0 and expect_error in result.stdout, result.stdout
    else:
        assert result.returncode==0,result.stdout
    print('PASS',*flags,flush=True)


check('--stage','baseline','--width','0')
check('--stage','capability','--width','4')
check('--stage','joint','--width','4','--placement','different')
directory=root/'joint/native__k4__different'
before=run.core.load_pt(directory/'final.pt')
before_hash=run.core.tensor_hash(before['parameters'].items())
check('--stage','joint','--width','4','--placement','different')
after=run.core.load_pt(directory/'final.pt')
assert before_hash==run.core.tensor_hash(after['parameters'].items())
saved=run.core.load_pt(directory/'checkpoint.pt')
assert saved['step']==2 and len(saved['history'])==2
assert all(int(s['step'])==2 for s in saved['optimizer']['state'].values())
check('--stage','baseline','--width','0','--evaluate')
assert (root/'baseline_test.json').exists()
data=json.loads((root/'data.json').read_text())
changed=copy.deepcopy(data)
changed['train'][0]['answer']='z'
try:
    run.core.atomic_json(root/'data.json',changed)
    check('--stage','capability','--width','4',expect_error='Dataset checksum differs')
finally:
    run.core.atomic_json(root/'data.json',data)
run.core.atomic_json(root/'VALIDATION.json',dict(native_gradient_ownership=True,
    zero_delta_witness=True,resume_parameter_hash=before_hash,optimizer_step=2,
    baseline_test=True,dataset_corruption_rejected=True,
    fingerprint=run.fingerprint(root)))
print('ALL POSTFLIGHT CHECKS PASSED',flush=True)
