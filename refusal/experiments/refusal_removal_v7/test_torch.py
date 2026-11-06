"""CPU checks using the already installed server Torch; no model downloads."""
import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
import attack


class TorchTests(unittest.TestCase):
    def test_materialization_and_unrestricted_gradients(self):
        e,_=attack.engine()
        from removal import merge_native
        torch.manual_seed(77);model=nn.Linear(4,3,bias=False);model.requires_grad_(False)
        sliced=e.core.NativeSlice(model.weight,0,[[0],[1]],[True,True])
        with torch.no_grad():sliced.deltas[0].fill_(.3)
        parametrize.register_parametrization(model,'weight',sliced)
        x=torch.randn(2,4);before=model(x).detach().clone()
        merge_native(model)
        self.assertTrue(torch.equal(before,model(x)))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(any('parametrizations' in n for n,_ in model.named_parameters()))
        F.cross_entropy(model(x),torch.tensor([0,1])).backward()
        self.assertGreater(float(model.weight.grad[2].norm()),0)

    def test_saved_optimizer_resume(self):
        e,_=attack.engine();torch.manual_seed(17)
        a=nn.Linear(4,3);state=attack.snapshot(a)
        b=nn.Linear(4,3);attack.load_parameters(b,state,torch)
        x=torch.randn(2,4);y=torch.tensor([0,1])
        def step(model,opt):
            opt.zero_grad(set_to_none=True);F.cross_entropy(model(x),y).backward();opt.step()
        oa=torch.optim.AdamW(a.parameters(),lr=1e-3,weight_decay=0,foreach=False)
        ob=torch.optim.AdamW(b.parameters(),lr=1e-3,weight_decay=0,foreach=False)
        step(a,oa);step(a,oa);step(b,ob)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'jobs/toy/resume.pt'
            attack.atomic_state(p,dict(parameters=attack.snapshot(b),optimizer=ob.state_dict()),1024)
            saved=e.core.load_pt(p);c=nn.Linear(4,3);attack.load_parameters(c,saved['parameters'],torch)
            oc=torch.optim.AdamW(c.parameters(),lr=1e-3,weight_decay=0,foreach=False);oc.load_state_dict(saved['optimizer'])
            step(c,oc)
            self.assertEqual(e.core.tensor_hash(a.named_parameters()),e.core.tensor_hash(c.named_parameters()))
            self.assertEqual(attack.optimizer_hash(e,oa),attack.optimizer_hash(e,oc))


if __name__=='__main__':unittest.main()
