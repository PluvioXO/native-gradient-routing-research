import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from run import PROTOCOL, Routing, make_data, strict_score


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(hidden_size=8,intermediate_size=512)
        self.model=nn.Module()
        self.model.layers=nn.ModuleList()
        for _ in range(24):
            layer=nn.Module()
            layer.mlp=nn.Module()
            layer.mlp.up_proj=nn.Linear(8,512,bias=False)
            layer.mlp.gate_proj=nn.Linear(8,512,bias=False)
            layer.mlp.down_proj=nn.Linear(512,8,bias=False)
            self.model.layers.append(layer)

    def forward(self,x):
        for layer in self.model.layers:
            m=layer.mlp
            x=x+.01*m.down_proj(m.up_proj(x)*torch.sigmoid(m.gate_proj(x)))
        return x


class Checks(unittest.TestCase):
    def test_data(self):
        data=make_data(PROTOCOL)
        worlds=[]
        for split in ('train','dev','test','wording'):
            rows=data[split]
            worlds.extend({r['world'] for r in rows})
            for a,b in zip(rows[::2],rows[1::2]):
                self.assertEqual((a['world'],a['answer'],a['path']),(b['world'],b['answer'],b['path']))
                self.assertNotIn(a['marker'],a['path'])
                self.assertIn(b['marker'],b['path'][1:-1])
        self.assertEqual(len(worlds),len(set(worlds)))
        general=[r['prompt'] for s in ('rehearsal','general_dev','general_test') for r in data[s]]
        self.assertEqual(len(general),len(set(general)))

    def test_parser(self):
        row=dict(task='pointer',answer='d')
        self.assertTrue(strict_score(row,'REFUSE')['refusal'])
        self.assertFalse(strict_score(row,'REFUSE d')['refusal'])
        self.assertFalse(strict_score(row,'')['refusal'])
        self.assertFalse(strict_score(row,'Answer: d')['correct'])
        self.assertTrue(strict_score(row,'Answer: d')['semantic_correct'])
        self.assertTrue(strict_score(row,'d')['correct'])

    def test_native_routing(self):
        model=Toy()
        x=torch.randn(2,8)
        before=model(x).detach()
        route=Routing(model,'native',4,PROTOCOL,'different')
        self.assertTrue(torch.equal(before,model(x).detach()))
        frozen=route.frozen_hash()
        for group,other in [(route.cap,route.ref),(route.ref,route.cap)]:
            model.zero_grad(set_to_none=True)
            route.enable(group)
            model(x).sum().backward()
            self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 for p in group))
            self.assertTrue(all(p.grad is None for p in other))
        opt=torch.optim.AdamW(route.all,lr=.01,weight_decay=0)
        opt.step()
        self.assertEqual(frozen,route.frozen_hash())
        active=model(x).detach()
        route.enable(route.cap)
        self.assertTrue(torch.equal(active,model(x).detach()))
        state=route.state()
        route.load({n:torch.zeros_like(v) for n,v in state.items()})
        self.assertTrue(torch.equal(before,model(x).detach()))
        route.load(state)
        self.assertTrue(torch.equal(active,model(x).detach()))

    def test_nested_masks_and_counts(self):
        for geometry in ('native','scattered'):
            masks=[]
            for width in (1,4):
                route=Routing(Toy(),geometry,width,PROTOCOL)
                self.assertEqual(route.count,3*8*width)
                masks.append({e['name']:set(e.get('flat_indices',e.get('indices',[[]])[0])) for e in route.layout})
            for name,small in masks[0].items():
                self.assertLessEqual(small,masks[1][name])
        shared=Routing(Toy(),'native',4,PROTOCOL)
        self.assertEqual([id(p) for p in shared.cap],[id(p) for p in shared.ref])


if __name__=='__main__':
    unittest.main()
