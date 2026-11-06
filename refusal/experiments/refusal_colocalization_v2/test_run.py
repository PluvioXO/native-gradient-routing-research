import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn
import run
import data
import observe


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=type('Config',(),dict(hidden_size=8,intermediate_size=512))()
        self.model=nn.Module(); self.model.layers=nn.ModuleList()
        for _ in range(24):
            layer=nn.Module(); layer.mlp=nn.Module()
            for name,shape in [('up_proj',(8,512)),('gate_proj',(8,512)),('down_proj',(512,8))]:
                setattr(layer.mlp,name,nn.Linear(*shape,bias=False))
            self.model.layers.append(layer)
    def forward(self,x):
        for layer in self.model.layers:
            m=layer.mlp; x=x+.01*m.down_proj(m.up_proj(x)*torch.sigmoid(m.gate_proj(x)))
        return x


class Checks(unittest.TestCase):
    def test_counterfactual_data(self):
        cfg=copy.deepcopy(run.CFG)
        cfg.update(tiny_worlds=2,train_worlds=4,dev_worlds=2,test_worlds=2,observe_worlds=2)
        d=data.make_data(cfg)
        for name in ['train_one','train_two']:
            groups={w:[r for r in d[name] if r['world']==w] for w in {r['world'] for r in d[name]}}
            for rows in groups.values():
                self.assertEqual(len(rows),11)
                self.assertEqual({r['marker'] for r in rows},{None,*'abcdefghij'})
                self.assertEqual(len({r['answer'] for r in rows}),1)
                for r in rows:
                    p=r['start']; visited=[]
                    for _ in range(r['depth']):p=r['edges'][p]; visited.append(p)
                    self.assertEqual(visited,r['path'])
        splits=[{r['world'] for r in d[k]} for k in ['tiny_one','train_one','dev_one','test_one','policy_observe']]
        self.assertEqual(sum(map(len,splits)),len(set().union(*splits)))
        for a,b in zip(d['policy_train'][::2],d['policy_train'][1::2]):
            self.assertEqual((a['world'],a['answer']),(b['world'],b['answer']))
            self.assertNotIn(a['marker'],a['path'][:-1]); self.assertIn(b['marker'],b['path'][:-1])

    def test_balanced_policy_gradient(self):
        logits=torch.zeros(20,requires_grad=True)
        negative=run.policy_loss(logits,list(range(10)),15,False)
        negative.backward(); self.assertGreater(logits.grad[15],0)
        logits.grad=None
        positive=run.policy_loss(logits,list(range(10)),15,True)
        positive.backward(); self.assertLess(logits.grad[15],0)

    def test_parser_and_observation(self):
        row=dict(task='pointer',path=['b','c'],depth=2)
        self.assertTrue(run.score(row,'b c')['path_correct'])
        self.assertTrue(run.score(row,'b -> c')['path_correct'])
        self.assertFalse(run.score(row,'c')['path_correct'])
        self.assertFalse(run.score(row,'REFUSE c')['refusal'])
        self.assertEqual(observe.cosine(torch.tensor([1.,0]),torch.tensor([0.,1])),0.)
        self.assertEqual(run.CFG['direction_penalty'],0)
        self.assertFalse(run.CFG['direction_intervention'])

    def test_sample_coverage(self):
        for count in (2,4):
            cache={}; picks=[run.pick(16,s,m,31,cache,per_step=count) for s in range(16//count) for m in range(count)]
            self.assertEqual(sorted(picks),list(range(16)))

    def test_actual_resumed_update_and_job_identity(self):
        x=torch.tensor([[.1,.2,.3,.4,.5,.6,.7,.8]])
        def create():
            torch.manual_seed(31); model=Toy()
            route=run.Routing(model,'native',4,run.CFG,'different')
            opt=torch.optim.AdamW(route.all,lr=.001,weight_decay=0)
            return model,route,opt
        def step(model,route,opt,index):
            opt.zero_grad(set_to_none=True)
            for group in [route.cap,route.ref]:
                route.enable(group); model(x).square().mean().backward()
            route.enable(route.all)
            for g in opt.param_groups:g['lr']=.001*(index+1)/2
            opt.step()
        m,r,o=create(); ident=run.identity('r',{'id':'different'},'tiny',r.layout)
        step(m,r,o,0)
        with tempfile.TemporaryDirectory() as tmp:
            # Small CPU checkpoints do not need the production filesystem guard.
            path=Path(tmp)/'state.pt'
            run.core.atomic_torch(path,dict(identity=ident,fingerprint=run.core.digest(ident),parameters=r.state(),
                optimizer=o.state_dict(),step=1))
            saved=run.core.load_pt(path)
            step(m,r,o,1)
            m2,r2,o2=create(); run.state_load(r2,saved,ident); o2.load_state_dict(saved['optimizer'])
            step(m2,r2,o2,1)
            self.assertTrue(all(torch.equal(v,r2.state()[n]) for n,v in r.state().items()))
            a,b=o.state_dict(),o2.state_dict()
            self.assertEqual(a['param_groups'],b['param_groups'])
            for key in a['state']:
                self.assertTrue(all(torch.equal(v,b['state'][key][name]) for name,v in a['state'][key].items()))
            wrong=run.identity('r',{'id':'shared'},'tiny',r2.layout)
            with self.assertRaises(ValueError):run.state_load(r2,saved,wrong)

    def test_resume_after_saved_gate_does_not_train_more(self):
        model=Toy(); model.gradient_checkpointing_enable=lambda **kwargs:None
        route=run.Routing(model,'native',4,run.CFG)
        cfg=copy.deepcopy(run.CFG); cfg['consecutive_passes']=1
        job=dict(kind='acquire',geometry='native',width=4,placement='same',conserve=0.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); directory=root/'jobs/example/tiny'; directory.mkdir(parents=True)
            manifest={'test':'terminal resume'}
            run.core.atomic_json(root/'manifest.json',manifest)
            run.core.atomic_json(root/'base.json',{'general_dev':{'metrics':{'path_correct':1.}}})
            ident=run.identity(run.core.digest(manifest),dict(id='example',**job),'tiny',route.layout)
            opt=torch.optim.AdamW(route.cap,lr=cfg['sparse_lr'],weight_decay=0)
            with mock.patch('run.free_space'):
                run.save_checkpoint(directory/'checkpoint.pt',route,opt,1,[{'step':1}],ident,dict(passes=1,bad=0))
            report={'task':{'metrics':{'path_correct':1.,'by_label':{'allowed':{'path_correct':1.}}}}}
            run.core.atomic_json(directory/'dev_0001.json',report)
            before=route.state()
            tokenizer=type('Tokenizer',(),{'encode':lambda self,text,**kwargs:[ord(text)] if len(text)==1 else [999,1000]})()
            with mock.patch('run.free_space'),mock.patch('run.legacy.nll',side_effect=AssertionError('Unexpected resumed update')):
                result,saved=run.phase_fit(root,cfg,'example',job,'tiny',route,tokenizer,{},[],None,[],[],5,.98)
            self.assertTrue(result['qualified']); self.assertEqual(saved['step'],1)
            self.assertTrue(all(torch.equal(v,route.state()[n]) for n,v in before.items()))


if __name__=='__main__':unittest.main()
