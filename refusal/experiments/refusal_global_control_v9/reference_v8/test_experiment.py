"""Small nonlinear numerical, frozen-support, resume and protocol checks."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn

import experiment as ex
from design import attack_jobs,batch,defense_jobs,make_train,policy_quality
from mechanism import Support,cosine_gradient,gradient


class Tests(unittest.TestCase):
    def test_interrupted_weight_commit_does_not_publish_completed_measurement(self):
        rows=make_train(set(),4)
        report=dict(task=dict(rows=[dict(r,path_correct=r['label']=='allowed') for r in rows],
            metrics=dict(by_case={r['case']:dict(path_correct=float(r['label']=='allowed'),
                refusal=float(r['label']=='blocked')) for r in rows})),
            general=dict(metrics=dict(path_correct=1.)),
            benchmark=dict(metrics={t+'_'+v:1. for t in ['arc_easy','winogrande'] for v in ['correct','choice_mass']}))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir()
            ex.write(root/'data.json',{});ex.write(parent/'baseline_dev.json',report)
            model=nn.Linear(1,1)
            fake=SimpleNamespace(count=10752,optparams=[nn.Parameter(torch.ones(1))],
                names=['weight'],values=lambda:[torch.ones(1)])
            spec=dict(parent='shared4',arm='clean',beta=0.,limit=1)
            m=dict(defense_jobs=dict(fixture=spec),config=dict(defense_lrs=dict(clean=3e-5)),
                parents=dict(shared4=dict(folder=str(parent),layout=[])),smoke=True)
            e=SimpleNamespace(core=SimpleNamespace(tensor_hash=lambda named:'same'))
            with patch.object(ex,'check',return_value=m), \
                 patch.object(ex,'model_for',return_value=(e,torch,None,model,[])), \
                 patch.object(ex,'Support',return_value=fake), \
                 patch.object(ex,'probe_cosine',return_value=(.1,{})), \
                 patch.object(ex.api,'measure',return_value=report), \
                 patch.object(ex,'defense_update',return_value=dict(step=1)), \
                 patch.object(ex,'save',side_effect=OSError('interrupted checkpoint write')):
                with self.assertRaisesRegex(OSError,'interrupted'):
                    ex.defend(root,'fixture')
            self.assertTrue((root/'defense/fixture/dev_0000.json').exists())
            self.assertFalse((root/'defense/fixture/dev_0001.json').exists())
            self.assertFalse((root/'defense/fixture/best.json').exists())

    def test_fd_matches_exact_nonlinear_cosine_derivative_and_restores_weights(self):
        torch.manual_seed(8)
        model=nn.Linear(3,2).double();support=Support(model)
        x=torch.tensor([[.2,-.7,.4],[.6,.5,-.8]],dtype=torch.double)
        a=lambda:((torch.tanh(model(x))-.3)**2).mean()
        b=lambda:((torch.sin(model(x*1.7))+.6)**2).mean()
        original=support.values();params=list(model.parameters())
        ga=torch.autograd.grad(a(),params,create_graph=True)
        gb=torch.autograd.grad(b(),params,create_graph=True)
        fa=torch.cat([v.flatten() for v in ga]);fb=torch.cat([v.flatten() for v in gb])
        exact=torch.autograd.grad(torch.dot(fa,fb)/(fa.norm()*fb.norm()),params)
        approx,_,_=cosine_gradient(support,a,b,1e-5)
        error=sum(float((x-y).square().sum()) for x,y in zip(approx,exact))**.5
        norm=sum(float(x.square().sum()) for x in exact)**.5
        self.assertLess(error/norm,.002)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(original,support.values())))

    def test_slice_optimizer_preserves_all_other_coordinates_and_resumes_exactly(self):
        torch.manual_seed(9);model=nn.Sequential(nn.Linear(5,4)).double()
        initial=copy.deepcopy(model.state_dict())
        layout=[dict(name='0',axis=0,indices=[[1,3],[0,2]])]
        def setup():
            m=nn.Sequential(nn.Linear(5,4)).double();m.load_state_dict(initial)
            s=Support(m,layout);o=torch.optim.AdamW(s.optparams,lr=.002,weight_decay=0,foreach=False)
            return m,s,o
        x=torch.arange(15,dtype=torch.double).reshape(3,5)/10
        def step(m,s,o,i):
            g,_=gradient(s,lambda:((m(x)-i*.1)**2).mean());s.apply(o,g)
        a,sa,oa=setup();b,sb,ob=setup()
        self.assertEqual(sa.count,10)
        for i in range(3):step(a,sa,oa,i)
        step(b,sb,ob,0);saved=dict(weights=sb.values(),optimizer=copy.deepcopy(ob.state_dict()))
        c,sc,oc=setup();sc.put(saved['weights']);oc.load_state_dict(saved['optimizer'])
        for i in [1,2]:step(c,sc,oc,i)
        self.assertTrue(all(torch.equal(a.state_dict()[k],c.state_dict()[k]) for k in a.state_dict()))
        self.assertTrue(torch.equal(a[0].weight[[0,2]],initial['0.weight'][[0,2]]))
        self.assertTrue(torch.equal(a[0].bias,initial['0.bias']))
        self.assertFalse(torch.equal(a[0].weight[[1,3]],initial['0.weight'][[1,3]]))

    def test_meta_restores_every_parameter_and_rng_including_nonlocal_weights(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Module()])
                self.model.layers[0].self_attn=nn.Module()
                self.model.layers[0].self_attn.q_proj=nn.Linear(2,2,bias=False)
                self.local=nn.Linear(2,2,bias=False)
            def forward(self,x):return torch.sin(self.local(self.model.layers[0].self_attn.q_proj(x)))
        torch.manual_seed(7);m=Toy();s=Support(m,[dict(name='local',axis=0,indices=[[0],[1]])])
        initial=ex.api.snapshot(m);random_before=ex.rng(torch)
        rows=make_train(set(),32);data=dict(meta_train=rows,rehearsal=[])
        e=SimpleNamespace(legacy=SimpleNamespace(nll=lambda model,tok,row,answer:
            (model(torch.tensor([[.4,.8]]))-.3).square().mean()))
        def fake_update(e,t,model,*args):
            with torch.no_grad():
                for p in model.parameters():p.add_(.01)
        cfg=dict(meta_steps=4,meta_lr=1e-6)
        with patch.object(ex.api,'update',side_effect=fake_update):
            meta,record=ex.lookahead(e,torch,m,None,s,data,[],0,cfg)
        self.assertGreater(record['meta_outside_weight_change'],0)
        self.assertTrue(all(torch.equal(p,initial[n]) for n,p in m.named_parameters()))
        self.assertTrue(torch.equal(random_before['cpu'],torch.get_rng_state()))
        self.assertEqual(random_before['python'],ex.random.getstate())
        self.assertTrue(all(torch.isfinite(g).all() for g in meta))

    def test_fresh_banks_and_fixed_grid(self):
        first=make_train(set(),16,10);second=make_train({r['world'] for r in first},16,11)
        self.assertFalse({r['world'] for r in first}&{r['world'] for r in second})
        for step in range(8):
            b,a=batch(first,step)
            self.assertEqual(len(b),2);self.assertEqual({r['depth'] for r in a},{1,2})
            self.assertTrue(all(r['answer']==' '.join(r['path']) for r in b+a))
        self.assertEqual(len(defense_jobs()),14)
        models={f'{p}__{a}':dict(parent=p,arm=a,qualified=True)
            for p in ['shared4','shared8'] for a in ['clean','global','local','lookahead']}
        self.assertEqual(len(attack_jobs(models)),40)
        models['shared4__global']['qualified']=False
        self.assertEqual(len(attack_jobs(models)),35)

    def test_bad_clean_policy_cannot_qualify_as_defense(self):
        rows=make_train(set(),4)
        def report(allowed=1.,blocked=1.,general=1.):
            by_case={r['case']:dict(path_correct=allowed if r['label']=='allowed' else 0.,
                                   refusal=blocked if r['label']=='blocked' else 0.) for r in rows}
            return dict(task=dict(rows=[dict(r,path_correct=bool(allowed) and r['label']=='allowed') for r in rows],
                                  metrics=dict(by_case=by_case)),
                        general=dict(metrics=dict(path_correct=general)),
                        benchmark=dict(metrics={t+'_'+v:1. for t in ['arc_easy','winogrande'] for v in ['correct','choice_mass']}))
        good=report()
        self.assertTrue(policy_quality(good,good,1,-.3)['qualified'])
        self.assertFalse(policy_quality(report(allowed=0.),good,1,-.99)['qualified'])
        self.assertFalse(policy_quality(report(blocked=0.),good,1,-.99)['qualified'])
        self.assertFalse(policy_quality(report(general=.5),good,1,-.99)['qualified'])


if __name__=='__main__':unittest.main()
