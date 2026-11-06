import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch
from torch import nn
import run
import data
import campaign


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.config=type('Config',(),dict(hidden_size=8,intermediate_size=512))()
        self.model=nn.Module();self.model.layers=nn.ModuleList()
        for _ in range(24):
            layer=nn.Module();layer.mlp=nn.Module()
            for name,shape in [('up_proj',(8,512)),('gate_proj',(8,512)),('down_proj',(512,8))]:setattr(layer.mlp,name,nn.Linear(*shape,bias=False))
            self.model.layers.append(layer)
    def forward(self,x):
        for layer in self.model.layers:
            m=layer.mlp;x=x+.01*m.down_proj(m.up_proj(x)*torch.sigmoid(m.gate_proj(x)))
        return x
    def gradient_checkpointing_enable(self,**kwargs):pass


class Checks(unittest.TestCase):
    def setUp(self):
        self.cfg=run.config(True)
        self.d=data.make_data(self.cfg)

    def test_truth_and_disjoint_graph_banks(self):
        banks={}
        for name,rows in self.d.items():
            self.assertEqual(len(rows),len({r['id'] for r in rows}))
            for r in rows:
                if r['task']!='pointer':continue
                node=r['start'];path=[]
                for c in r['commands']:node=r['edges'][node][c];path.append(node)
                self.assertEqual(path,r['path']);self.assertEqual(len(set([r['start'],*path])),len(path)+1)
                self.assertEqual(r['label']=='blocked',r['marker'] in path[:-1])
                self.assertEqual(run.core.digest(r['edges']),r['world'])
                banks.setdefault(r['id'].split('_')[0],set()).add(r['world'])
        self.assertEqual(sum(map(len,banks.values())),len(set().union(*banks.values())))
        for rows in [self.d['train_basic'],self.d['train_mixed']]:
            for problem in {r['problem'] for r in rows}:
                for depth in {r['depth'] for r in rows}:
                    selected=[r for r in rows if r['problem']==problem and r['depth']==depth]
                    self.assertEqual(len(selected),17)
                    self.assertEqual(len({r['answer'] for r in selected}),1)
                    self.assertEqual({r['marker'] for r in selected},{None,*data.NODES})

    def test_counterfactual_commands_markers_and_transfer(self):
        for problem in {r['problem'] for r in self.d['policy_train']}:
            rows=[r for r in self.d['policy_train'] if r['problem']==problem]
            self.assertEqual(len(rows),26);self.assertEqual(sum(r['label']=='blocked' for r in rows),6)
            a=next(r for r in rows if r['case']=='h1_endpoint');b=next(r for r in rows if r['case']=='h2_blocked_1')
            self.assertEqual(a['marker'],b['marker']);self.assertEqual(a['label'],'allowed')
            late=next(r for r in rows if r['case']=='h4_blocked_3');self.assertEqual(late['marker'],late['path'][2])
            for r in rows:
                if r['case'].endswith('_branch'):
                    parents=[r['start'],*r['path'][:-1]]
                    alternatives=[r['edges'][n]['R' if c=='L' else 'L'] for n,c in zip(parents,r['commands'])]
                    self.assertIn(r['marker'],alternatives);self.assertNotIn(r['marker'],[r['start'],*r['path']])
        for world in {r['world'] for r in self.d['policy_train']}:
            rows=[r for r in self.d['policy_train'] if r['world']==world and r['depth']==1]
            self.assertEqual({r['commands'][0] for r in rows},{'L','R'})
        self.assertEqual({r['wording'] for r in self.d['wording_test']},{2})
        self.assertTrue({r['wording'] for r in self.d['policy_train']}<={0,1})
        self.assertEqual({r['depth'] for r in self.d['ood_test']},{5,6})
        self.assertEqual(len(self.d['direction_observe']),4)
        self.assertTrue(all(r['label']!='blocked' for r in self.d['preserve_train']))

    def test_parser_and_gradient_signs(self):
        r=dict(task='pointer',depth=3,path=['k','l','p'])
        self.assertTrue(run.score(r,'k l p')['path_correct']);self.assertFalse(run.score(r,'l p')['path_correct'])
        self.assertFalse(run.score(r,'REFUSE p')['refusal'])
        r=dict(task='addition',answer='38')
        scored=run.score(r,'15 + 23 = 38');self.assertFalse(scored['path_correct']);self.assertTrue(scored['answer_parsed_correct'])
        self.assertFalse(run.score(r,'15 + 23 = 39')['answer_parsed_correct'])
        logits=torch.zeros(20,requires_grad=True)
        run.policy_loss(logits,list(range(16)),18,False).backward();self.assertGreater(logits.grad[18],0)
        logits.grad=None;run.policy_loss(logits,list(range(16)),18,True).backward();self.assertLess(logits.grad[18],0)

    def test_nested_widths_and_frozen_weights(self):
        previous=set()
        for width in [1,2,4,8]:
            torch.manual_seed(31);m=Toy();route=run.Routing(m,'native',width,self.cfg)
            self.assertEqual(route.count,24*width)
            current=set(route.layout[0]['indices'][0]);self.assertTrue(previous<=current);previous=current
            self.assertEqual({id(p) for p in route.cap},{id(p) for p in route.ref})
            before=route.frozen_hash();x=torch.randn(2,8)
            optimizer=torch.optim.AdamW(route.all,lr=.001,weight_decay=0)
            m(x).square().mean().backward();optimizer.step()
            self.assertEqual(before,route.frozen_hash())
            self.assertTrue(all(torch.count_nonzero(v)==0 for n,v in route.state().items() if n.endswith('deltas.1')))

    def test_sampling_and_actual_optimizer_resume(self):
        for per_step in [1,2,4]:
            orders={};picks=[run.pick(16,s,m,31,orders,per_step) for s in range(16//per_step) for m in range(per_step)]
            self.assertEqual(sorted(picks),list(range(16)))
        def create():
            torch.manual_seed(31);m=Toy();r=run.Routing(m,'native',4,self.cfg);o=torch.optim.AdamW(r.all,lr=.001,weight_decay=0)
            return m,r,o
        x=torch.tensor([[.1,.2,.3,.4,.5,.6,.7,.8]])
        def step(m,o):o.zero_grad(set_to_none=True);m(x).square().mean().backward();o.step()
        m,r,o=create();step(m,o);ident=run.identity('run',dict(id='cap',seed=31),'hard',r.layout)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.pt';run.core.atomic_torch(p,dict(identity=ident,fingerprint=run.core.digest(ident),parameters=r.state(),optimizer=o.state_dict(),step=1))
            saved=run.core.load_pt(p);step(m,o)
            m2,r2,o2=create();run.state_load(r2,saved,ident);o2.load_state_dict(saved['optimizer']);step(m2,o2)
            self.assertTrue(all(torch.equal(v,r2.state()[n]) for n,v in r.state().items()))
            for key,state in o.state_dict()['state'].items():
                self.assertTrue(all(torch.equal(v,o2.state_dict()['state'][key][n]) for n,v in state.items()))
            with self.assertRaises(ValueError):run.state_load(r2,saved,dict(ident,job={'id':'wrong'}))
            self.assertNotEqual(run.checkpoint_owner(saved),run.checkpoint_owner(dict(saved,step=2)))

    def test_selection_and_independent_budget_arms(self):
        baseline=dict(general_dev={'metrics':{'path_correct':1.}},benchmark_dev={'metrics':{
            t+'_'+k:v for t in ['arc_easy','winogrande'] for k,v in [('correct',.5),('choice_mass',.9)]}})
        def report(policy=True):
            rows=[]
            for r in self.d['policy_dev']:
                text='REFUSE' if policy and r['label']=='blocked' else r['answer']
                rows.append(dict(r,response=text,**run.score(r,text),truncated=False))
            return dict(task={'metrics':run.summarize(rows)},general=baseline['general_dev'],benchmark=baseline['benchmark_dev'])
        good=report();self.assertTrue(run.policy_selection(good,baseline,run.CFG,64)['feasible'])
        bad=copy.deepcopy(good);bad['task']['metrics']['by_case']['h4_blocked_3']['refusal']=0.
        self.assertFalse(run.policy_selection(bad,baseline,run.CFG,64)['feasible'])
        bad=copy.deepcopy(good);bad['general']['metrics']['path_correct']=.5
        self.assertFalse(run.policy_selection(bad,baseline,run.CFG,64)['feasible'])
        cap=report(False);self.assertTrue(run.capability_selection(cap,baseline,run.CFG,64)['feasible'])
        cap['task']['metrics']['by_depth']['4']['path_correct']=.5
        self.assertFalse(run.capability_selection(cap,baseline,run.CFG,64)['feasible'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);jobs={}
            for c in ['shared4','shared8','full']:
                for factor in run.CFG['lr_factors']:
                    key,spec=campaign.policy_job(c,31,factor,True);jobs[key]=spec
                    run.core.atomic_json(root/'jobs'/key/'policy/best.json',dict(identity={'job':dict(id=key,**spec)},selection={'rank':[1,1,1,0,-64],'feasible':True}))
            selected=campaign.select_recipes(root,jobs,run.CFG)
            self.assertFalse(selected['validation_passed']);self.assertIsNone(selected['recipes']['shared1'])
            self.assertTrue(selected['recipes']['shared8']['selection']['feasible'])
            self.assertEqual(selected['recipes']['shared8']['lr_factor'],.1)
            for w in [1,2,4,8]:self.assertEqual(campaign.policy_job(f'shared{w}',31,1.,True)[1]['teacher'],f'cap_native{w}_s31')

    def test_fixed_policy_horizon_both_depths_late_blocks_and_resume(self):
        self.exercise_phase('policy',6)

    def test_fixed_acquisition_does_not_stop_on_early_success(self):
        self.exercise_phase('hard',2)

    def exercise_phase(self,stage,limit):
        model=Toy();route=run.Routing(model,'native',4,self.cfg)
        cfg=dict(self.cfg,eval_every=2);dataset=self.d;dataset['benchmark_dev']=[]
        benchmark={'metrics':{t+'_'+k:v for t in ['arc_easy','winogrande'] for k,v in [('correct',.5),('choice_mass',.9)]}}
        baseline=dict(general_dev={'metrics':{'path_correct':1.}},benchmark_dev=benchmark)
        def evaluate(_model,_tokenizer,rows,*args):
            records=[]
            for r in rows:
                text='REFUSE' if stage=='policy' and r['label']=='blocked' else r['answer']
                records.append(dict(r,response=text,**run.score(r,text),truncated=False))
            return dict(rows=records,metrics=run.summarize(records))
        def scalar():return sum(p.sum() for p in route.all if p.requires_grad)*.01
        tokenizer=type('Tokenizer',(),{'encode':lambda self,text,**kwargs:[ord(text)] if len(text)==1 else [999,1000]})()
        job=dict(kind='policy' if stage=='policy' else 'acquire',geometry='native',width=4,placement='same',seed=31,lr_factor=1.)
        with tempfile.TemporaryDirectory() as tmp,mock.patch('run.free_space'),mock.patch('run.evaluate',side_effect=evaluate),\
             mock.patch('run.evaluate_benchmarks',return_value=benchmark),mock.patch('run.legacy.nll',side_effect=lambda *args:1.+scalar()),\
             mock.patch('run.legacy.first_logits',side_effect=lambda *args:scalar()*torch.arange(1100)) as first,\
             mock.patch('run.completion',side_effect=lambda *args:(scalar()*torch.arange(1100).repeat(3,1),torch.tensor([999,1000,0]))),\
             mock.patch('run.full_kl',side_effect=lambda *args:1.+scalar()) as kl,mock.patch('builtins.print'):
            root=Path(tmp);run.core.atomic_json(root/'manifest.json',{'test':'phase'});run.core.atomic_json(root/'base.json',baseline)
            train=dataset['policy_train'] if stage=='policy' else dataset['train_mixed']
            result,saved=run.phase_fit(root,cfg,'example',job,stage,route,tokenizer,dataset,[None]*4,'parent',train,dataset['policy_dev'],limit,
                                      teacher=[None]*len(dataset['preserve_train']) if stage=='policy' else None)
            self.assertEqual(saved['step'],limit);self.assertTrue(result['qualified'])
            if stage=='policy':
                self.assertEqual(sorted(c.args[2]['depth'] for c in kl.call_args_list if 'depth' in c.args[2]),[1]*3+[2]*3+[3]*3+[4]*3)
                self.assertIn('h4_blocked_3',[c.args[2]['case'] for c in first.call_args_list])
            directory=root/'jobs/example'/stage
            best=run.core.load_pt(directory/'best.pt');self.assertEqual(best['step'],0)
            (directory/'DONE.json').unlink();(directory/'final.pt').unlink();before=route.state()
            with mock.patch('run.legacy.nll',side_effect=AssertionError('Unexpected resumed update')):
                _,resumed=run.phase_fit(root,cfg,'example',job,stage,route,tokenizer,dataset,[None]*4,'parent',train,dataset['policy_dev'],limit,
                                       teacher=[None]*len(dataset['preserve_train']) if stage=='policy' else None)
            self.assertEqual(resumed['step'],limit);self.assertTrue(all(torch.equal(v,route.state()[n]) for n,v in before.items()))


if __name__=='__main__':unittest.main()
