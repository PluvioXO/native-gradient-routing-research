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
        for world in {r['world'] for r in d['policy_train']}:
            rows=[r for r in d['policy_train'] if r['world']==world]
            self.assertEqual(len(rows),9)
            self.assertEqual(len({r['case'] for r in rows}),9)
            self.assertEqual(sum(r['label']=='blocked' for r in rows),1)
            for row in rows:
                self.assertEqual(row['label']=='blocked',row['marker'] in row['path'][:-1])
            endpoint=next(r for r in rows if r['case']=='h1_endpoint')
            blocked=next(r for r in rows if r['case']=='h2_blocked')
            self.assertEqual(endpoint['marker'],blocked['marker'])
            self.assertEqual(endpoint['label'],'allowed')
        for depth in [1,2]:
            rows=[r for r in d['train_mixed'] if r['depth']==depth]
            self.assertEqual(len(rows),44)
        self.assertEqual(len(d['direction_observe']),4)

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

    def test_native_budget_mask_and_identical_warm_start(self):
        def create(width,placement):
            torch.manual_seed(31); model=Toy()
            return model,run.Routing(model,'native',width,run.CFG,placement)
        m,cap=create(4,'same')
        with torch.no_grad():
            for p in cap.cap:p.normal_(0,.01)
        ident=run.identity('r',dict(id='cap',geometry='native',width=4),'mixed',cap.layout)
        saved=dict(identity=ident,fingerprint=run.core.digest(ident),parameters=cap.state())
        md,diff=create(4,'different'); run.load_parent(diff,saved,cap.layout)
        ms,shared=create(8,'same'); run.load_parent(shared,saved,cap.layout)
        x=torch.randn(2,8)
        self.assertTrue(torch.equal(m(x),md(x))); self.assertTrue(torch.equal(m(x),ms(x)))
        self.assertEqual(shared.count,diff.count)
        self.assertEqual(run.effective_native_hash(cap),run.effective_native_hash(shared))
        self.assertEqual(set(shared.layout[0]['indices'][0]),set(sum(diff.layout[0]['indices'],[])))
        diff.enable(diff.ref); md(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in diff.cap))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in diff.ref))

    def test_best_feasible_checkpoint_and_group_failure(self):
        baseline=dict(general_dev={'metrics':{'path_correct':1.}},benchmark_dev={'metrics':{
            task+'_'+key:value for task in ['arc_easy','winogrande'] for key,value in [('correct',.5),('choice_mass',.9)]}})
        def report(accuracy,general):
            groups={f'h{depth}_{case}':dict(path_correct=accuracy,refusal=.01) for depth in [1,2]
                    for case in ['unmarked','start','endpoint','outside']}
            groups['h2_blocked']=dict(path_correct=.05,refusal=.95)
            return dict(task={'metrics':{'by_case':groups}},general={'metrics':{'path_correct':general}},
                        benchmark=baseline['benchmark_dev'])
        good=report(.9,.95); bad=report(1.,.5)
        self.assertTrue(run.policy_selection(good,baseline,run.CFG,32)['feasible'])
        self.assertFalse(run.policy_selection(bad,baseline,run.CFG,64)['feasible'])
        broken=copy.deepcopy(good); broken['task']['metrics']['by_case']['h1_endpoint']['path_correct']=0.
        self.assertFalse(run.policy_selection(broken,baseline,run.CFG,32)['feasible'])
        route=run.Routing(Toy(),'native',4,run.CFG)
        ident=run.identity('r',dict(id='shared'), 'policy',route.layout)
        with tempfile.TemporaryDirectory() as tmp,mock.patch('run.free_space'):
            directory=Path(tmp)
            run.consider_best(directory,route,ident,32,good,baseline,run.CFG)
            run.consider_best(directory,route,ident,64,bad,baseline,run.CFG)
            self.assertEqual(run.core.load_pt(directory/'best.pt')['step'],32)
            self.assertEqual(run.read(directory/'best.json')['selection']['step'],32)

    def test_recipe_validation_gate_and_equal_search_budget(self):
        import campaign
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); jobs={}
            for condition in campaign.CONDITIONS:
                for factor in run.CFG['lr_factors']:
                    key,spec=campaign.policy_job(condition,31,factor,True); jobs[key]=spec
                    run.core.atomic_json(root/'jobs'/key/'policy/best.json',dict(identity={'job':dict(id=key,**spec)},
                        selection=dict(rank=[1,.9,.95,-.01,-128],feasible=True)))
            selected=campaign.select_recipes(root,jobs,run.CFG)
            self.assertTrue(selected['validation_passed'])
            self.assertTrue(all(c['lr_factor']==.1 for c in selected['recipes'].values()))
            for factor in run.CFG['lr_factors']:
                key,_=campaign.policy_job('disjoint4x4',31,factor,True)
                path=root/'jobs'/key/'policy/best.json'; value=run.read(path)
                value['selection'].update(feasible=False,rank=[0,.9,.95,-.01,-128]); run.core.atomic_json(path,value)
            self.assertFalse(campaign.select_recipes(root,jobs,run.CFG)['validation_passed'])

    def test_policy_update_logging_both_depths_and_terminal_resume(self):
        model=Toy(); model.gradient_checkpointing_enable=lambda **kwargs:None
        route=run.Routing(model,'native',4,run.CFG)
        cfg=dict(run.CFG,tiny_worlds=2,train_worlds=2,dev_worlds=2,test_worlds=2,observe_worlds=2,
                 rehearsal_per_task=2,general_per_task=2,eval_every=1)
        dataset=data.make_data(cfg)
        dataset['benchmark_dev']=[]
        benchmark={'metrics':{t+'_'+k:v for t in ['arc_easy','winogrande'] for k,v in [('correct',.5),('choice_mass',.9)]}}
        baseline=dict(general_dev={'metrics':{'path_correct':1.}},benchmark_dev=benchmark)
        def evaluate(_model,_tokenizer,rows):
            records=[dict(r,response='REFUSE' if r['label']=='blocked' else r['answer'],
                          **run.score(r,'REFUSE' if r['label']=='blocked' else r['answer']),truncated=False) for r in rows]
            return dict(rows=records,metrics=run.summarize(records))
        def scalar():return sum(p.sum() for p in route.all if p.requires_grad)*.01
        tokenizer=type('Tokenizer',(),{'encode':lambda self,text,**kwargs:[ord(text)] if len(text)==1 else [999,1000]})()
        job=dict(kind='policy',geometry='native',width=4,placement='same',seed=31,lr_factor=1.)
        with tempfile.TemporaryDirectory() as tmp,mock.patch('run.free_space'),mock.patch('run.evaluate',side_effect=evaluate),\
             mock.patch('run.evaluate_benchmarks',return_value=benchmark),\
             mock.patch('run.legacy.nll',side_effect=lambda *args:1.+scalar()),\
             mock.patch('run.legacy.first_logits',side_effect=lambda *args:scalar()*torch.arange(1100)),\
             mock.patch('run.completion',side_effect=lambda *args:(scalar()*torch.arange(1100).repeat(3,1),torch.tensor([999,1000,0]))),\
             mock.patch('run.full_kl',side_effect=lambda *args:1.+scalar()) as kl,mock.patch('builtins.print'):
            root=Path(tmp); run.core.atomic_json(root/'manifest.json',{'test':'policy loop'})
            run.core.atomic_json(root/'base.json',baseline)
            allowed=[r for r in dataset['policy_train'] if r['label']!='blocked']
            result,saved=run.phase_fit(root,cfg,'policy',job,'policy',route,tokenizer,dataset,[None]*4,'parent',
                                      dataset['policy_train'],dataset['policy_dev'],1,teacher=[None]*len(allowed))
            self.assertEqual(saved['step'],1); self.assertTrue(result['qualified'])
            self.assertEqual(sorted(c.args[2]['depth'] for c in kl.call_args_list if 'depth' in c.args[2]),[1,2])
            directory=root/'jobs/policy/policy'
            (directory/'DONE.json').unlink(); (directory/'final.pt').unlink()
            with mock.patch('run.legacy.nll',side_effect=AssertionError('Unexpected extra policy update')):
                resumed,state=run.phase_fit(root,cfg,'policy',job,'policy',route,tokenizer,dataset,[None]*4,'parent',
                                            dataset['policy_train'],dataset['policy_dev'],1,teacher=[None]*len(allowed))
            self.assertEqual(state['step'],1); self.assertTrue(resumed['qualified'])

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
