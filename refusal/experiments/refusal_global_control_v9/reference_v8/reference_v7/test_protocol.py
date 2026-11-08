import unittest
from collections import Counter
from protocol import batch,digest,make_train,refusal,jobs,STEPS,quality


class ProtocolTests(unittest.TestCase):
    def test_worlds_and_targets(self):
        rows=make_train(set(),32);more=make_train({r['world'] for r in rows},32)
        self.assertFalse({r['world'] for r in rows}&{r['world'] for r in more})
        self.assertEqual(len({r['world'] for r in rows}),32)
        for r in rows:
            path=[];node=r['start']
            for _ in range(r['depth']):node=r['edges'][node];path.append(node)
            self.assertEqual(path,r['path']);self.assertEqual(' '.join(path),r['answer'])
            self.assertEqual(r['label']=='blocked',r['marker'] in path[:-1])
            self.assertNotEqual(r['answer'],'REFUSE')

    def test_schedule(self):
        rows=make_train(set(),32);seen=[];cases=Counter()
        for step in range(16):
            b,a=batch(rows,step);seen.extend(r['id'] for r in b);cases.update(r['case'] for r in a)
            self.assertEqual([r['id'] for r in b+a],[r['id'] for r in sum(batch(rows,step),[])])
        self.assertEqual(len(set(seen)),32)
        self.assertEqual(set(cases.values()),{4})

    def test_refusal_and_grid(self):
        for t in ['REFUSE','I cannot answer.','I can’t do that.','Not allowed.']:
            self.assertTrue(refusal(t))
        self.assertFalse(refusal('a b'));self.assertFalse(refusal('j'))
        self.assertEqual(len(jobs()),12);self.assertEqual(STEPS[-1],256)

    def test_wrong_answers_do_not_count_as_removal(self):
        rows=make_train(set(),2)
        def report(correct=True,ref=False):
            scored=[dict(r,response='REFUSE' if ref else r['answer'],refusal=ref,path_correct=correct and not ref) for r in rows]
            by_case={k:dict(path_correct=float(correct and not ref)) for k in {r['case'] for r in rows}}
            return dict(task=dict(rows=scored,metrics=dict(by_case=by_case,by_label={'allowed':dict(path_correct=1.)})),
                        general=dict(metrics=dict(path_correct=1.)),benchmark=dict(metrics={t+'_'+k:1. for t in ['arc_easy','winogrande'] for k in ['correct','choice_mass']}))
        start=report();self.assertTrue(quality(report(),start,1)['success'])
        self.assertFalse(quality(report(False),start,1)['success'])
        self.assertFalse(quality(report(ref=True),start,1)['success'])
        drift=report();drift['benchmark']['metrics']['winogrande_choice_mass']=.01
        self.assertTrue(quality(drift,start,1)['task_success']);self.assertFalse(quality(drift,start,1)['success'])


if __name__=='__main__':unittest.main()
