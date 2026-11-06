import random
import unittest
from collections import Counter
import sampling


class SamplingChecks(unittest.TestCase):
    def setUp(self):
        self.pools = {('world0', depth): [dict(id=f'h{depth}p{position}', problem='world0',
            depth=depth, case=f'h{depth}_blocked_{position}', label='blocked')
            for position in range(1, depth)] for depth in [2, 3, 4]}

    def test_original_is_exact_historical_schedule(self):
        counts = Counter()
        for step in range(1024):
            for micro, depth in enumerate(([1, 3] if step % 2 == 0 else [2, 4])):
                a = dict(problem='world0', depth=depth)
                actual = sampling.blocked_example(self.pools, a, [1, 2, 3, 4], step, micro, 'original')
                old = self.pools[('world0', max(2, depth))]
                self.assertEqual(actual, old[(step // 2 + micro) % len(old)])
                counts[actual['case']] += 1
        self.assertEqual(counts, {'h2_blocked_1': 1024, 'h3_blocked_1': 256, 'h3_blocked_2': 256,
                                  'h4_blocked_1': 170, 'h4_blocked_2': 171, 'h4_blocked_3': 171})

    def test_balanced_covers_every_group_and_resumes_by_step(self):
        counts = Counter()
        for step in range(1024):
            for micro in range(2):
                row = sampling.blocked_example(self.pools, dict(problem='world0', depth=1),
                                               [1, 2, 3, 4], step, micro, 'balanced')
                counts[row['case']] += 1
        self.assertEqual(len(counts), 6)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        a = dict(problem='world0', depth=4)
        before = [sampling.blocked_example(self.pools, a, [1, 2, 3, 4], s, 1, 'balanced')['id'] for s in range(1024)]
        self.assertEqual(before[641:], [sampling.blocked_example(self.pools, a, [1, 2, 3, 4], s, 1, 'balanced')['id'] for s in range(641, 1024)])
        with self.assertRaises(ValueError):
            sampling.blocked_example(self.pools, a, [1, 2, 3, 4], 0, 0, 'typo')

    def test_schedules_keep_nonpositive_examples_identical(self):
        data = dict(policy_train=[], preserve_train=[], rehearsal=[dict(id='c0'), dict(id='c1')])
        for depth in [1, 2, 3, 4]:
            data['policy_train'].append(dict(id=f'a{depth}', label='allowed', depth=depth, problem='world0'))
            data['policy_train'].extend(self.pools.get(('world0', depth), []))
            data['preserve_train'].append(dict(id=f't{depth}', depth=depth))
        def pick(n, step, micro, seed, orders, per_step=1):
            return (step * per_step + micro) % n
        audit = sampling.audit(data, dict(policy_steps=1024, train_depths=[1, 2, 3, 4], seed=31), pick)
        self.assertTrue(audit['identical_allowed_teacher_conservation'])


if __name__ == '__main__':
    unittest.main()
