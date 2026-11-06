import copy
import random
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import study
from test_run import Checks, Toy


class EngineChecks(unittest.TestCase):
    def test_balanced_runs_real_gradient_path(self):
        # Reuse the V4 phase test, changing only the job's positive sampler.
        checker = Checks('test_fixed_policy_horizon_both_depths_late_blocks_and_resume')
        checker.setUp()
        original = study.engine.sampling.blocked_example
        seen = []
        def balanced(pools, allowed, depths, step, micro, mode):
            row = original(pools, allowed, depths, step, micro, 'balanced')
            seen.append(row['case'])
            return row
        with mock.patch.object(study.engine.sampling, 'blocked_example', side_effect=balanced):
            checker.exercise_phase('policy', 6)
        self.assertEqual(set(seen), {f'h{d}_blocked_{p}' for d in [2, 3, 4] for p in range(1, d)})

    def test_rng_is_weights_only_serializable_and_restored(self):
        engine = study.engine
        random.seed(37)
        engine.torch.manual_seed(37)
        engine.core.np.random.seed(37)
        state = engine.rng_state()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.pt'
            engine.core.atomic_torch(path, state)
            expected = (random.random(), engine.torch.rand(5), engine.core.np.random.rand(5))
            engine.restore_rng(engine.core.load_pt(path))
            actual = (random.random(), engine.torch.rand(5), engine.core.np.random.rand(5))
            self.assertEqual(expected[0], actual[0])
            self.assertTrue(engine.torch.equal(expected[1], actual[1]))
            self.assertTrue(engine.core.np.array_equal(expected[2], actual[2]))

    def test_job_matrix_and_selection_do_not_drop_failures(self):
        old = {name: dict(kind='acquire', seed=31, geometry='full' if name == 'cap_full_s31' else 'native',
                         width=0 if name == 'cap_full_s31' else int(name.split('_')[1][6:]), placement='same')
               for name in study.PARENTS.values()}
        jobs = study.make_jobs(old, study.config(False))
        policies = [j for j in jobs.values() if j['kind'] == 'policy']
        self.assertEqual(len(policies), 18)
        self.assertEqual({j['seed'] for j in policies}, {31})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for key, job in jobs.items():
                if job['kind'] == 'policy':
                    study.engine.core.atomic_json(root / 'jobs' / key / 'policy/best.json', dict(
                        identity=dict(job=dict(id=key, **job)), selection=dict(feasible=False, rank=[0, .5, .5, 0., -64])))
            result = study.recipes(root, jobs)
            self.assertEqual(len(result['recipes']), 6)
            self.assertTrue(all(not row['selection']['feasible'] for row in result['recipes'].values()))
            self.assertTrue(all(row['lr_factor'] == .1 for row in result['recipes'].values()))
            self.assertTrue(result['no_all_conditions_success_gate'])


if __name__ == '__main__':
    unittest.main()
