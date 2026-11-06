import unittest
import numpy as np
from logic import auc, calibrate, summarize


def row(key, blocked, score, correct=True, case=None):
    return dict(id=key, split='dev', case=case or ('blocked' if blocked else 'allowed'),
        label='blocked' if blocked else 'allowed', margin_sum=score, margin_max=score+2.,
        greedy_refusal=score+2. > 0, greedy_path_correct=correct and score+2. <= 0,
        argmax_refusal=score+2. > 0, node_path_correct=correct)


class LogicChecks(unittest.TestCase):
    def setUp(self):
        self.rows = [row('a', False, -2.), row('b', False, -1.), row('c', True, 1.), row('d', True, 2.)]

    def test_sum_max_disagreement_can_recover_answer(self):
        self.assertGreater(.2, .04)
        self.assertLess(np.log(.2)-np.log(16*.04), 0)
        rows = [row('a', False, -.5), row('b', True, 1.)]
        self.assertEqual(summarize(rows, 'greedy')['false_refusal'], 1.)
        s = summarize(rows, 'sum_zero')
        self.assertEqual(s['false_refusal'], 0.)
        self.assertEqual(s['correct_answers_recovered'], 1)

    def test_auc_ties_and_order(self):
        self.assertEqual(auc(self.rows, 'margin_sum'), 1.)
        for r in self.rows:r['margin_sum'] = 0
        self.assertEqual(auc(self.rows, 'margin_sum'), .5)

    def test_calibration_recovers_separation(self):
        result = calibrate(self.rows, True)
        self.assertTrue(result['chosen']['feasible'])
        self.assertEqual(result['development']['worst_joint'], 1.)

    def test_selection_rejects_test_data(self):
        self.rows[0]['split'] = 'test'
        with self.assertRaises(AssertionError):calibrate(self.rows, True)

    def test_retention_failure_stays_infeasible(self):
        self.assertFalse(calibrate(self.rows, False)['chosen']['feasible'])

    def test_casewise_gate_and_bruteforce(self):
        rows = self.rows+[row('e', False, 1.5, case='hard_allowed')]
        result = calibrate(rows, True)
        for c in result['candidates']:
            s = summarize(rows, 'sum_calibrated', c['threshold'])
            self.assertAlmostEqual(c['worst_joint'], s['worst_joint'])
            self.assertAlmostEqual(c['allowed_accuracy'], s['allowed_accuracy'])
            self.assertAlmostEqual(c['blocked_refusal'], s['blocked_refusal'])
        self.assertFalse(result['chosen']['feasible'])


if __name__ == '__main__':unittest.main()
