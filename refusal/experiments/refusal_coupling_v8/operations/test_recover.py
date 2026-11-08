"""Replay checks must tolerate timing differences but detect numerical drift."""
import unittest
from recover import same_updates


class ReplayTests(unittest.TestCase):
    def test_accepts_timing_changes_and_new_unmeasured_steps(self):
        expected = [dict(loss=.25, step=1, seconds=8.)]
        actual = [dict(loss=.25, step=1, seconds=9.), dict(loss=.2, step=2)]
        self.assertTrue(same_updates(actual, expected))

    def test_rejects_changed_measurements_or_example_identity(self):
        expected = [dict(loss=.25, ids=["a"], seconds=8.)]
        self.assertFalse(same_updates([dict(loss=.26, ids=["a"])], expected))
        self.assertFalse(same_updates([dict(loss=.25, ids=["b"])], expected))


if __name__ == "__main__":
    unittest.main()
