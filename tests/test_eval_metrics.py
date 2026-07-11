import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "rsl_rl"))

from eval_metrics import get_new_episode_log_weight


class EpisodeLogWeightTest(unittest.TestCase):
    def test_ignores_persisted_log_when_no_episode_finished(self):
        weights = [
            get_new_episode_log_weight(done_count=1024, reset_count=1024.0),
            get_new_episode_log_weight(done_count=0, reset_count=1024.0),
        ]

        self.assertEqual(sum(weights), 1024.0)

    def test_uses_done_count_when_reset_count_is_missing(self):
        self.assertEqual(get_new_episode_log_weight(done_count=12, reset_count=None), 12.0)

    def test_rejects_reset_count_that_disagrees_with_done_count(self):
        with self.assertRaisesRegex(ValueError, "reset count"):
            get_new_episode_log_weight(done_count=12, reset_count=24.0)


if __name__ == "__main__":
    unittest.main()
