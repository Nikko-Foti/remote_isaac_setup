import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "rsl_rl"))

from eval_metrics import (
    get_new_episode_log_weight,
    summarize_distribution,
    validate_episode_log_total,
    validate_exclusive_end_counts,
    validate_sequential_funnel,
)


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

    def test_rejects_near_but_unequal_large_reset_count(self):
        with self.assertRaisesRegex(ValueError, "reset count"):
            get_new_episode_log_weight(done_count=1_000_000_000, reset_count=1_000_000_001.0)

    def test_final_total_requires_exact_episode_count(self):
        with self.assertRaisesRegex(ValueError, "aggregated weight"):
            validate_episode_log_total(completed_episodes=1_000_000_000, logged_episode_weight=1_000_000_001.0)


class DistributionSummaryTest(unittest.TestCase):
    def test_summarizes_all_samples_instead_of_averaging_batch_medians(self):
        summary = summarize_distribution([0.1, 0.2, 0.3, 10.0])

        self.assertAlmostEqual(summary["mean"], 2.65)
        self.assertAlmostEqual(summary["median"], 0.25)
        self.assertAlmostEqual(summary["p10"], 0.13)
        self.assertAlmostEqual(summary["p90"], 7.09)

    def test_empty_distribution_has_no_statistics(self):
        self.assertEqual(summarize_distribution([]), {})


class PlacementLogValidationTest(unittest.TestCase):
    def test_accepts_exclusive_end_counts(self):
        validate_exclusive_end_counts(
            completed_episodes=100,
            success_count=4.0,
            timeout_count=90.0,
            drop_count=5.0,
            other_count=1.0,
        )

    def test_rejects_missing_end_reason(self):
        with self.assertRaisesRegex(ValueError, "end-reason count"):
            validate_exclusive_end_counts(
                completed_episodes=100,
                success_count=4.0,
                timeout_count=90.0,
                drop_count=5.0,
                other_count=0.0,
            )

    def test_accepts_monotonic_stage_funnel(self):
        validate_sequential_funnel(
            [("completed", 100.0), ("grasp", 90.0), ("lift", 80.0), ("success", 5.0)]
        )

    def test_rejects_non_monotonic_stage_funnel(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_sequential_funnel(
                [("completed", 100.0), ("grasp", 90.0), ("lift", 95.0)]
            )


if __name__ == "__main__":
    unittest.main()
