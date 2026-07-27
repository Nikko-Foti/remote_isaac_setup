import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Torch is available in the Isaac Lab runtime")
class EntryBonusStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).parents[1] / "custom_tasks" / "object_in_bowl"))
        from object_in_bowl.rewards import (
            compute_lifted_object_inside_target_radius,
            reset_event_latch,
            update_first_event_latch,
        )

        cls.compute_inside = staticmethod(compute_lifted_object_inside_target_radius)
        cls.reset_latch = staticmethod(reset_event_latch)
        cls.update_latch = staticmethod(update_first_event_latch)

    def test_entry_pays_once_and_cannot_be_farmed(self):
        latch = torch.zeros(3, dtype=torch.bool)

        first = self.update_latch(torch.tensor([True, False, False]), latch)
        remains_inside = self.update_latch(torch.tensor([True, False, False]), latch)
        leaves = self.update_latch(torch.tensor([False, False, False]), latch)
        reenters = self.update_latch(torch.tensor([True, False, False]), latch)

        self.assertEqual(first.tolist(), [True, False, False])
        self.assertFalse(remains_inside.any().item())
        self.assertFalse(leaves.any().item())
        self.assertFalse(reenters.any().item())

    def test_partial_reset_only_rearms_selected_environment(self):
        latch = torch.tensor([True, True, True])

        self.reset_latch(latch, torch.tensor([1]))
        event = self.update_latch(torch.tensor([True, True, True]), latch)

        self.assertEqual(event.tolist(), [False, True, False])

    def test_full_reset_rearms_every_environment(self):
        latch = torch.tensor([True, True])

        self.reset_latch(latch)
        event = self.update_latch(torch.tensor([True, True]), latch)

        self.assertEqual(event.tolist(), [True, True])

    def test_exact_boundary_is_inside_but_exact_height_is_not_lifted(self):
        target = torch.zeros((2, 3))
        object_position = torch.tensor(
            [
                [0.08, 0.0, 0.106],
                [0.08, 0.0, 0.105],
            ]
        )

        inside = self.compute_inside(
            object_position,
            target,
            radius=0.08,
            minimal_height=0.105,
        )

        self.assertEqual(inside.tolist(), [True, False])

    def test_entry_bonus_fires_when_lift_reward_turns_off(self):
        active = torch.tensor([True, False])
        lift_latch = torch.zeros(2, dtype=torch.bool)
        bonus_latch = torch.zeros(2, dtype=torch.bool)

        self.update_latch(active, lift_latch)
        lift_reward_enabled = torch.logical_not(lift_latch)
        entry_bonus = self.update_latch(active, bonus_latch)

        self.assertEqual(lift_reward_enabled.tolist(), [False, True])
        self.assertEqual(entry_bonus.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
