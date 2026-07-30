import importlib.util
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
        module_path = (
            Path(__file__).parents[1]
            / "custom_tasks"
            / "object_in_bowl"
            / "object_in_bowl"
            / "reward_state.py"
        )
        spec = importlib.util.spec_from_file_location("object_in_bowl_reward_state", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        cls.compute_inside = staticmethod(module.compute_lifted_object_inside_target_radius)
        cls.compute_action_rate = staticmethod(module.compute_action_rate_l2)
        cls.compute_excess_speed = staticmethod(module.compute_excess_speed_squared)
        cls.reset_latch = staticmethod(module.reset_event_latch)
        cls.update_latch = staticmethod(module.update_first_event_latch)

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

    def test_action_rate_excludes_first_step_after_reset(self):
        current = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
        previous = torch.zeros_like(current)
        has_previous = torch.tensor([False, True])

        penalty = self.compute_action_rate(current, previous, has_previous)

        self.assertEqual(penalty.tolist(), [0.0, 5.0])

    def test_speed_penalty_has_free_threshold_and_gate(self):
        speed = torch.tensor([0.05, 0.30, 0.30])
        active = torch.tensor([True, True, False])

        penalty = self.compute_excess_speed(speed, free_speed=0.10, active=active)

        self.assertTrue(torch.allclose(penalty, torch.tensor([0.0, 0.04, 0.0])))


if __name__ == "__main__":
    unittest.main()
