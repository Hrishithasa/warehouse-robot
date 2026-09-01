import unittest

from src.environment.rewards import (
    AdaptiveReward,
)


class TestAdaptiveReward(unittest.TestCase):

    def setUp(self):
        self.reward = AdaptiveReward()

    def test_goal_reward_is_positive(self):
        value = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 1),
            goal_position=(1, 1),
            reached_goal=True,
        )

        self.assertGreater(
            value,
            90.0,
        )

    def test_collision_is_penalized(self):
        value = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(0, 0),
            goal_position=(5, 5),
            collided=True,
        )

        self.assertLess(
            value,
            -15.0,
        )

    def test_progress_towards_goal_is_positive(self):
        value = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 1),
            goal_position=(5, 5),
        )

        self.assertGreater(
            value,
            -1.0,
        )

    def test_moving_away_has_lower_reward(self):
        towards = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 1),
            goal_position=(5, 5),
        )

        away = self.reward.calculate(
            context="open",
            previous_position=(1, 1),
            current_position=(0, 0),
            goal_position=(5, 5),
        )

        self.assertGreater(
            towards,
            away,
        )

    def test_waypoint_reward(self):
        without_waypoint = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 0),
            goal_position=(5, 5),
            waypoint_reached=False,
        )

        with_waypoint = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 0),
            goal_position=(5, 5),
            waypoint_reached=True,
        )

        self.assertGreater(
            with_waypoint,
            without_waypoint,
        )

    def test_proximity_penalty(self):
        far = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 0),
            goal_position=(5, 5),
            nearest_dynamic_distance=10,
        )

        near = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 0),
            goal_position=(5, 5),
            nearest_dynamic_distance=1,
            risk_level="high",
        )

        self.assertGreater(
            far,
            near,
        )

    def test_context_weights_differ(self):
        open_weights = self.reward.get_weights(
            context="open"
        )

        dense_weights = self.reward.get_weights(
            context="dense"
        )

        self.assertNotEqual(
            open_weights.safety,
            dense_weights.safety,
        )

        self.assertNotEqual(
            open_weights.collision,
            dense_weights.collision,
        )

    def test_high_risk_increases_safety_weight(self):
        low = self.reward.get_weights(
            context="open",
            risk_level="low",
        )

        high = self.reward.get_weights(
            context="open",
            risk_level="high",
        )

        self.assertGreater(
            abs(high.safety),
            abs(low.safety),
        )

        self.assertGreater(
            abs(high.collision),
            abs(low.collision),
        )

    def test_path_efficiency(self):
        inefficient = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 0),
            goal_position=(5, 5),
            path_length=20,
            optimal_path_length=10,
        )

        efficient = self.reward.calculate(
            context="open",
            previous_position=(0, 0),
            current_position=(1, 0),
            goal_position=(5, 5),
            path_length=10,
            optimal_path_length=10,
        )

        self.assertGreater(
            efficient,
            inefficient,
        )

    def test_invalid_context(self):
        with self.assertRaises(ValueError):
            self.reward.get_weights(
                context="invalid"
            )


if __name__ == "__main__":
    unittest.main()