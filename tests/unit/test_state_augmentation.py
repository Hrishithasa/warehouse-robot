import unittest

import numpy as np

from src.environment.state_augmentation import (
    StateAugmenter,
)


class TestStateAugmenter(unittest.TestCase):

    def setUp(self):
        self.grid = np.zeros(
            (12, 12),
            dtype=np.int8,
        )

        self.augmenter = StateAugmenter(
            static_grid=self.grid
        )

    def test_state_is_numpy_array(self):
        state = self.augmenter.build_state(
            robot_position=(5, 5),
            goal_position=(10, 10),
        )

        self.assertIsInstance(
            state,
            np.ndarray,
        )

    def test_state_is_float32(self):
        state = self.augmenter.build_state(
            robot_position=(5, 5),
            goal_position=(10, 10),
        )

        self.assertEqual(
            state.dtype,
            np.float32,
        )

    def test_state_has_expected_dimension(self):
        state = self.augmenter.build_state(
            robot_position=(5, 5),
            goal_position=(10, 10),
        )

        # 25 local
        # + 2 robot
        # + 2 goal
        # + 12 dynamic
        # + 4 waypoint
        # + 3 context
        # + 3 risk
        self.assertEqual(
            state.shape,
            (51,),
        )

    def test_local_occupancy_is_5x5(self):
        occupancy = (
            self.augmenter._local_occupancy(
                (5, 5)
            )
        )

        self.assertEqual(
            occupancy.shape,
            (25,),
        )

    def test_shelf_appears_in_local_occupancy(self):
        self.grid[5, 6] = 1

        augmenter = StateAugmenter(
            static_grid=self.grid
        )

        occupancy = (
            augmenter._local_occupancy(
                (5, 5)
            )
        )

        # Center of 5x5 is index 12.
        # East of center is index 13.
        self.assertEqual(
            occupancy[13],
            1.0,
        )

    def test_goal_relative_position(self):
        result = (
            self.augmenter
            ._goal_relative_position(
                (5, 5),
                (7, 8),
            )
        )

        self.assertGreater(
            result[0],
            0,
        )

        self.assertGreater(
            result[1],
            0,
        )

    def test_context_encoding(self):
        encoded = (
            self.augmenter
            ._context_encoding(
                "aisle"
            )
        )

        self.assertEqual(
            encoded.tolist(),
            [0.0, 1.0, 0.0],
        )

    def test_risk_encoding(self):
        encoded = (
            self.augmenter
            ._risk_encoding(
                "high"
            )
        )

        self.assertEqual(
            encoded.tolist(),
            [0.0, 0.0, 1.0],
        )

    def test_waypoint_is_encoded(self):
        waypoint = (
            self.augmenter
            ._waypoint_information(
                (5, 5),
                (6, 7),
            )
        )

        self.assertEqual(
            waypoint[3],
            1.0,
        )

    def test_missing_waypoint_is_zero(self):
        waypoint = (
            self.augmenter
            ._waypoint_information(
                (5, 5),
                None,
            )
        )

        self.assertTrue(
            np.allclose(
                waypoint,
                np.zeros(4),
            )
        )

    def test_invalid_context_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            self.augmenter.build_state(
                (5, 5),
                (10, 10),
                context="invalid",
            )

    def test_invalid_risk_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            self.augmenter.build_state(
                (5, 5),
                (10, 10),
                risk_level="critical",
            )

    def test_dynamic_obstacle_information(self):
        from src.environment.obstacles import Worker

        worker = Worker(
            position=(5, 6),
            movement_pattern="predefined_patrol",
            patrol_route=[(5, 6)],
            active=True,
        )

        state = self.augmenter.build_state(
            robot_position=(5, 5),
            goal_position=(10, 10),
            workers=[worker],
        )

        # State should still have the same fixed dimension.
        self.assertEqual(
            state.shape,
            (51,),
        )
if __name__ == "__main__":
    unittest.main()