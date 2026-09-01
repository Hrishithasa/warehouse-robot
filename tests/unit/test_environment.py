import unittest

from src.environment.warehouse_env import WarehouseEnv


class TestWarehouseEnvironment(unittest.TestCase):

    def test_default_grid_size(self):
        env = WarehouseEnv(config="open")

        self.assertEqual(env.grid_size, 12)

        env.close()

    def test_open_layout(self):
        env = WarehouseEnv(config="open")

        self.assertEqual(env._static_grid.shape, (12, 12))

        env.close()

    def test_aisle_layout(self):
        env = WarehouseEnv(config="aisle")

        self.assertEqual(env._static_grid.shape, (12, 12))

        env.close()

    def test_dense_layout(self):
        env = WarehouseEnv(config="dense")

        self.assertEqual(env._static_grid.shape, (12, 12))

        env.close()

    def test_reset_returns_valid_observation(self):
        env = WarehouseEnv(config="open", seed=42)

        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (4,))
        self.assertIn("config", info)

        env.close()

    def test_random_start_and_goal_are_different(self):
        env = WarehouseEnv(config="open", seed=42)

        env.reset(seed=42)

        self.assertNotEqual(
            env.robot_pos,
            env.goal_pos
        )

        env.close()

    def test_eight_direction_action_space(self):
        env = WarehouseEnv(config="open")

        self.assertEqual(
            env.action_space.n,
            8
        )

        env.close()

    def test_goal_detection(self):
        env = WarehouseEnv(config="open", seed=42)

        env.reset(seed=42)

        # Choose a known free cell and place the goal one cell east.
        env.robot_pos = (6, 5)
        env.goal_pos = (6, 6)

        # Make sure the test positions are free.
        self.assertFalse(env._is_shelf(6, 5))
        self.assertFalse(env._is_shelf(6, 6))

        # Action 2 = East.
        obs, reward, terminated, truncated, info = env.step(2)

        self.assertEqual(env.robot_pos, (6, 6))
        self.assertEqual(env.robot_pos, env.goal_pos)
        self.assertTrue(terminated)
        self.assertGreater(reward, 90.0)

        env.close()

    def test_shelf_collision(self):
        env = WarehouseEnv(config="aisle", seed=42)

        env.reset(seed=42)

        # Find an actual shelf cell.
        shelf_positions = [
            (r, c)
            for r in range(env.grid_size)
            for c in range(env.grid_size)
            if env._is_shelf(r, c)
        ]

        self.assertGreater(len(shelf_positions), 0)

        shelf_position = shelf_positions[0]

        # Put robot immediately to the left of a shelf
        r, c = shelf_position

        if c > 0 and not env._is_shelf(r, c - 1):
            env.robot_pos = (r, c - 1)

            obs, reward, terminated, truncated, info = env.step(2)

            self.assertTrue(info["collided"])
            self.assertEqual(env.robot_pos, (r, c - 1))

        env.close()

    def test_gymnasium_step_interface(self):
        env = WarehouseEnv(config="open", seed=42)

        env.reset(seed=42)

        result = env.step(0)

        self.assertEqual(len(result), 5)

        obs, reward, terminated, truncated, info = result

        self.assertEqual(obs.shape, (4,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

        env.close()


if __name__ == "__main__":
    unittest.main()