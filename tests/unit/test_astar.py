import unittest

from src.astar import AStarPlanner


class TestAStarPlanner(unittest.TestCase):

    def setUp(self):
        self.grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ]

        self.planner = AStarPlanner(self.grid)

    def test_path_exists(self):
        path = self.planner.find_path((0, 0), (3, 3))

        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 3))

    def test_start_equals_goal(self):
        path = self.planner.find_path((2, 2), (2, 2))

        self.assertEqual(path, [(2, 2)])

    def test_no_path(self):
        grid = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]

        planner = AStarPlanner(grid)

        path = planner.find_path((0, 0), (2, 2))

        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()