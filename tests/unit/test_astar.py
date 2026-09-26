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

    def test_diagonal_movement_is_supported(self):
        grid = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        planner = AStarPlanner(grid)

        path = planner.find_path(
            (0, 0),
            (2, 2),
        )

        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

        # With unrestricted diagonal movement,
        # the shortest path contains two diagonal moves.
        assert len(path) == 3


    def test_diagonal_cost_is_sqrt_two(self):
        grid = [
            [0, 0],
            [0, 0],
        ]

        planner = AStarPlanner(grid)

        path = planner.find_path(
            (0, 0),
            (1, 1),
        )

        assert path == [
            (0, 0),
            (1, 1),
        ]

        assert planner.heuristic(
            (0, 0),
            (1, 1),
        ) == 2 ** 0.5


    def test_diagonal_corner_cutting_is_blocked(self):
        grid = [
            [0, 1],
            [1, 0],
        ]

        planner = AStarPlanner(grid)

        path = planner.find_path(
            (0, 0),
            (1, 1),
        )

        assert path is None


    def test_diagonal_allowed_when_both_sides_are_free(self):
        grid = [
            [0, 0],
            [0, 0],
        ]

        planner = AStarPlanner(grid)

        path = planner.find_path(
            (0, 0),
            (1, 1),
        )

        assert path == [
            (0, 0),
            (1, 1),
        ]


    def test_mixed_diagonal_and_straight_movement(self):
        grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        planner = AStarPlanner(grid)

        path = planner.find_path(
            (0, 0),
            (2, 3),
        )

        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 3)

if __name__ == "__main__":
    unittest.main()