"""
Unit tests for Day 6 DynamicReplanner.

Tests cover:

- path blockage detection
- unchanged valid path
- dynamic obstacle triggered replanning
- replanned path avoiding blocked cells
- multiple blocked cells
- no-path scenario
"""

from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner


def make_open_grid(size=8):
    """
    Create a simple obstacle-free test grid.
    """

    return [
        [0 for _ in range(size)]
        for _ in range(size)
    ]


def test_path_not_blocked():
    grid = make_open_grid()

    planner = AStarPlanner(grid)
    replanner = DynamicReplanner(planner)

    path = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]

    blocked_cells = {
        (6, 6)
    }

    assert replanner.is_path_blocked(
        path,
        blocked_cells
    ) is False


def test_path_is_blocked():
    grid = make_open_grid()

    planner = AStarPlanner(grid)
    replanner = DynamicReplanner(planner)

    path = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]

    blocked_cells = {
        (2, 2)
    }

    assert replanner.is_path_blocked(
        path,
        blocked_cells
    ) is True


def test_valid_path_is_reused():
    grid = make_open_grid()

    planner = AStarPlanner(grid)
    replanner = DynamicReplanner(planner)

    original_path = planner.find_path(
        (0, 0),
        (7, 7)
    )

    assert original_path is not None

    new_path = replanner.replan(
        current_position=(0, 0),
        goal=(7, 7),
        current_path=original_path,
        blocked_cells={(6, 0)}
    )

    assert new_path == original_path


def test_replanning_after_dynamic_obstacle_blocks_path():
    grid = make_open_grid()

    planner = AStarPlanner(grid)
    replanner = DynamicReplanner(planner)

    start = (0, 0)
    goal = (7, 7)

    original_path = planner.find_path(
        start,
        goal
    )

    assert original_path is not None
    assert len(original_path) > 2

    blocked_cell = original_path[
        len(original_path) // 2
    ]

    new_path = replanner.replan(
        current_position=start,
        goal=goal,
        current_path=original_path,
        blocked_cells={blocked_cell}
    )

    assert new_path is not None

    assert blocked_cell not in new_path


def test_replanned_path_avoids_multiple_dynamic_obstacles():
    grid = make_open_grid()

    planner = AStarPlanner(grid)
    replanner = DynamicReplanner(planner)

    start = (0, 0)
    goal = (7, 7)

    original_path = planner.find_path(
        start,
        goal
    )

    assert original_path is not None

    # Select internal cells so start and goal remain available.
    blocked_cells = set(
        original_path[2:4]
    )

    new_path = replanner.replan(
        current_position=start,
        goal=goal,
        current_path=original_path,
        blocked_cells=blocked_cells
    )

    assert new_path is not None

    for cell in blocked_cells:
        assert cell not in new_path


def test_blocked_goal_has_no_path():
    grid = make_open_grid()

    planner = AStarPlanner(grid)
    replanner = DynamicReplanner(planner)

    start = (0, 0)
    goal = (7, 7)

    original_path = planner.find_path(
        start,
        goal
    )

    assert original_path is not None

    new_path = replanner.replan(
        current_position=start,
        goal=goal,
        current_path=original_path,
        blocked_cells={goal}
    )

    assert new_path is None