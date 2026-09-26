"""
Day 7 — A* to Waypoint compatibility tests.

Member 1 responsibility:
Verify that AStarPlanner produces a path representation
that can be passed directly to the waypoint-management side.

The WaypointManager itself is implemented by Member 2.
"""

from src.astar.astar_planner import AStarPlanner


def test_astar_path_is_waypoint_compatible():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    planner = AStarPlanner(grid)

    start = (0, 0)
    goal = (4, 4)

    path = planner.find_path(
        start,
        goal,
    )

    assert path is not None
    assert len(path) > 0

    # First waypoint must be the start.
    assert path[0] == start

    # Last waypoint must be the goal.
    assert path[-1] == goal

    # Every waypoint must be a (row, column) tuple.
    for waypoint in path:
        assert isinstance(waypoint, tuple)
        assert len(waypoint) == 2

        row, col = waypoint

        assert isinstance(row, int)
        assert isinstance(col, int)
'''
---------------------------------------------
Test eight direction waypoint compatibility
---------------------------------------------
'''

def test_astar_waypoints_support_eight_direction_movement():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    planner = AStarPlanner(grid)

    start = (0, 0)
    goal = (4, 4)

    path = planner.find_path(
        start,
        goal,
    )

    assert path is not None

    for current, next_position in zip(
        path,
        path[1:],
    ):
        row_change = abs(
            next_position[0] - current[0]
        )

        col_change = abs(
            next_position[1] - current[1]
        )

        # Each movement is at most one cell.
        assert row_change <= 1
        assert col_change <= 1

        # The robot must actually move.
        assert (row_change, col_change) != (0, 0)
'''
---------------------------------------------
Test waypoint path avoids blocked cells
---------------------------------------------
'''
def test_astar_waypoint_path_avoids_dynamic_blocked_cells():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    planner = AStarPlanner(grid)

    start = (0, 0)
    goal = (4, 4)

    blocked_cells = {
        (2, 2),
    }

    path = planner.find_path(
        start,
        goal,
        blocked_cells=blocked_cells,
    )

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal

    for waypoint in path:
        assert waypoint not in blocked_cells

    