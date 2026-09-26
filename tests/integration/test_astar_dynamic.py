from src.environment.warehouse_env import WarehouseEnv
from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner

def test_astar_dynamic_replanning_and_visualization():
    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    grid = [
        [0 for _ in range(12)]
        for _ in range(12)
    ]

    planner = AStarPlanner(
        grid
    )

    replanner = DynamicReplanner(
        planner
    )

    start = (0, 0)
    goal = (11, 11)

    original_path = planner.find_path(
        start,
        goal,
    )

    assert original_path is not None
    assert original_path[0] == start
    assert original_path[-1] == goal

    env.robot_pos = start
    env.goal_pos = goal

    env.set_astar_path(
        original_path
    )

    assert (
        env.current_astar_path
        == original_path
    )

    blocked_cell = original_path[
        len(original_path) // 2
    ]

    new_path = replanner.replan(
        current_position=start,
        goal=goal,
        current_path=original_path,
        blocked_cells={
            blocked_cell
        },
    )

    assert new_path is not None
    assert blocked_cell not in new_path

    env.set_astar_path(
        new_path
    )

    assert (
        env.current_astar_path
        == new_path
    )

    env.close()