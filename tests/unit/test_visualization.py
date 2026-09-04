from src.environment.obstacles import (
    Worker,
    DynamicRobot,
)
from src.environment.warehouse_env import WarehouseEnv
def test_environment_accepts_astar_path():
    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    path = [
        env.robot_pos,
        env.goal_pos,
    ]

    env.set_astar_path(path)

    assert env.current_astar_path == path

    env.close()

def test_rgb_render_with_astar_path():
    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    path = [
        env.robot_pos,
        env.goal_pos,
    ]

    env.set_astar_path(path)

    image = env.render(
        mode="rgb_array"
    )

    assert image is not None
    assert image.ndim == 3
    assert image.shape[2] == 3

    env.close()

def test_rgb_render_with_dynamic_obstacles():
    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    worker = Worker(
        position=(2, 2),
        movement_pattern="predefined_patrol",
        patrol_route=[
            (2, 2),
            (2, 3),
            (2, 4),
        ],
        grid_size=(12, 12),
    )

    dynamic_robot = DynamicRobot(
        position=(8, 8),
        movement_pattern="linear",
        movement_direction=(0, 1),
        grid_size=(12, 12),
    )

    env.set_dynamic_obstacles(
        workers=[worker],
        dynamic_robots=[dynamic_robot],
    )

    image = env.render(
        mode="rgb_array"
    )

    assert image is not None
    assert image.ndim == 3
    assert image.shape[2] == 3

    env.close()