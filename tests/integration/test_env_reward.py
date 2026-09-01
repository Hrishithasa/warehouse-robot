from src.environment.warehouse_env import WarehouseEnv
from src.environment.constants import ACTIONS

def test_step_returns_adaptive_reward():

    env = WarehouseEnv(
        config="open",
        seed=42,
    )

    env.reset(seed=42)

    obs, reward, terminated, truncated, info = env.step(0)

    assert obs is not None
    assert isinstance(reward, float)

    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    assert "context" in info
    assert "risk_level" in info
    assert "obstacle_density" in info
    assert "nearest_static_distance" in info
    assert "nearest_dynamic_distance" in info
    assert "available_directions" in info

    env.close()


def test_context_is_valid():

    env = WarehouseEnv(
        config="open",
        seed=42,
    )

    env.reset(seed=42)

    _, _, _, _, info = env.step(0)

    assert info["context"] in [
        "open",
        "aisle",
        "dense",
    ]

    env.close()


def test_no_dynamic_obstacles_have_low_risk():

    env = WarehouseEnv(
        config="open",
        seed=42,
    )

    env.reset(seed=42)

    _, _, _, _, info = env.step(0)

    assert info["risk_level"] == "low"

    env.close()


def test_goal_produces_large_positive_reward():

    env = WarehouseEnv(
        config="open",
        seed=42,
    )

    env.reset(seed=42)

    # Place the robot one valid move away from the goal.
    goal_r, goal_c = env.goal_pos

    candidate_positions = [
        (goal_r - 1, goal_c),
        (goal_r + 1, goal_c),
        (goal_r, goal_c - 1),
        (goal_r, goal_c + 1),
    ]

    start_position = None

    for position in candidate_positions:
        r, c = position

        if (
            env._in_bounds(r, c)
            and not env._is_shelf(r, c)
        ):
            start_position = position
            break

    assert start_position is not None

    env.robot_pos = start_position

    # Select the action that moves directly onto the goal.
    dr = goal_r - start_position[0]
    dc = goal_c - start_position[1]

    action = None

    for action_id, direction in ACTIONS.items():
        if direction == (dr, dc):
            action = action_id
            break

    assert action is not None

    _, reward, terminated, _, info = env.step(action)

    assert terminated is True
    assert reward > 90.0
    assert info["collided"] is False

    env.close()