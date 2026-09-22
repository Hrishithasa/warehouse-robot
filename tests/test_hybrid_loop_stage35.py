"""
tests/test_hybrid_loop_stage35.py

Focused regression tests for DDQN oscillation recovery.

Important:
The project already defines its own ACTIONS ordering in
src.environment.constants. The test therefore derives the expected
recovery action from ACTIONS instead of assuming that NW is action 7.
"""

from src.environment.constants import ACTIONS
from src.hybrid.hybrid_agent import HybridAgent


class DummyDDQN:
    def __init__(self, action=2):
        self.action = action

    def select_action(self, state):
        return self.action


class DummyWaypointManager:
    def __init__(self, waypoint=(2, 2)):
        self.path = [(1, 1), (2, 2), (3, 3)]
        self.waypoint = waypoint

    def set_path(self, path):
        self.path = list(path)

    def is_waypoint_invalid(self, blocked_cells):
        return False

    def is_waypoint_reached(self, current_pos):
        return current_pos == self.waypoint

    def advance_waypoint(self):
        pass

    def get_current_waypoint(self):
        return self.waypoint


class DummyReplanner:
    def is_path_blocked(self, path, blocked_cells):
        return False

    def replan(self, current_position, goal, current_path, blocked_cells):
        return current_path


class DummyPlanner:
    def find_path(self, current_pos, goal_pos, blocked_cells):
        return [(1, 1), (2, 2), (3, 3)]


def test_loop_recovery_breaks_two_cell_oscillation():
    agent = HybridAgent(
        DummyDDQN(action=2),
        DummyWaypointManager(waypoint=(2, 2)),
        DummyReplanner(),
        DummyPlanner(),
        loop_window=6,
    )

    state = [0.0] * 51

    positions = [
        (3, 3),
        (3, 4),
        (3, 3),
        (3, 4),
        (3, 3),
        (3, 4),
    ]

    actions = []

    for position in positions:
        actions.append(
            agent.get_action(
                state,
                current_pos=position,
                waypoint=(2, 2),
                blocked_cells=set(),
            )
        )

    # From (3,4) to waypoint (2,2), the recovery direction is (-1,-1).
    expected_recovery_action = next(
        action
        for action, move in ACTIONS.items()
        if tuple(move) == (-1, -1)
    )

    assert agent.last_loop_detected is True
    assert actions[-1] == expected_recovery_action
    assert agent.loop_recoveries == 1


def test_normal_ddqn_action_is_preserved_without_loop():
    agent = HybridAgent(
        DummyDDQN(action=2),
        DummyWaypointManager(waypoint=(5, 5)),
        DummyReplanner(),
        DummyPlanner(),
        loop_window=6,
    )

    state = [0.0] * 51

    action = agent.get_action(
        state,
        current_pos=(1, 1),
        waypoint=(5, 5),
        blocked_cells=set(),
    )

    assert action == 2
    assert agent.loop_recoveries == 0
