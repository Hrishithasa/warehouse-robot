"""
Stage 3.8 regression tests.

Tests:
1. Progress-stall detection.
2. A* directional guidance during normal DDQN control.
"""

import numpy as np

from src.hybrid.hybrid_agent import HybridAgent


class DummyDDQN:
    num_actions = 8

    def __init__(self):
        self.q_values = np.array(
            [0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.84],
            dtype=np.float32,
        )

    def select_action(self, state):
        return 3

    def select_action_masked(
        self,
        state,
        valid_actions,
        epsilon=0.0,
    ):
        return max(valid_actions)

    def get_q_values(self, state):
        return self.q_values


class DummyWaypointManager:
    def __init__(self):
        self.path = []

    def set_path(self, path):
        self.path = path

    def is_waypoint_invalid(self, blocked_cells):
        return False

    def is_waypoint_reached(self, current_pos):
        return False

    def advance_waypoint(self):
        pass

    def get_current_waypoint(self):
        return (2, 2)


class DummyReplanner:
    def is_path_blocked(self, path, blocked_cells):
        return False


class DummyPlanner:
    def find_path(
        self,
        start,
        goal,
        blocked_cells=None,
    ):
        # From (3,4) to (2,2), A* recommends NW.
        if start == (3, 4) and goal == (2, 2):
            return [
                start,
                (2, 3),
                (2, 2),
            ]

        return [start, goal]


def make_agent(guidance_weight=0.18):
    return HybridAgent(
        DummyDDQN(),
        DummyWaypointManager(),
        DummyReplanner(),
        DummyPlanner(),
        loop_window=6,
        progress_window=6,
        recovery_steps=3,
        guidance_weight=guidance_weight,
    )


def test_progress_stall_is_detected():
    agent = make_agent()

    # The robot keeps moving but remains at the same Chebyshev distance
    # from the waypoint. This represents movement without useful progress.
    positions = [
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 4),
        (5, 3),
        (5, 2),
    ]

    for position in positions:
        agent.get_action(
            [0.0] * 51,
            current_pos=position,
            waypoint=(2, 2),
            blocked_cells=set(),
            valid_actions=set(range(8)),
        )

    assert agent.progress_recoveries >= 1
    assert agent.last_progress_stall is True


def test_astar_guidance_can_prefer_route_aligned_action():
    agent = make_agent(guidance_weight=0.18)

    # DDQN's raw maximum is action 3.
    # A* recommends action 7.
    #
    # Q(action 3) = 0.90
    # Q(action 7) = 0.84
    #
    # After normalization and the Stage 3.8 guidance bonus,
    # the A* aligned action should be selected.
    action = agent.get_action(
        [0.0] * 51,
        current_pos=(3, 4),
        waypoint=(2, 2),
        blocked_cells=set(),
        valid_actions=set(range(8)),
    )

    assert action == 7
    assert agent.last_astar_action == 7