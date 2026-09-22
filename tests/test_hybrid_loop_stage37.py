"""
Stage 3.7 regression tests.

The important change is that loop recovery lasts for multiple actions and
uses a fresh A* route from the robot's current position.
"""

from src.hybrid.hybrid_agent import HybridAgent


class DummyDDQN:
    num_actions = 8

    def __init__(self, action=2):
        self.action = action

    def select_action(self, state):
        return self.action

    def select_action_masked(self, state, valid_actions, epsilon=0.0):
        if self.action in valid_actions:
            return self.action
        return sorted(valid_actions)[0]


class DummyWaypointManager:
    def __init__(self, waypoint=(2, 2)):
        self.path = [(3, 3), (3, 4), (2, 4), (2, 3), (2, 2)]
        self.waypoint = waypoint

    def set_path(self, path):
        self.path = path

    def is_waypoint_invalid(self, blocked_cells):
        return False

    def is_waypoint_reached(self, current_pos):
        return current_pos == self.waypoint

    def advance_waypoint(self):
        return None

    def get_current_waypoint(self):
        return self.waypoint


class DummyReplanner:
    def is_path_blocked(self, path, blocked_cells):
        return False


class DummyPlanner:
    def __init__(self):
        self.calls = []

    def find_path(self, start, goal, blocked_cells=None):
        self.calls.append((start, goal))
        # From (3,4) toward (2,2), local A* chooses NW first.
        return [start, (2, 3), (2, 2)]


def make_agent(recovery_steps=4):
    planner = DummyPlanner()
    agent = HybridAgent(
        DummyDDQN(action=2),
        DummyWaypointManager(),
        DummyReplanner(),
        planner,
        loop_window=6,
        recovery_steps=recovery_steps,
    )
    return agent, planner


def test_loop_recovery_persists_for_multiple_steps():
    agent, planner = make_agent(recovery_steps=3)

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
                valid_actions=set(range(8)),
            )
        )

    # The loop is detected at the final position. The recovery action is
    # the first action of a fresh local A* route: NW = action 7.
    assert actions[-1] == 7
    assert agent.loop_recoveries == 1

    # Recovery remains active after the trigger instead of immediately
    # returning control to DDQN.
    assert agent._recovery_remaining == 2
    assert planner.calls[-1] == ((3, 4), (2, 2))


def test_normal_ddqn_action_is_preserved_without_loop():
    agent, planner = make_agent()

    action = agent.get_action(
        [0.0] * 51,
        current_pos=(1, 1),
        waypoint=(1, 3),
        blocked_cells=set(),
        valid_actions=set(range(8)),
    )

    assert action == 2
    assert agent.loop_recoveries == 0
    assert planner.calls == []
