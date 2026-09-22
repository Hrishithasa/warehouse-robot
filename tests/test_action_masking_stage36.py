"""
tests/test_action_masking_stage36.py

Regression tests for collision-aware DDQN action masking.
"""

import torch

from src.ddqn.new_agent import DDQNAgent
from src.environment.constants import ACTIONS


def test_masked_action_never_selects_invalid_action():
    agent = DDQNAgent(
        state_dim=51,
        num_actions=8,
        epsilon_start=0.0,
        epsilon_min=0.0,
    )

    # Force deterministic Q-values through a lightweight fake network.
    class FakeNetwork:
        def __call__(self, state):
            return torch.tensor(
                [[0.1, 0.2, 0.3, 9.0, 0.5, 0.6, 0.7, 0.8]],
                dtype=torch.float32,
                device=state.device,
            )

    agent.online_network = FakeNetwork()

    # Action 3 has the highest raw Q-value, but it is invalid.
    valid_actions = {0, 1, 2, 4, 5, 6, 7}

    selected = agent.select_action_masked(
        [0.0] * 51,
        valid_actions,
        epsilon=0.0,
    )

    assert selected == 7
    assert selected in valid_actions
    assert selected != 3


def test_valid_action_set_from_project_mapping():
    # Confirm the project still exposes exactly eight actions.
    assert len(ACTIONS) == 8
    assert set(ACTIONS.keys()) == set(range(8))
