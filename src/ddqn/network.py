"""
src/ddqn/network.py

Member 2 — Day 1 Deliverable: Q-network.

Maps a state vector to 8 Q-values, one per direction:
    N, NE, E, SE, S, SW, W, NW

Architecture: 2 hidden layers of 128 neurons each (per the project's
"don't build unnecessarily complex ML architectures" guidance — the
novelty here is the hybrid framework and adaptive reward, not the
network size).
"""

import torch
import torch.nn as nn


# Fixed action ordering — every other module (agent, environment,
# hybrid layer) must agree on this order when mapping action index -> move.
ACTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NUM_ACTIONS = len(ACTIONS)


class QNetwork(nn.Module):
    """
    Simple feed-forward Q-network.

        state (state_dim,) -> FC(128) -> ReLU -> FC(128) -> ReLU -> FC(8)

    state_dim is configurable rather than hard-coded, since Member 1's
    final state representation (Day 5) is not required to be a fixed
    32-dim vector — whatever dimensionality the final state ends up
    being, this network adapts to it via the constructor argument.
    """

    def __init__(self, state_dim, hidden_size=128, num_actions=NUM_ACTIONS):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        """Standard Kaiming init for ReLU networks — keeps early Q-values
        from starting wildly large/small, which helps training stability."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        """
        Args:
            state: torch.Tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            torch.Tensor of shape (batch_size, num_actions) or (num_actions,)
            — one Q-value per action.
        """
        return self.net(state)
