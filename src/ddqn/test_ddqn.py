"""
tests/unit/test_ddqn.py

Member 2 — Day 1 Deliverable: Initial DDQN tests.

Checklist (per Day 1 plan):
  [x] network initializes
  [x] network output has 8 values
  [x] replay buffer accepts transitions
  [x] replay sampling works

Run with: pytest tests/unit/test_ddqn.py -v
(requires torch installed for the QNetwork tests)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pytest

from ddqn.replay_buffer import ReplayBuffer

STATE_DIM = 53  # matches HybridAgent.STATE_DIM; adjust once Member 1's final state lands (Day 5)


# ---------------------------------------------------------------------
# QNetwork tests (require torch — skipped automatically if unavailable)
# ---------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="torch not installed in this environment")
from ddqn.network import QNetwork, NUM_ACTIONS


def test_network_initializes():
    """Network should construct without error and hold the right dimensions."""
    net = QNetwork(state_dim=STATE_DIM)
    assert net.state_dim == STATE_DIM
    assert net.num_actions == NUM_ACTIONS == 8


def test_network_output_has_8_values():
    """A single state should produce exactly 8 Q-values (one per direction)."""
    net = QNetwork(state_dim=STATE_DIM)
    state = torch.randn(STATE_DIM)
    q_values = net(state)
    assert q_values.shape == (8,)


def test_network_output_batched():
    """A batch of states should produce a (batch_size, 8) Q-value tensor."""
    net = QNetwork(state_dim=STATE_DIM)
    batch = torch.randn(16, STATE_DIM)
    q_values = net(batch)
    assert q_values.shape == (16, 8)


def test_network_output_is_finite():
    """Freshly initialized weights shouldn't already produce NaN/inf."""
    net = QNetwork(state_dim=STATE_DIM)
    state = torch.randn(STATE_DIM)
    q_values = net(state)
    assert torch.isfinite(q_values).all()


# ---------------------------------------------------------------------
# ReplayBuffer tests (pure numpy, no torch dependency)
# ---------------------------------------------------------------------
def test_replay_buffer_starts_empty():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0


def test_replay_buffer_accepts_transitions():
    buf = ReplayBuffer(capacity=100)
    state = np.random.rand(STATE_DIM)
    next_state = np.random.rand(STATE_DIM)
    buf.push(state, action=3, reward=1.0, next_state=next_state, done=False)
    assert len(buf) == 1


def test_replay_buffer_respects_capacity():
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
        buf.push(np.zeros(STATE_DIM), 0, 0.0, np.zeros(STATE_DIM), False)
    assert len(buf) == 5  # oldest transitions dropped


def test_replay_buffer_sampling_works():
    buf = ReplayBuffer(capacity=100)
    for i in range(20):
        buf.push(np.random.rand(STATE_DIM), i % 8, float(i), np.random.rand(STATE_DIM), i == 19)

    states, actions, rewards, next_states, dones = buf.sample(batch_size=8)
    assert states.shape == (8, STATE_DIM)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, STATE_DIM)
    assert dones.shape == (8,)
    assert actions.dtype.kind in ("i", "u")   # integer action indices
    assert dones.dtype == np.bool_


def test_replay_buffer_sample_raises_when_insufficient_data():
    buf = ReplayBuffer(capacity=100)
    buf.push(np.zeros(STATE_DIM), 0, 0.0, np.zeros(STATE_DIM), False)
    with pytest.raises(ValueError):
        buf.sample(batch_size=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
