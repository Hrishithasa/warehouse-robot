"""
src/ddqn/replay_buffer.py

Member 2 — Day 1 Deliverable: Experience replay buffer.

Stores (state, action, reward, next_state, done) transitions and
supports random batch sampling for training. Capacity-limited: oldest
transitions are dropped once full.
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer.

    Usage:
        buffer = ReplayBuffer(capacity=50_000)
        buffer.push(state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = buffer.sample(batch_size=64)
    """

    def __init__(self, capacity=50_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Stores one transition.

        Args:
            state: array-like, shape (state_dim,)
            action: int, index into the 8-action space
            reward: float
            next_state: array-like, shape (state_dim,)
            done: bool, whether the episode ended on this transition
        """
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        """
        Randomly samples a batch of transitions (without replacement
        within the batch).

        Returns:
            tuple of 5 numpy arrays:
                states:      (batch_size, state_dim)
                actions:     (batch_size,)
                rewards:     (batch_size,)
                next_states: (batch_size, state_dim)
                dones:       (batch_size,)
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only has {len(self.buffer)} transitions."
            )

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)
