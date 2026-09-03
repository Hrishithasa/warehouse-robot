"""
src/ddqn/ — Member 2's DDQN core.

Day 1: QNetwork, ReplayBuffer (this file's exports).
Day 2+: DDQNAgent, Trainer (added to this package as they're built).
"""

from .replay_buffer import ReplayBuffer

try:
    from .network import QNetwork
except ImportError:
    # torch not installed in this environment — ReplayBuffer (no torch
    # dependency) still works standalone; QNetwork requires torch.
    QNetwork = None

__all__ = ["QNetwork", "ReplayBuffer"]
