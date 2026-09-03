"""
src/ddqn/config.py

Member 2 — STUB (implemented Day 4).

Will hold configurable hyperparameters: batch size, learning rate,
discount factor (gamma), replay warm-up steps, target update frequency,
epsilon schedule, checkpoint frequency.
"""

# Placeholder defaults — finalized Day 4 during training stabilization.
DEFAULT_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "target_update_frequency": 500,   # steps
    "replay_warmup": 1000,            # steps before training starts
    "checkpoint_frequency": 50,       # episodes
}
