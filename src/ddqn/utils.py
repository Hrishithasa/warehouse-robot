"""
src/ddqn/utils.py

Member 2 — shared small helpers for the DDQN package.
Filled in as needed (e.g. epsilon-decay schedule, seeding, action-index
<-> direction-name conversion) starting Day 2.
"""

from .network import ACTIONS


def action_index_to_name(index):
    """Converts an action index (0-7) to its direction name, e.g. 0 -> 'N'."""
    return ACTIONS[index]


def action_name_to_index(name):
    """Converts a direction name (e.g. 'NE') to its action index."""
    return ACTIONS.index(name)
