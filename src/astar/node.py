"""
node.py

Defines the Node class used by the A* path planning algorithm.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=True)
class Node:
    """
    Represents a single grid cell in the A* search.
    """

    # Used for automatic sorting in the priority queue
    f: float = field(init=False)

    # Actual data
    row: int = field(compare=False)
    col: int = field(compare=False)

    g: float = field(default=float("inf"), compare=False)
    h: float = field(default=0.0, compare=False)

    parent: Optional["Node"] = field(default=None, compare=False)

    def __post_init__(self):
        self.f = self.g + self.h

    @property
    def position(self):
        """Return node position as (row, col)."""
        return (self.row, self.col)

    def update_costs(self, g: float, h: float):
        """
        Update the node costs.
        """
        self.g = g
        self.h = h
        self.f = g + h