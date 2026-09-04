"""
replanner.py

Dynamic A* replanning support.

Day 6 responsibility:
- Detect whether the current A* path has become blocked
  by dynamic obstacles.
- Request a new A* path when the current route is invalid.

This module does not control the robot and does not implement
DDQN or hybrid-agent behavior.
"""

from typing import Iterable, List, Optional, Tuple

from .astar_planner import AStarPlanner


Position = Tuple[int, int]
Path = List[Position]


class DynamicReplanner:
    """
    Detect and handle dynamic blockage of an existing A* route.
    """

    def __init__(
        self,
        planner: AStarPlanner
    ):
        """
        Parameters
        ----------
        planner:
            Existing AStarPlanner instance.
        """

        self.planner = planner

    def is_path_blocked(
        self,
        path: Optional[Path],
        blocked_cells: Iterable[Position]
    ) -> bool:
        """
        Return True if any cell in the current path is occupied
        by a dynamic obstacle.
        """

        if not path:
            return False

        blocked = set(
            blocked_cells or []
        )

        return any(
            position in blocked
            for position in path
        )

    def replan(
        self,
        current_position: Position,
        goal: Position,
        current_path: Optional[Path],
        blocked_cells: Iterable[Position]
    ) -> Optional[Path]:
        """
        Keep the current path if it is still valid.

        If a dynamic obstacle blocks the current path, generate
        a new A* path using the current robot position and goal.
        """

        blocked = set(
            blocked_cells or []
        )

        if not self.is_path_blocked(
            current_path,
            blocked
        ):
            return current_path

        return self.planner.find_path(
            current_position,
            goal,
            blocked_cells=blocked
        )