"""
Waypoint Manager for the Hybrid Architecture.

Day 6 responsibilities:
- receive A* path
- select next waypoint
- check waypoint reached
- move to next waypoint
- detect invalid waypoint
"""

from typing import List, Tuple, Optional, Set

Position = Tuple[int, int]
Path = List[Position]

class WaypointManager:
    """
    Manages the active A* path and feeds waypoints to the DDQN agent one by one.
    """
    def __init__(self):
        self.path: Path = []
        self.current_index: int = 0

    def set_path(self, path: Optional[Path]) -> None:
        """Receive the A* path and reset the tracker."""
        self.path = list(path) if path else []
        self.current_index = 0

    def get_current_waypoint(self) -> Optional[Position]:
        """Select the next waypoint the robot should target."""
        if not self.path or self.current_index >= len(self.path):
            return None
        return self.path[self.current_index]

    def is_waypoint_reached(self, robot_position: Position) -> bool:
        """Check if the robot has successfully reached the current waypoint."""
        waypoint = self.get_current_waypoint()
        if not waypoint:
            return False
        return robot_position == waypoint

    def advance_waypoint(self) -> None:
        """Move to the next waypoint in the sequence."""
        if self.current_index < len(self.path):
            self.current_index += 1

    def is_waypoint_invalid(self, blocked_cells: Set[Position]) -> bool:
        """
        Detect if the current waypoint is invalid.
        This happens if a dynamic obstacle moves onto our target cell.
        """
        waypoint = self.get_current_waypoint()
        if not waypoint:
            return False
        return waypoint in blocked_cells