"""
replanning.py

Hybrid Replanning coordinator logic.
"""

from typing import Tuple, Set, Optional, List

Position = Tuple[int, int]
Path = List[Position]

class HybridReplannerWrapper:
    """
    Coordinates checking path validity and triggering replans.
    """
    def __init__(self, replanner, waypoint_manager):
        self.replanner = replanner
        self.waypoint_manager = waypoint_manager

    def check_and_replan(
        self,
        current_pos: Position,
        goal_pos: Position,
        blocked_cells: Set[Position]
    ) -> bool:
        """
        Checks if the current route or waypoint is blocked. 
        If so, forces a replan and updates the waypoint manager.
        Returns True if a replan occurred.
        """
        is_blocked = self.replanner.is_path_blocked(self.waypoint_manager.path, blocked_cells)
        is_invalid = self.waypoint_manager.is_waypoint_invalid(blocked_cells)

        if is_blocked or is_invalid:
            new_path = self.replanner.replan(
                current_position=current_pos,
                goal=goal_pos,
                current_path=self.waypoint_manager.path,
                blocked_cells=blocked_cells
            )
            self.waypoint_manager.set_path(new_path)
            return True
            
        return False