"""
hybrid_agent.py

The central brain of the hybrid architecture.
Combines global A* pathfinding (WaypointManager & DynamicReplanner)
with local DDQN obstacle avoidance.
"""

from typing import Tuple, Set, Optional

Position = Tuple[int, int]


class HybridAgent:
    """
    Coordinates global A* routing with local DDQN action selection.
    """
    def __init__(
        self, 
        ddqn_agent, 
        waypoint_manager, 
        replanner, 
        astar_planner
    ):
        self.ddqn = ddqn_agent
        self.waypoint_manager = waypoint_manager
        self.replanner = replanner
        self.planner = astar_planner

    def get_action(
        self, 
        state, 
        current_pos: Position, 
        goal_pos: Position, 
        blocked_cells: Optional[Set[Position]] = None
    ) -> int:
        """
        Determine the next action by ensuring the global path is valid,
        updating the waypoint, and querying the DDQN.
        """
        if blocked_cells is None:
            blocked_cells = set()

        # 1. Initialize path if we don't have one
        if not self.waypoint_manager.path:
            initial_path = self.planner.find_path(current_pos, goal_pos, blocked_cells)
            self.waypoint_manager.set_path(initial_path)

        # 2. Check for dynamic obstacles blocking our route
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

        # 3. Advance to the next waypoint if the current one is reached
        if self.waypoint_manager.is_waypoint_reached(current_pos):
            self.waypoint_manager.advance_waypoint()

        # 4. Ask the DDQN for the optimal local movement (0-7)
        # Note: The state passed here must already include the current waypoint vector!
        action = self.ddqn.select_action(state)

        return action