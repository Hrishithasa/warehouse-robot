"""
astar_planner.py

Implements the A* path planning algorithm for warehouse navigation.

Day 5 enhancement:
- 8-directional movement
- Straight movement cost = 1
- Diagonal movement cost = sqrt(2)
- Octile-distance heuristic
- Diagonal corner-cutting prevention

Day 6 enhancement:
- Temporary dynamic obstacle support through blocked_cells
"""

import heapq
from typing import Iterable, List, Tuple, Optional

from .node import Node
from .utils import (
    octile_distance,
    get_neighbors,
    movement_cost,
    is_valid_position,
)


Position = Tuple[int, int]


class AStarPlanner:
    """
    A* path planner for grid-based warehouse environments.

    Grid convention:

        0 = free
        1 = obstacle

    Movement:

        N
        NE
        E
        SE
        S
        SW
        W
        NW
    """

    def __init__(
        self,
        grid: List[List[int]]
    ):
        """
        Initialize the planner.

        Parameters
        ----------
        grid : List[List[int]]
            Warehouse grid.
        """

        if not grid:
            raise ValueError(
                "grid must not be empty."
            )

        if not grid[0]:
            raise ValueError(
                "grid must contain at least one column."
            )

        row_length = len(grid[0])

        if any(
            len(row) != row_length
            for row in grid
        ):
            raise ValueError(
                "grid must be rectangular."
            )

        self.grid = grid

    def heuristic(
        self,
        current: Position,
        goal: Position
    ) -> float:
        """
        Calculate the octile-distance heuristic.
        """

        return octile_distance(
            current,
            goal
        )

    def reconstruct_path(
        self,
        goal_node: Node
    ) -> List[Position]:
        """
        Reconstruct the path from goal to start.
        """

        path: List[Position] = []

        current = goal_node

        while current is not None:
            path.append(
                current.position
            )
            current = current.parent

        path.reverse()

        return path

    def find_path(
        self,
        start: Position,
        goal: Position,
        blocked_cells: Iterable[Position] = None
    ) -> Optional[List[Position]]:
        """
        Find the shortest path using A*.

        Parameters
        ----------
        start : (row, col)
            Starting position.

        goal : (row, col)
            Goal position.

        blocked_cells : iterable of (row, col), optional
            Temporary cells occupied by dynamic obstacles.

        Returns
        -------
        Optional[List[Position]]
            Shortest valid path, or None when no path exists.
        """

        # --------------------------------------------------------------
        # Normalize temporary dynamic obstacles
        # --------------------------------------------------------------

        blocked_cells = set(
            blocked_cells or []
        )

        # --------------------------------------------------------------
        # Validate start and goal
        # --------------------------------------------------------------

        # The robot's current position is the starting point and must
        # remain usable even if the caller accidentally includes it
        # in blocked_cells.
        if not is_valid_position(
            start,
            self.grid
        ):
            return None

        # A blocked goal cannot be reached.
        if not is_valid_position(
            goal,
            self.grid,
            blocked_cells
        ):
            return None

        # --------------------------------------------------------------
        # Start node
        # --------------------------------------------------------------

        start_node = Node(
            start[0],
            start[1]
        )

        start_node.update_costs(
            g=0.0,
            h=self.heuristic(
                start,
                goal
            )
        )

        # --------------------------------------------------------------
        # Open set
        # --------------------------------------------------------------

        open_set: List[Node] = []

        heapq.heappush(
            open_set,
            start_node
        )

        # --------------------------------------------------------------
        # Closed set
        # --------------------------------------------------------------

        closed_set = set()

        # --------------------------------------------------------------
        # Best known g-cost
        # --------------------------------------------------------------

        g_score = {
            start: 0.0
        }

        # --------------------------------------------------------------
        # Main A* loop
        # --------------------------------------------------------------

        while open_set:

            current = heapq.heappop(
                open_set
            )

            if current.position in closed_set:
                continue

            closed_set.add(
                current.position
            )

            # ----------------------------------------------------------
            # Goal reached
            # ----------------------------------------------------------

            if current.position == goal:
                return self.reconstruct_path(
                    current
                )

            # ----------------------------------------------------------
            # Explore neighbours
            # ----------------------------------------------------------

            for neighbor_pos in get_neighbors(
                current.position,
                self.grid,
                blocked_cells
            ):

                if neighbor_pos in closed_set:
                    continue

                step_cost = movement_cost(
                    current.position,
                    neighbor_pos
                )

                tentative_g = (
                    current.g
                    + step_cost
                )

                if tentative_g < g_score.get(
                    neighbor_pos,
                    float("inf")
                ):

                    g_score[
                        neighbor_pos
                    ] = tentative_g

                    neighbor = Node(
                        neighbor_pos[0],
                        neighbor_pos[1]
                    )

                    neighbor.parent = current

                    neighbor.update_costs(
                        g=tentative_g,
                        h=self.heuristic(
                            neighbor_pos,
                            goal
                        )
                    )

                    heapq.heappush(
                        open_set,
                        neighbor
                    )

        # --------------------------------------------------------------
        # No valid path
        # --------------------------------------------------------------

        return None