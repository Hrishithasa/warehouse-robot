"""
astar_planner.py

Implements the A* path planning algorithm for warehouse navigation.
"""

import heapq
from typing import List, Tuple, Optional

from .node import Node
from .utils import manhattan_distance, get_neighbors

Position = Tuple[int, int]


class AStarPlanner:
    """
    A* path planner for grid-based warehouse environments.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the planner.

        Parameters
        ----------
        grid : List[List[int]]
            Warehouse grid.
            0 = Free cell
            1 = Obstacle
        """
        self.grid = grid

    def heuristic(self, current: Position, goal: Position) -> int:
        """
        Calculate Manhattan distance heuristic.
        """
        return manhattan_distance(current, goal)

    def reconstruct_path(self, goal_node: Node) -> List[Position]:
        """
        Reconstruct the path from goal to start.

        Parameters
        ----------
        goal_node : Node

        Returns
        -------
        List[Position]
            Path from start to goal.
        """
        path: List[Position] = []

        current = goal_node

        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()

        return path

    def find_path(
        self,
        start: Position,
        goal: Position
    ) -> Optional[List[Position]]:
        """
        Find the shortest path using the A* algorithm.

        Parameters
        ----------
        start : (row, col)
        goal : (row, col)

        Returns
        -------
        Optional[List[Position]]
            Shortest path if one exists, otherwise None.
        """

        # Create start node
        start_node = Node(start[0], start[1])
        start_node.update_costs(
            g=0,
            h=self.heuristic(start, goal)
        )

        # Open Set (priority queue)
        open_set: List[Node] = []
        heapq.heappush(open_set, start_node)

        # Closed Set
        closed_set = set()

        # Best known cost to each position
        g_score = {
            start: 0
        }

        while open_set:

            current = heapq.heappop(open_set)

            if current.position in closed_set:
                continue

            closed_set.add(current.position)

            # Goal reached
            if current.position == goal:
                return self.reconstruct_path(current)

            # Explore neighbours
            for neighbor_pos in get_neighbors(current.position, self.grid):

                if neighbor_pos in closed_set:
                    continue

                tentative_g = current.g + 1

                if tentative_g < g_score.get(neighbor_pos, float("inf")):

                    g_score[neighbor_pos] = tentative_g

                    neighbor = Node(
                        neighbor_pos[0],
                        neighbor_pos[1]
                    )

                    neighbor.parent = current

                    neighbor.update_costs(
                        g=tentative_g,
                        h=self.heuristic(neighbor_pos, goal)
                    )

                    heapq.heappush(open_set, neighbor)

        # No valid path found
        return None