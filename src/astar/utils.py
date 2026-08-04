"""
utils.py

Helper functions for the A* path planner.
"""

from typing import List, Tuple


Position = Tuple[int, int]


def manhattan_distance(start: Position, goal: Position) -> int:
    """
    Compute the Manhattan distance between two grid positions.

    Parameters
    ----------
    start : (row, col)
    goal : (row, col)

    Returns
    -------
    int
        Manhattan distance.
    """
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def is_valid_position(
    position: Position,
    grid: List[List[int]]
) -> bool:
    """
    Check whether a position is inside the grid and not blocked.

    Grid convention:
        0 = free cell
        1 = obstacle
    """

    row, col = position

    rows = len(grid)
    cols = len(grid[0])

    if row < 0 or row >= rows:
        return False

    if col < 0 or col >= cols:
        return False

    if grid[row][col] == 1:
        return False

    return True


def get_neighbors(
    position: Position,
    grid: List[List[int]]
) -> List[Position]:
    """
    Return all valid neighbouring cells.

    Movement is limited to four directions:
    Up, Down, Left, Right.
    """

    row, col = position

    candidates = [
        (row - 1, col),   # Up
        (row + 1, col),   # Down
        (row, col - 1),   # Left
        (row, col + 1),   # Right
    ]

    neighbors = []

    for cell in candidates:
        if is_valid_position(cell, grid):
            neighbors.append(cell)

    return neighbors