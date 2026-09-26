"""
utils.py

Helper functions for the A* path planner.

Day 5:
- Eight-directional movement
- Diagonal movement support
- Diagonal corner-cutting prevention

Day 6:
- Temporary dynamic obstacle blocking
- blocked_cells support
"""

from typing import Iterable, List, Tuple


Position = Tuple[int, int]


def manhattan_distance(
    start: Position,
    goal: Position
) -> int:
    """
    Compute Manhattan distance between two grid positions.
    """

    return (
        abs(start[0] - goal[0])
        + abs(start[1] - goal[1])
    )


def octile_distance(
    start: Position,
    goal: Position
) -> float:
    """
    Compute the octile-distance heuristic for 8-directional movement.

    Straight movement cost  = 1
    Diagonal movement cost  = sqrt(2)
    """

    dr = abs(start[0] - goal[0])
    dc = abs(start[1] - goal[1])

    diagonal_steps = min(dr, dc)
    straight_steps = max(dr, dc) - diagonal_steps

    return (
        diagonal_steps * 2 ** 0.5
        + straight_steps
    )


def is_valid_position(
    position: Position,
    grid: List[List[int]],
    blocked_cells: Iterable[Position] = None
) -> bool:
    """
    Check whether a position is inside the grid, is not a static
    obstacle, and is not temporarily blocked by a dynamic obstacle.

    Parameters
    ----------
    position:
        (row, col) position to check.

    grid:
        Static warehouse grid.

    blocked_cells:
        Optional collection of temporary blocked cells occupied by
        dynamic obstacles.
    """

    row, col = position

    if not grid:
        return False

    rows = len(grid)
    cols = len(grid[0])

    if row < 0 or row >= rows:
        return False

    if col < 0 or col >= cols:
        return False

    if grid[row][col] == 1:
        return False

    if blocked_cells is not None:
        if position in blocked_cells:
            return False

    return True


def is_diagonal_move(
    current: Position,
    neighbor: Position
) -> bool:
    """
    Return True when current -> neighbor is diagonal.
    """

    dr = abs(
        neighbor[0] - current[0]
    )

    dc = abs(
        neighbor[1] - current[1]
    )

    return dr == 1 and dc == 1


def is_diagonal_move_safe(
    current: Position,
    neighbor: Position,
    grid: List[List[int]],
    blocked_cells: Iterable[Position] = None
) -> bool:
    """
    Prevent diagonal corner cutting.

    For a diagonal movement, both orthogonal cells touched by the
    diagonal must be free of static and temporary dynamic obstacles.
    """

    if not is_diagonal_move(
        current,
        neighbor
    ):
        return True

    current_row, current_col = current
    neighbor_row, neighbor_col = neighbor

    row_step = (
        neighbor_row - current_row
    )

    col_step = (
        neighbor_col - current_col
    )

    side_a = (
        current_row + row_step,
        current_col,
    )

    side_b = (
        current_row,
        current_col + col_step,
    )

    return (
        is_valid_position(
            side_a,
            grid,
            blocked_cells
        )
        and
        is_valid_position(
            side_b,
            grid,
            blocked_cells
        )
    )


def get_neighbors(
    position: Position,
    grid: List[List[int]],
    blocked_cells: Iterable[Position] = None
) -> List[Position]:
    """
    Return all valid neighbouring cells using 8-directional movement.

    Dynamic obstacles are treated as temporary blocked cells.

    Diagonal corner cutting is prevented for both static and dynamic
    obstacles.
    """

    row, col = position

    candidates = [
        (row - 1, col),       # N
        (row - 1, col + 1),   # NE
        (row, col + 1),       # E
        (row + 1, col + 1),   # SE
        (row + 1, col),       # S
        (row + 1, col - 1),   # SW
        (row, col - 1),       # W
        (row - 1, col - 1),   # NW
    ]

    neighbors = []

    for cell in candidates:

        if not is_valid_position(
            cell,
            grid,
            blocked_cells
        ):
            continue

        if not is_diagonal_move_safe(
            position,
            cell,
            grid,
            blocked_cells
        ):
            continue

        neighbors.append(cell)

    return neighbors


def movement_cost(
    current: Position,
    neighbor: Position
) -> float:
    """
    Return movement cost between two adjacent cells.

    Straight movement = 1
    Diagonal movement = sqrt(2)
    """

    if is_diagonal_move(
        current,
        neighbor
    ):
        return 2 ** 0.5

    return 1.0