"""
Context classification for the warehouse environment.

Day 4 scope:
- Complete base context classification
- Open / Aisle / Dense contexts
- Dynamic obstacle risk classification
- Local obstacle density
- Nearest static obstacle distance
- Nearest dynamic obstacle distance
- Available movement directions
- Aisle structure

The classifier provides both the base warehouse context and
dynamic-risk information required by the adaptive reward system.
"""

from typing import Iterable, Optional, Tuple

import numpy as np

from src.environment.constants import (
    SHELF,
    CONTEXT_OPEN,
    CONTEXT_AISLE,
    CONTEXT_DENSE,
)


Position = Tuple[int, int]


class ContextClassifier:
    """
    Analyze the warehouse environment around the robot.

    Parameters
    ----------
    static_grid:
        2D numpy array containing FREE/SHELF cells.

    workers:
        Optional collection of Worker objects.

    dynamic_robots:
        Optional collection of DynamicRobot objects.

    local_radius:
        Radius of the local sensing window.
        Radius 2 gives a 5x5 local area.
    """

    def __init__(
        self,
        static_grid: np.ndarray,
        workers: Optional[Iterable] = None,
        dynamic_robots: Optional[Iterable] = None,
        local_radius: int = 2,
    ):
        if static_grid.ndim != 2:
            raise ValueError(
                "static_grid must be a 2D array."
            )

        if static_grid.shape[0] != static_grid.shape[1]:
            raise ValueError(
                "static_grid must be square."
            )

        if local_radius < 1:
            raise ValueError(
                "local_radius must be at least 1."
            )

        self.static_grid = static_grid
        self.grid_size = static_grid.shape[0]

        self.workers = list(workers or [])
        self.dynamic_robots = list(dynamic_robots or [])

        self.local_radius = local_radius

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def classify(
        self,
        robot_position: Position,
    ) -> dict:
        """
        Analyze the environment around the robot.

        Returns
        -------
        dict
            Complete context and risk information.
        """

        context = self._classify_context(
            robot_position
        )

        dynamic_count = self.active_dynamic_obstacle_count()

        dynamic_density = self.local_dynamic_density(
            robot_position
        )

        risk_level = self.classify_risk(
            robot_position
        )

        return {
            "context": context,
            "obstacle_density": (
                self.local_obstacle_density(
                    robot_position
                )
            ),
            "nearest_static_distance": (
                self.nearest_static_distance(
                    robot_position
                )
            ),
            "nearest_dynamic_distance": (
                self.nearest_dynamic_distance(
                    robot_position
                )
            ),
            "available_directions": (
                self.available_directions(
                    robot_position
                )
            ),
            "aisle_structure": (
                self.detect_aisle_structure(
                    robot_position
                )
            ),
            "dynamic_obstacle_count": dynamic_count,
            "dynamic_density": dynamic_density,
            "risk_level": risk_level,
        }

    # ------------------------------------------------------------------
    # DYNAMIC OBSTACLE MANAGEMENT
    # ------------------------------------------------------------------

    def set_dynamic_obstacles(
        self,
        workers: Optional[Iterable] = None,
        dynamic_robots: Optional[Iterable] = None,
    ) -> None:
        """Update the dynamic obstacles considered by the classifier."""

        self.workers = list(workers or [])
        self.dynamic_robots = list(
            dynamic_robots or []
        )

    def active_dynamic_obstacle_count(self) -> int:
        """Return the number of active dynamic obstacles."""

        return sum(
            1
            for obstacle in (
                self.workers + self.dynamic_robots
            )
            if obstacle.active
        )

    # ------------------------------------------------------------------
    # LOCAL STATIC OBSTACLE DENSITY
    # ------------------------------------------------------------------

    def local_obstacle_density(
        self,
        robot_position: Position,
    ) -> float:
        """
        Calculate the fraction of shelf cells in the local window.
        """

        row, col = robot_position

        shelf_count = 0
        total_cells = 0

        for r in range(
            row - self.local_radius,
            row + self.local_radius + 1,
        ):
            for c in range(
                col - self.local_radius,
                col + self.local_radius + 1,
            ):
                if not self._in_bounds(r, c):
                    continue

                total_cells += 1

                if self.static_grid[r, c] == SHELF:
                    shelf_count += 1

        if total_cells == 0:
            return 0.0

        return shelf_count / total_cells

    # ------------------------------------------------------------------
    # LOCAL DYNAMIC OBSTACLE DENSITY
    # ------------------------------------------------------------------

    def local_dynamic_density(
        self,
        robot_position: Position,
    ) -> float:
        """
        Calculate the fraction of cells in the local window
        occupied by active dynamic obstacles.
        """

        row, col = robot_position

        total_cells = 0
        dynamic_cells = 0

        for r in range(
            row - self.local_radius,
            row + self.local_radius + 1,
        ):
            for c in range(
                col - self.local_radius,
                col + self.local_radius + 1,
            ):
                if not self._in_bounds(r, c):
                    continue

                total_cells += 1

        if total_cells == 0:
            return 0.0

        for obstacle in (
            self.workers + self.dynamic_robots
        ):
            if not obstacle.active:
                continue

            obstacle_row, obstacle_col = (
                obstacle.position
            )

            if (
                row - self.local_radius
                <= obstacle_row
                <= row + self.local_radius
                and
                col - self.local_radius
                <= obstacle_col
                <= col + self.local_radius
                and
                self._in_bounds(
                    obstacle_row,
                    obstacle_col,
                )
            ):
                dynamic_cells += 1

        return min(
            dynamic_cells / total_cells,
            1.0,
        )

    # ------------------------------------------------------------------
    # NEAREST STATIC OBSTACLE
    # ------------------------------------------------------------------

    def nearest_static_distance(
        self,
        robot_position: Position,
    ) -> Optional[int]:
        """Return Manhattan distance to the nearest shelf."""

        shelf_positions = np.argwhere(
            self.static_grid == SHELF
        )

        if len(shelf_positions) == 0:
            return None

        row, col = robot_position

        return min(
            abs(int(r) - row)
            + abs(int(c) - col)
            for r, c in shelf_positions
        )

    # ------------------------------------------------------------------
    # NEAREST DYNAMIC OBSTACLE
    # ------------------------------------------------------------------

    def nearest_dynamic_distance(
        self,
        robot_position: Position,
    ) -> Optional[int]:
        """Return Manhattan distance to nearest active dynamic obstacle."""

        dynamic_positions = [
            obstacle.position
            for obstacle in (
                self.workers + self.dynamic_robots
            )
            if obstacle.active
        ]

        if not dynamic_positions:
            return None

        row, col = robot_position

        return min(
            abs(position[0] - row)
            + abs(position[1] - col)
            for position in dynamic_positions
        )

    # ------------------------------------------------------------------
    # AVAILABLE DIRECTIONS
    # ------------------------------------------------------------------

    def available_directions(
        self,
        robot_position: Position,
    ) -> int:
        """Count valid static free neighboring cells."""

        row, col = robot_position

        count = 0

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):

                if dr == 0 and dc == 0:
                    continue

                new_r = row + dr
                new_c = col + dc

                if not self._in_bounds(
                    new_r,
                    new_c,
                ):
                    continue

                if self.static_grid[
                    new_r,
                    new_c
                ] == SHELF:
                    continue

                count += 1

        return count

    # ------------------------------------------------------------------
    # AISLE STRUCTURE
    # ------------------------------------------------------------------

    def detect_aisle_structure(
        self,
        robot_position: Position,
    ) -> bool:
        """
        Detect a simple aisle-like structure.
        """

        row, col = robot_position

        north = self._is_shelf(
            row - 1,
            col,
        )

        south = self._is_shelf(
            row + 1,
            col,
        )

        east = self._is_shelf(
            row,
            col + 1,
        )

        west = self._is_shelf(
            row,
            col - 1,
        )

        vertical_constraint = (
            north and south
        )

        horizontal_constraint = (
            east and west
        )

        return bool(
            vertical_constraint
            or horizontal_constraint
        )

    # ------------------------------------------------------------------
    # BASE CONTEXT
    # ------------------------------------------------------------------

    def _classify_context(
        self,
        robot_position: Position,
    ) -> str:
        """
        Classify the static warehouse structure.

        Dense has priority over aisle.
        """

        density = self.local_obstacle_density(
            robot_position
        )

        if density >= 0.50:
            return CONTEXT_DENSE

        if self.detect_aisle_structure(
            robot_position
        ):
            return CONTEXT_AISLE

        return CONTEXT_OPEN

    # ------------------------------------------------------------------
    # DYNAMIC RISK
    # ------------------------------------------------------------------

    def classify_risk(
        self,
        robot_position: Position,
    ) -> str:
        """
        Classify local dynamic/navigation risk.

        Risk is determined primarily from:
        - nearest dynamic obstacle
        - local dynamic density
        - available movement directions

        Returns:
            low
            medium
            high
        """

        nearest_dynamic = (
            self.nearest_dynamic_distance(
                robot_position
            )
        )

        dynamic_density = (
            self.local_dynamic_density(
                robot_position
            )
        )

        available = (
            self.available_directions(
                robot_position
            )
        )

        # Immediate dynamic obstacle proximity.
        if (
            nearest_dynamic is not None
            and nearest_dynamic <= 1
        ):
            return "high"

        # High local dynamic density.
        if dynamic_density >= 0.12:
            return "high"

        # Restricted movement + nearby dynamic obstacle.
        if (
            nearest_dynamic is not None
            and nearest_dynamic <= 3
            and available <= 3
        ):
            return "high"

        # Moderate dynamic proximity.
        if (
            nearest_dynamic is not None
            and nearest_dynamic <= 3
        ):
            return "medium"

        if dynamic_density >= 0.05:
            return "medium"

        return "low"

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _in_bounds(
        self,
        row: int,
        col: int,
    ) -> bool:
        return (
            0 <= row < self.grid_size
            and
            0 <= col < self.grid_size
        )

    def _is_shelf(
        self,
        row: int,
        col: int,
    ) -> bool:
        """
        Out-of-bounds is treated as constrained.
        """

        if not self._in_bounds(
            row,
            col,
        ):
            return True

        return (
            self.static_grid[row, col]
            == SHELF
        )