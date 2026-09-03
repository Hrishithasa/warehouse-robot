"""
Final state representation for the warehouse robot.

Day 5 — Member 1

State components:
1. 5x5 local occupancy
2. Robot information
3. Goal relative position
4. Dynamic obstacle information
5. A* waypoint information
6. Context
7. Risk

The representation intentionally does not force a fixed 32-dimensional
vector. The dimensionality is determined by the information that is
actually represented.
"""

from typing import Iterable, Optional, Tuple

import numpy as np

from src.environment.constants import (
    FREE,
    SHELF,
    CONTEXT_OPEN,
    CONTEXT_AISLE,
    CONTEXT_DENSE,
)


Position = Tuple[int, int]


class StateAugmenter:
    """
    Build the final environment state for the navigation agent.

    The output is a flat float32 numpy array so it can be passed
    directly to a neural network.

    State layout:

        [local_occupancy]
        [robot_position]
        [goal_relative_position]
        [dynamic_obstacles]
        [waypoint_information]
        [context_one_hot]
        [risk_one_hot]
    """

    CONTEXTS = (
        CONTEXT_OPEN,
        CONTEXT_AISLE,
        CONTEXT_DENSE,
    )

    RISKS = (
        "low",
        "medium",
        "high",
    )

    def __init__(
        self,
        static_grid: np.ndarray,
        local_radius: int = 2,
        max_dynamic_obstacles: int = 4,
    ):
        if static_grid.ndim != 2:
            raise ValueError(
                "static_grid must be a 2D array."
            )

        if (
            static_grid.shape[0]
            != static_grid.shape[1]
        ):
            raise ValueError(
                "static_grid must be square."
            )

        if local_radius != 2:
            raise ValueError(
                "Day 5 state representation requires "
                "a 5x5 local occupancy window."
            )

        if max_dynamic_obstacles < 0:
            raise ValueError(
                "max_dynamic_obstacles must be >= 0."
            )

        self.static_grid = static_grid
        self.grid_size = static_grid.shape[0]

        self.local_radius = local_radius
        self.max_dynamic_obstacles = (
            max_dynamic_obstacles
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def build_state(
        self,
        robot_position: Position,
        goal_position: Position,
        *,
        workers: Optional[Iterable] = None,
        dynamic_robots: Optional[Iterable] = None,
        waypoint: Optional[Position] = None,
        context: str = CONTEXT_OPEN,
        risk_level: str = "low",
    ) -> np.ndarray:
        """
        Build the final flattened state vector.
        """

        workers = list(workers or [])
        dynamic_robots = list(
            dynamic_robots or []
        )

        self._validate_context(
            context
        )

        self._validate_risk(
            risk_level
        )

        local_occupancy = (
            self._local_occupancy(
                robot_position
            )
        )

        robot_info = (
            self._robot_information(
                robot_position
            )
        )

        goal_relative = (
            self._goal_relative_position(
                robot_position,
                goal_position,
            )
        )

        dynamic_info = (
            self._dynamic_obstacle_information(
                robot_position,
                workers,
                dynamic_robots,
            )
        )

        waypoint_info = (
            self._waypoint_information(
                robot_position,
                waypoint,
            )
        )

        context_info = (
            self._context_encoding(
                context
            )
        )

        risk_info = (
            self._risk_encoding(
                risk_level
            )
        )

        state = np.concatenate(
            [
                local_occupancy,
                robot_info,
                goal_relative,
                dynamic_info,
                waypoint_info,
                context_info,
                risk_info,
            ]
        )

        return state.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # 1. LOCAL 5x5 OCCUPANCY
    # ------------------------------------------------------------------

    def _local_occupancy(
        self,
        robot_position: Position,
    ) -> np.ndarray:
        """
        Return a flattened 5x5 local occupancy map.

        Encoding:

            0.0 = free
            1.0 = shelf
            2.0 = outside boundary

        The robot's own cell is represented as free because the
        static grid contains only static shelf information.
        """

        row, col = robot_position

        values = []

        for dr in range(
            -self.local_radius,
            self.local_radius + 1,
        ):
            for dc in range(
                -self.local_radius,
                self.local_radius + 1,
            ):
                r = row + dr
                c = col + dc

                if not self._in_bounds(
                    r,
                    c,
                ):
                    values.append(2.0)
                elif (
                    self.static_grid[r, c]
                    == SHELF
                ):
                    values.append(1.0)
                else:
                    values.append(0.0)

        return np.asarray(
            values,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # 2. ROBOT INFORMATION
    # ------------------------------------------------------------------

    def _robot_information(
        self,
        robot_position: Position,
    ) -> np.ndarray:
        """
        Encode the robot's normalized absolute position.
        """

        row, col = robot_position

        denominator = max(
            self.grid_size - 1,
            1,
        )

        return np.asarray(
            [
                row / denominator,
                col / denominator,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # 3. GOAL RELATIVE POSITION
    # ------------------------------------------------------------------

    def _goal_relative_position(
        self,
        robot_position: Position,
        goal_position: Position,
    ) -> np.ndarray:
        """
        Encode goal displacement relative to the robot.

        Values are normalized to approximately [-1, 1].
        """

        robot_row, robot_col = (
            robot_position
        )

        goal_row, goal_col = (
            goal_position
        )

        denominator = max(
            self.grid_size - 1,
            1,
        )

        delta_row = (
            goal_row - robot_row
        ) / denominator

        delta_col = (
            goal_col - robot_col
        ) / denominator

        return np.asarray(
            [
                delta_row,
                delta_col,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # 4. DYNAMIC OBSTACLE INFORMATION
    # ------------------------------------------------------------------

    def _dynamic_obstacle_information(
        self,
        robot_position: Position,
        workers: Iterable,
        dynamic_robots: Iterable,
    ) -> np.ndarray:
        """
        Encode the nearest active dynamic obstacles.

        Each obstacle contributes:

            relative_row
            relative_col
            distance

        The number of slots is fixed by max_dynamic_obstacles.
        Empty slots are zero-padded.
        """

        obstacles = [
            obstacle
            for obstacle in (
                list(workers)
                + list(dynamic_robots)
            )
            if obstacle.active
        ]

        robot_row, robot_col = (
            robot_position
        )

        denominator = max(
            self.grid_size - 1,
            1,
        )

        obstacles.sort(
            key=lambda obstacle: (
                abs(
                    obstacle.position[0]
                    - robot_row
                )
                +
                abs(
                    obstacle.position[1]
                    - robot_col
                )
            )
        )

        values = []

        for obstacle in obstacles[
            : self.max_dynamic_obstacles
        ]:
            obstacle_row, obstacle_col = (
                obstacle.position
            )

            relative_row = (
                obstacle_row - robot_row
            ) / denominator

            relative_col = (
                obstacle_col - robot_col
            ) / denominator

            distance = (
                abs(
                    obstacle_row
                    - robot_row
                )
                +
                abs(
                    obstacle_col
                    - robot_col
                )
            ) / denominator

            values.extend(
                [
                    relative_row,
                    relative_col,
                    distance,
                ]
            )

        expected_values = (
            self.max_dynamic_obstacles
            * 3
        )

        while len(values) < expected_values:
            values.append(0.0)

        return np.asarray(
            values,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # 5. A* WAYPOINT INFORMATION
    # ------------------------------------------------------------------

    def _waypoint_information(
        self,
        robot_position: Position,
        waypoint: Optional[Position],
    ) -> np.ndarray:
        """
        Encode the current A* waypoint.

        Values:

            relative_row
            relative_col
            normalized_distance
            waypoint_available
        """

        if waypoint is None:
            return np.zeros(
                4,
                dtype=np.float32,
            )

        robot_row, robot_col = (
            robot_position
        )

        waypoint_row, waypoint_col = (
            waypoint
        )

        denominator = max(
            self.grid_size - 1,
            1,
        )

        relative_row = (
            waypoint_row - robot_row
        ) / denominator

        relative_col = (
            waypoint_col - robot_col
        ) / denominator

        distance = (
            abs(
                waypoint_row
                - robot_row
            )
            +
            abs(
                waypoint_col
                - robot_col
            )
        ) / denominator

        return np.asarray(
            [
                relative_row,
                relative_col,
                distance,
                1.0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # 6. CONTEXT
    # ------------------------------------------------------------------

    def _context_encoding(
        self,
        context: str,
    ) -> np.ndarray:
        """Return one-hot encoding for warehouse context."""

        encoded = np.zeros(
            len(self.CONTEXTS),
            dtype=np.float32,
        )

        encoded[
            self.CONTEXTS.index(context)
        ] = 1.0

        return encoded

    # ------------------------------------------------------------------
    # 7. RISK
    # ------------------------------------------------------------------

    def _risk_encoding(
        self,
        risk_level: str,
    ) -> np.ndarray:
        """Return one-hot encoding for dynamic risk."""

        encoded = np.zeros(
            len(self.RISKS),
            dtype=np.float32,
        )

        encoded[
            self.RISKS.index(risk_level)
        ] = 1.0

        return encoded

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_context(
        self,
        context: str,
    ) -> None:
        if context not in self.CONTEXTS:
            raise ValueError(
                f"Unknown context '{context}'. "
                f"Expected one of "
                f"{self.CONTEXTS}."
            )

    def _validate_risk(
        self,
        risk_level: str,
    ) -> None:
        if risk_level not in self.RISKS:
            raise ValueError(
                f"Unknown risk level '{risk_level}'. "
                f"Expected one of "
                f"{self.RISKS}."
            )

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