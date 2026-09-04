"""
WarehouseEnv — core 2D grid simulation.

Member 1 Day 4:
- Environment foundation
- Dynamic obstacle registration
- Collision model
- Context classification
- Adaptive reward integration

Gymnasium-style API:

    reset() -> (observation, info)

    step(action) ->
        (observation, reward, terminated, truncated, info)
"""

from typing import Optional

import numpy as np

from src.environment.constants import (
    FREE,
    SHELF,
    ROBOT,
    GOAL,
    ACTIONS,
    NUM_ACTIONS,
    DEFAULT_GRID_SIZE,
)

from src.environment.configs import get_layout
from src.environment.collision import CollisionModel
from src.environment.context import ContextClassifier
from src.environment.rewards import AdaptiveReward


try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True

except ImportError:
    _HAS_GYM = False


_BaseEnv = gym.Env if _HAS_GYM else object


class WarehouseEnv(_BaseEnv):
    """2D grid warehouse environment."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ]
    }

    def __init__(
        self,
        config: str = "open",
        grid_size: int = DEFAULT_GRID_SIZE,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        self.config_name = config
        self.grid_size = grid_size
        self.max_steps = max_steps

        self._rng = np.random.default_rng(seed)

        # --------------------------------------------------------------
        # Static shelf layout
        # --------------------------------------------------------------

        self._static_grid = get_layout(
            config,
            grid_size,
        )

        # --------------------------------------------------------------
        # Dynamic obstacles
        # --------------------------------------------------------------

        self.workers = []
        self.dynamic_robots = []

        # --------------------------------------------------------------
        # Collision model
        # --------------------------------------------------------------

        self.collision_model = CollisionModel(
            grid_size=self.grid_size,
            static_grid=self._static_grid,
            workers=self.workers,
            dynamic_robots=self.dynamic_robots,
        )

        # --------------------------------------------------------------
        # Context classifier
        # --------------------------------------------------------------

        self.context_classifier = ContextClassifier(
            static_grid=self._static_grid,
            workers=self.workers,
            dynamic_robots=self.dynamic_robots,
        )

        # --------------------------------------------------------------
        # Adaptive reward system
        # --------------------------------------------------------------

        self.reward_system = AdaptiveReward()

        # --------------------------------------------------------------
        # Robot / goal
        # --------------------------------------------------------------

        self.robot_pos = None
        self.goal_pos = None
        self._step_count = 0

        # --------------------------------------------------------------
        # Rendering
        # --------------------------------------------------------------

        self._fig = None
        self._ax = None
        # --------------------------------------------------------------
        # A* route for visualization
        # --------------------------------------------------------------

        self.current_astar_path = None
        # --------------------------------------------------------------
        # Gymnasium spaces
        # --------------------------------------------------------------

        if _HAS_GYM:
            self.observation_space = spaces.Box(
                low=0,
                high=max(grid_size, 4),
                shape=(4,),
                dtype=np.float32,
            )

            self.action_space = spaces.Discrete(
                NUM_ACTIONS
            )

    # ==================================================================
    # Dynamic obstacle management
    # ==================================================================

    def set_dynamic_obstacles(
        self,
        workers=None,
        dynamic_robots=None,
    ):
        """
        Register dynamic obstacles with the environment,
        collision model and context classifier.
        """

        self.workers = list(
            workers or []
        )

        self.dynamic_robots = list(
            dynamic_robots or []
        )

        self.collision_model.set_dynamic_obstacles(
            workers=self.workers,
            dynamic_robots=self.dynamic_robots,
        )

        self.context_classifier.set_dynamic_obstacles(
            workers=self.workers,
            dynamic_robots=self.dynamic_robots,
        )

    # ==================================================================
    # Core API
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if seed is not None:
            self._rng = np.random.default_rng(
                seed
            )

        self._step_count = 0

        self.robot_pos = (
            self._random_free_cell()
        )

        self.goal_pos = (
            self._random_free_cell(
                exclude={
                    self.robot_pos
                }
            )
        )

        obs = self._get_obs()

        context_info = (
            self.context_classifier.classify(
                self.robot_pos
            )
        )

        info = {
            "config": self.config_name,
            "context": context_info[
                "context"
            ],
            "risk_level": context_info[
                "risk_level"
            ],
        }

        return obs, info

    def step(
        self,
        action: int,
    ):
        if action not in ACTIONS:
            raise ValueError(
                f"Invalid action {action}. "
                f"Must be 0-{NUM_ACTIONS - 1}."
            )

        self._step_count += 1

        previous_position = self.robot_pos

        dr, dc = ACTIONS[action]

        new_r = (
            self.robot_pos[0] + dr
        )

        new_c = (
            self.robot_pos[1] + dc
        )

        new_position = (
            new_r,
            new_c,
        )

        collided = False

        # --------------------------------------------------------------
        # Collision-aware movement
        # --------------------------------------------------------------

        is_diagonal = (
            abs(dr) == 1
            and abs(dc) == 1
        )

        if is_diagonal:
            can_move = (
                self.collision_model.can_move_diagonal(
                    self.robot_pos,
                    new_position,
                )
            )
        else:
            can_move = (
                self.collision_model.can_move(
                    self.robot_pos,
                    new_position,
                )
            )

        if can_move:
            self.robot_pos = new_position
        else:
            collided = True

        # --------------------------------------------------------------
        # Goal detection
        # --------------------------------------------------------------

        reached_goal = (
            self.robot_pos == self.goal_pos
        )

        # --------------------------------------------------------------
        # Context classification
        # --------------------------------------------------------------

        context_info = (
            self.context_classifier.classify(
                self.robot_pos
            )
        )

        # --------------------------------------------------------------
        # Adaptive reward
        # --------------------------------------------------------------

        reward = (
            self.reward_system.calculate(
                context=context_info["context"],
                previous_position=previous_position,
                current_position=self.robot_pos,
                goal_position=self.goal_pos,
                collided=collided,
                reached_goal=reached_goal,
                nearest_static_distance=(
                    context_info[
                        "nearest_static_distance"
                    ]
                ),
                nearest_dynamic_distance=(
                    context_info[
                        "nearest_dynamic_distance"
                    ]
                ),
                risk_level=context_info[
                    "risk_level"
                ],
            )
        )

        # --------------------------------------------------------------
        # Episode termination
        # --------------------------------------------------------------

        terminated = reached_goal

        truncated = (
            self._step_count >= self.max_steps
        )

        # --------------------------------------------------------------
        # Observation
        # --------------------------------------------------------------

        obs = self._get_obs()

        # --------------------------------------------------------------
        # Information returned to caller
        # --------------------------------------------------------------

        info = {
            "collided": collided,
            "step_count": self._step_count,
            "context": context_info["context"],
            "risk_level": context_info["risk_level"],
            "obstacle_density": (
                context_info[
                    "obstacle_density"
                ]
            ),
            "nearest_static_distance": (
                context_info[
                    "nearest_static_distance"
                ]
            ),
            "nearest_dynamic_distance": (
                context_info[
                    "nearest_dynamic_distance"
                ]
            ),
            "available_directions": (
                context_info[
                    "available_directions"
                ]
            ),
            "dynamic_obstacle_count": (
                context_info[
                    "dynamic_obstacle_count"
                ]
            ),
            "dynamic_density": (
                context_info[
                    "dynamic_density"
                ]
            ),
        }

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )
    #==================================
    #Set A star path for visualization
    #==================================
    def set_astar_path(self, path=None):
        """
        Store the current A* route for visualization.

        Parameters
        ----------
        path:
            A* path represented as a list of
            (row, column) positions.

        Notes
        -----
        This method does not perform path planning.
        It only stores the path that was produced
        by AStarPlanner.
        """

        if path is None:
            self.current_astar_path = None
            return

        self.current_astar_path = list(path)
    # ==================================================================
    # Rendering
    # ==================================================================

    def render(
        self,
        mode: str = "human",
    ):
        import matplotlib

        if mode == "rgb_array":
            matplotlib.use("Agg", force=True)

        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        if self._fig is None:
            self._fig, self._ax = (
                plt.subplots(
                    figsize=(6, 6)
                )
            )

        display_grid = (
            self._static_grid.copy()
        )

        # --------------------------------------------------------------
        # Goal and robot
        # --------------------------------------------------------------

        gr, gc = self.goal_pos
        rr, rc = self.robot_pos

        display_grid[
            gr,
            gc
        ] = GOAL

        display_grid[
            rr,
            rc
        ] = ROBOT

        # --------------------------------------------------------------
        # Base warehouse rendering
        # --------------------------------------------------------------

        cmap = ListedColormap(
            [
                "white",
                "dimgray",
                "orange",
                "royalblue",
                "limegreen",
            ]
        )

        self._ax.clear()

        self._ax.imshow(
            display_grid,
            cmap=cmap,
            vmin=0,
            vmax=4,
        )

        # --------------------------------------------------------------
        # Grid
        # --------------------------------------------------------------

        self._ax.set_xticks(
            range(self.grid_size)
        )

        self._ax.set_yticks(
            range(self.grid_size)
        )

        self._ax.grid(
            color="lightgray",
            linewidth=0.5,
        )

        # --------------------------------------------------------------
        # A* route
        # --------------------------------------------------------------

        if self.current_astar_path:
            path_rows = [
                position[0]
                for position in self.current_astar_path
            ]

            path_cols = [
                position[1]
                for position in self.current_astar_path
            ]

            self._ax.plot(
                path_cols,
                path_rows,
                linewidth=2,
                marker="o",
                markersize=3,
                label="A* route",
            )

        # --------------------------------------------------------------
        # Dynamic workers
        # --------------------------------------------------------------

        for worker in self.workers:
            if not worker.active:
                continue

            row, col = worker.position

            self._ax.scatter(
                col,
                row,
                marker="s",
                s=80,
                label="Worker",
            )

        # --------------------------------------------------------------
        # Dynamic robots
        # --------------------------------------------------------------

        for dynamic_robot in self.dynamic_robots:
            if not dynamic_robot.active:
                continue

            row, col = dynamic_robot.position

            self._ax.scatter(
                col,
                row,
                marker="D",
                s=80,
                label="Dynamic Robot",
            )

        # --------------------------------------------------------------
        # Title
        # --------------------------------------------------------------

        self._ax.set_title(
            f"{self.config_name.capitalize()} "
            f"— step {self._step_count}"
        )

        # --------------------------------------------------------------
        # Legend
        # --------------------------------------------------------------

        handles, labels = (
            self._ax.get_legend_handles_labels()
        )

        if handles:
            # Remove duplicate labels.
            unique = dict(
                zip(labels, handles)
            )

            self._ax.legend(
                unique.values(),
                unique.keys(),
                loc="upper right",
            )

        # --------------------------------------------------------------
        # Output
        # --------------------------------------------------------------

        if mode == "human":
            plt.pause(0.001)
            return None

        elif mode == "rgb_array":
            self._fig.canvas.draw()

            buf = np.asarray(
                self._fig.canvas.buffer_rgba()
            )

            return buf[:, :, :3]

        else:
            raise ValueError(
                f"Unsupported render mode: {mode}"
            )

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)

            self._fig = None
            self._ax = None

    # ==================================================================
    # Internals
    # ==================================================================

    def _get_obs(
        self,
    ) -> np.ndarray:
        """
        Current observation remains the existing 4-dimensional
        robot/goal representation.

        The final state representation will be implemented later
        according to the Member 1 Day 5 task.
        """

        return np.array(
            [
                *self.robot_pos,
                *self.goal_pos,
            ],
            dtype=np.float32,
        )

    def _in_bounds(
        self,
        r: int,
        c: int,
    ) -> bool:
        return (
            0 <= r < self.grid_size
            and
            0 <= c < self.grid_size
        )

    def _is_shelf(
        self,
        r: int,
        c: int,
    ) -> bool:
        if not self._in_bounds(
            r,
            c,
        ):
            return True

        return (
            self._static_grid[
                r,
                c
            ]
            == SHELF
        )

    def _random_free_cell(
        self,
        exclude: Optional[set] = None,
    ) -> tuple:

        exclude = (
            exclude or set()
        )

        free_cells = [
            (r, c)
            for r in range(
                self.grid_size
            )
            for c in range(
                self.grid_size
            )
            if (
                self._static_grid[
                    r,
                    c
                ]
                == FREE
                and
                (r, c)
                not in exclude
            )
        ]

        if not free_cells:
            raise RuntimeError(
                "No free cells available "
                "for placement."
            )

        idx = self._rng.integers(
            0,
            len(free_cells),
        )

        return free_cells[idx]