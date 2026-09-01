"""
Adaptive reward system for warehouse navigation.

Day 4 scope:
- Step penalty
- Goal reward
- Collision penalty
- Distance-progress reward
- Waypoint reward
- Obstacle-proximity penalty
- Path-efficiency reward
- Context/risk-dependent weighting

The reward system is independent from the DDQN implementation.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple


Position = Tuple[int, int]


@dataclass
class RewardWeights:
    """
    Context-specific reward weights.

    The values are intentionally configurable so that experiments
    can tune the reward system without changing the environment code.
    """

    step: float
    progress: float
    goal: float
    collision: float
    safety: float
    waypoint: float
    efficiency: float


class AdaptiveReward:
    """
    Context-aware adaptive reward calculator.

    Reward structure:

        R =
            R_step
            + R_progress
            + R_goal
            + R_collision
            + R_safety
            + R_waypoint
            + R_efficiency

    The relative importance of the components changes according
    to the current warehouse context and dynamic risk.
    """

    DEFAULT_WEIGHTS = {
        "open": RewardWeights(
            step=-1.0,
            progress=2.0,
            goal=100.0,
            collision=-20.0,
            safety=-1.0,
            waypoint=2.0,
            efficiency=1.5,
        ),

        "aisle": RewardWeights(
            step=-1.0,
            progress=1.5,
            goal=100.0,
            collision=-30.0,
            safety=-3.0,
            waypoint=2.0,
            efficiency=1.0,
        ),

        "dense": RewardWeights(
            step=-1.0,
            progress=1.0,
            goal=100.0,
            collision=-40.0,
            safety=-5.0,
            waypoint=1.5,
            efficiency=0.5,
        ),
    }

    def __init__(
        self,
        weights: Optional[dict] = None,
        proximity_threshold: int = 3,
    ):
        self.weights = (
            weights
            if weights is not None
            else self.DEFAULT_WEIGHTS.copy()
        )

        self.proximity_threshold = (
            proximity_threshold
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def calculate(
        self,
        *,
        context: str,
        previous_position: Position,
        current_position: Position,
        goal_position: Position,
        collided: bool = False,
        reached_goal: bool = False,
        nearest_static_distance: Optional[int] = None,
        nearest_dynamic_distance: Optional[int] = None,
        waypoint_reached: bool = False,
        path_length: Optional[float] = None,
        optimal_path_length: Optional[float] = None,
        risk_level: str = "low",
    ) -> float:
        """
        Calculate the total adaptive reward.
        """

        reward_weights = self.get_weights(
            context=context,
            risk_level=risk_level,
        )

        step_reward = (
            reward_weights.step
        )

        progress_reward = (
            reward_weights.progress
            * self._progress_value(
                previous_position,
                current_position,
                goal_position,
            )
        )

        goal_reward = (
            reward_weights.goal
            if reached_goal
            else 0.0
        )

        collision_reward = (
            reward_weights.collision
            if collided
            else 0.0
        )

        safety_penalty = (
            reward_weights.safety
            * self._proximity_risk(
                nearest_static_distance,
                nearest_dynamic_distance,
            )
        )

        waypoint_reward = (
            reward_weights.waypoint
            if waypoint_reached
            else 0.0
        )

        efficiency_reward = (
            reward_weights.efficiency
            * self._path_efficiency(
                path_length,
                optimal_path_length,
            )
        )

        return float(
            step_reward
            + progress_reward
            + goal_reward
            + collision_reward
            + safety_penalty
            + waypoint_reward
            + efficiency_reward
        )

    # ------------------------------------------------------------------
    # CONTEXT ADAPTATION
    # ------------------------------------------------------------------

    def get_weights(
        self,
        *,
        context: str,
        risk_level: str = "low",
    ) -> RewardWeights:
        """
        Return reward weights for the current context and risk.
        """

        if context not in self.weights:
            raise ValueError(
                f"Unknown context '{context}'. "
                f"Expected one of: "
                f"{list(self.weights.keys())}"
            )

        base = self.weights[context]

        weights = RewardWeights(
            step=base.step,
            progress=base.progress,
            goal=base.goal,
            collision=base.collision,
            safety=base.safety,
            waypoint=base.waypoint,
            efficiency=base.efficiency,
        )

        # Dynamic risk strengthens safety-related terms.
        if risk_level == "medium":
            weights.safety *= 1.5
            weights.collision *= 1.15

        elif risk_level == "high":
            weights.safety *= 2.5
            weights.collision *= 1.5

        return weights

    # ------------------------------------------------------------------
    # PROGRESS
    # ------------------------------------------------------------------

    def _progress_value(
        self,
        previous_position: Position,
        current_position: Position,
        goal_position: Position,
    ) -> float:
        """
        Positive value when the robot moves closer to the goal.

        Negative value when the robot moves farther away.
        """

        previous_distance = self._euclidean_distance(
            previous_position,
            goal_position,
        )

        current_distance = self._euclidean_distance(
            current_position,
            goal_position,
        )

        return previous_distance - current_distance

    # ------------------------------------------------------------------
    # SAFETY / PROXIMITY
    # ------------------------------------------------------------------

    def _proximity_risk(
        self,
        nearest_static_distance: Optional[int],
        nearest_dynamic_distance: Optional[int],
    ) -> float:
        """
        Calculate proximity risk.

        Dynamic obstacles are given higher priority because they
        represent changing collision risk.
        """

        risk = 0.0

        if nearest_static_distance is not None:
            risk += self._distance_risk(
                nearest_static_distance
            )

        if nearest_dynamic_distance is not None:
            risk += 1.5 * self._distance_risk(
                nearest_dynamic_distance
            )

        return risk

    def _distance_risk(
        self,
        distance: int,
    ) -> float:
        """
        Convert distance to a bounded proximity-risk value.

        Distance 1 produces the strongest penalty.
        Risk becomes zero outside the configured threshold.
        """

        if distance <= 0:
            return 1.0

        if distance > self.proximity_threshold:
            return 0.0

        return (
            self.proximity_threshold
            - distance
            + 1
        ) / (
            self.proximity_threshold
            + 1
        )

    # ------------------------------------------------------------------
    # PATH EFFICIENCY
    # ------------------------------------------------------------------

    def _path_efficiency(
        self,
        path_length: Optional[float],
        optimal_path_length: Optional[float],
    ) -> float:
        """
        Calculate path-efficiency reward.

        Returns zero when path lengths are unavailable.
        """

        if (
            path_length is None
            or optimal_path_length is None
        ):
            return 0.0

        if path_length <= 0:
            return 0.0

        if optimal_path_length <= 0:
            return 0.0

        efficiency = (
            optimal_path_length
            / path_length
        )

        return max(
            0.0,
            min(efficiency, 1.0),
        )

    # ------------------------------------------------------------------
    # DISTANCE
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean_distance(
        first: Position,
        second: Position,
    ) -> float:
        """Return Euclidean distance between two grid cells."""

        return sqrt(
            (first[0] - second[0]) ** 2
            +
            (first[1] - second[1]) ** 2
        )