"""
src/hybrid/hybrid_agent.py

Stage 3.8:
- A* global route + DDQN local control
- Collision-aware action masking
- Oscillation detection
- Progress-aware stagnation detection
- Fresh local A* recovery
- A* directional guidance for DDQN decisions
"""

from collections import deque
from typing import Tuple, Set, Optional, Iterable

from src.environment.constants import ACTIONS

Position = Tuple[int, int]


class HybridAgent:
    """Coordinates global A* routing with local DDQN obstacle avoidance."""

    def __init__(
        self,
        ddqn_agent,
        waypoint_manager,
        replanner,
        astar_planner,
        loop_window: int = 6,
        min_progress: int = 1,
        recovery_steps: int = 4,
        progress_window: int = 6,
        guidance_weight: float = 0.18,
    ):
        self.ddqn = ddqn_agent
        self.waypoint_manager = waypoint_manager
        self.replanner = replanner
        self.planner = astar_planner

        self.loop_window = max(4, int(loop_window))
        self.min_progress = max(1, int(min_progress))
        self.recovery_steps = max(1, int(recovery_steps))
        self.progress_window = max(3, int(progress_window))
        self.guidance_weight = max(0.0, float(guidance_weight))

        self.position_history = deque(maxlen=self.loop_window)
        self.distance_history = deque(maxlen=self.progress_window)

        self.loop_recoveries = 0
        self.progress_recoveries = 0

        self.last_loop_detected = False
        self.last_progress_stall = False
        self.last_action_masked = False
        self.last_valid_actions = []
        self.last_recovery_reason = None
        self.last_astar_action = None

        self._recovery_remaining = 0

        self._action_lookup = {
            tuple(move): action
            for action, move in ACTIONS.items()
        }

    def reset_navigation_memory(self):
        self.position_history.clear()
        self.distance_history.clear()

        self.loop_recoveries = 0
        self.progress_recoveries = 0

        self.last_loop_detected = False
        self.last_progress_stall = False
        self.last_action_masked = False
        self.last_valid_actions = []
        self.last_recovery_reason = None
        self.last_astar_action = None

        self._recovery_remaining = 0

    # ------------------------------------------------------------------
    # A* route / waypoint management
    # ------------------------------------------------------------------

    def update_route_and_waypoint(
        self,
        current_pos: Position,
        goal_pos: Position,
        blocked_cells: Optional[Set[Position]] = None,
    ) -> Position:
        if blocked_cells is None:
            blocked_cells = set()

        if not self.waypoint_manager.path:
            initial_path = self.planner.find_path(
                current_pos,
                goal_pos,
                blocked_cells,
            )
            self.waypoint_manager.set_path(initial_path)

        is_blocked = self.replanner.is_path_blocked(
            self.waypoint_manager.path,
            blocked_cells,
        )

        is_invalid = self.waypoint_manager.is_waypoint_invalid(
            blocked_cells,
        )

        if is_blocked or is_invalid:
            new_path = self.replanner.replan(
                current_position=current_pos,
                goal=goal_pos,
                current_path=self.waypoint_manager.path,
                blocked_cells=blocked_cells,
            )
            self.waypoint_manager.set_path(new_path)

        if self.waypoint_manager.is_waypoint_reached(current_pos):
            self.waypoint_manager.advance_waypoint()

        return self.waypoint_manager.get_current_waypoint()

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_distance(a: Position, b: Optional[Position]) -> Optional[int]:
        """
        Chebyshev distance matches the project's 8-direction movement model.
        It represents the minimum number of king-moves when no obstacles exist.
        """
        if b is None:
            return None

        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _detect_progress_stall(self, current_pos: Position,
                               waypoint: Optional[Position]) -> bool:
        distance = self._grid_distance(current_pos, waypoint)

        if distance is None:
            return False

        self.distance_history.append(distance)

        if len(self.distance_history) < self.progress_window:
            return False

        recent = list(self.distance_history)

        # If the best distance in the current window was achieved earlier,
        # and the robot has not improved on that best by min_progress,
        # it is making little/no useful progress.
        best_distance = min(recent)
        current_distance = recent[-1]

        if current_distance <= best_distance - self.min_progress:
            return False

        # Also avoid triggering when the robot is already at the waypoint.
        if current_distance == 0:
            return False

        return True

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------

    def _detect_loop(self, current_pos: Position) -> bool:
        self.position_history.append(current_pos)

        if len(self.position_history) < self.loop_window:
            return False

        recent = list(self.position_history)
        newest = recent[-1]

        previous_indices = [
            i
            for i, position in enumerate(recent[:-1])
            if position == newest
        ]

        if not previous_indices:
            return False

        gap = len(recent) - 1 - previous_indices[-1]

        if gap <= 3:
            return True

        return len(set(recent)) <= 3

    # ------------------------------------------------------------------
    # A* local guidance / recovery
    # ------------------------------------------------------------------

    def _local_astar_action(
        self,
        current_pos: Position,
        target: Optional[Position],
        blocked_cells: Set[Position],
        valid_actions: Optional[Set[int]] = None,
    ) -> Optional[int]:
        """Return the first legal action of a fresh local A* path.

        If a planner/test double returns a non-adjacent [start, target] path,
        fall back to the direct 8-connected direction. This preserves the
        project's earlier recovery behavior while real A* paths still take
        priority.
        """
        if target is None or current_pos == target:
            return None

        try:
            local_path = self.planner.find_path(
                current_pos,
                target,
                blocked_cells,
            )
        except Exception:
            local_path = None

        if local_path and len(local_path) >= 2:
            next_pos = tuple(local_path[1])

            step = (
                next_pos[0] - current_pos[0],
                next_pos[1] - current_pos[1],
            )

            action = self._action_lookup.get(step)

            if action is not None:
                if valid_actions is None or action in valid_actions:
                    candidate = (
                        current_pos[0] + step[0],
                        current_pos[1] + step[1],
                    )

                    if candidate not in blocked_cells:
                        return action

        # Defensive fallback for a non-adjacent/incomplete planner result.
        dr = target[0] - current_pos[0]
        dc = target[1] - current_pos[1]

        step = (
            0 if dr == 0 else (1 if dr > 0 else -1),
            0 if dc == 0 else (1 if dc > 0 else -1),
        )

        action = self._action_lookup.get(step)

        if action is None:
            return None

        if valid_actions is not None and action not in valid_actions:
            return None

        candidate = (
            current_pos[0] + step[0],
            current_pos[1] + step[1],
        )

        if candidate in blocked_cells:
            return None

        return action

    def _begin_recovery(self, reason: str):
        self._recovery_remaining = self.recovery_steps
        self.last_recovery_reason = reason

        # Clear stale loop history so the same old A-B-A-B pattern does not
        # immediately retrigger recovery.
        self.position_history.clear()
        self.distance_history.clear()

    # ------------------------------------------------------------------
    # DDQN + A* guidance
    # ------------------------------------------------------------------

    def _guided_ddqn_action(
        self,
        state,
        valid_actions: Optional[Set[int]],
        epsilon: float = 0.0,
        astar_action: Optional[int] = None,
    ) -> int:
        """
        Select DDQN action while giving a small inference-time preference
        to the action recommended by the current local A* route.

        The DDQN network is NOT retrained and its weights are unchanged.
        """
        if valid_actions is None:
            return self.ddqn.select_action(state)

        valid_actions = sorted(valid_actions)

        if not valid_actions:
            return self.ddqn.select_action(state)

        # Preserve exploration behavior if explicitly requested.
        import random
        if random.random() < epsilon:
            return random.choice(valid_actions)

        # The real DDQNAgent exposes get_q_values(). Older test doubles and
        # lightweight agents may only expose select_action_masked(). Preserve
        # that interface instead of forcing every caller to implement the
        # Stage 3.8 inspection API.
        if not hasattr(self.ddqn, "get_q_values"):
            if hasattr(self.ddqn, "select_action_masked"):
                return self.ddqn.select_action_masked(
                    state,
                    set(valid_actions),
                    epsilon=epsilon,
                )
            return self.ddqn.select_action(state)

        q_values = self.ddqn.get_q_values(state)

        # Normalize Q-values only for comparing actions at this step.
        # This makes guidance_weight independent of the absolute Q scale.
        valid_q = [float(q_values[a]) for a in valid_actions]

        q_min = min(valid_q)
        q_max = max(valid_q)
        q_range = q_max - q_min

        scores = {}

        for action in valid_actions:
            if q_range > 1e-8:
                normalized_q = (
                    float(q_values[action]) - q_min
                ) / q_range
            else:
                normalized_q = 0.0

            scores[action] = normalized_q

        if astar_action is not None and astar_action in valid_actions:
            scores[astar_action] += self.guidance_weight

        return max(scores, key=scores.get)

    # ------------------------------------------------------------------
    # Main action selection
    # ------------------------------------------------------------------

    def get_action(
        self,
        state,
        epsilon: float = 0.0,
        current_pos: Optional[Position] = None,
        waypoint: Optional[Position] = None,
        blocked_cells: Optional[Set[Position]] = None,
        valid_actions: Optional[Iterable[int]] = None,
    ) -> int:

        blocked_cells = blocked_cells or set()

        self.last_loop_detected = False
        self.last_progress_stall = False
        self.last_action_masked = False
        self.last_astar_action = None
        self.last_recovery_reason = None

        valid_set = None

        if valid_actions is not None:
            valid_set = set(int(a) for a in valid_actions)
            self.last_valid_actions = sorted(valid_set)

            if len(valid_set) < self.ddqn.num_actions:
                self.last_action_masked = True

        # Backward-compatible behavior.
        if current_pos is None or waypoint is None:
            if valid_set is not None:
                return self.ddqn.select_action_masked(
                    state,
                    valid_set,
                    epsilon=epsilon,
                )
            return self.ddqn.select_action(state)

        # --------------------------------------------------------------
        # Calculate current A* directional recommendation.
        # --------------------------------------------------------------
        # --------------------------------------------------------------
        # Calculate A* directional guidance only when the DDQN exposes
        # Q-values. This keeps older/lightweight DDQN interfaces fully
        # backward compatible while enabling Stage 3.8 for the real agent.
        # --------------------------------------------------------------
        astar_action = None

        if hasattr(self.ddqn, "get_q_values"):
            astar_action = self._local_astar_action(
                current_pos,
                waypoint,
                blocked_cells,
                valid_actions=valid_set,
            )

        self.last_astar_action = astar_action

        # --------------------------------------------------------------
        # Detect either repeated oscillation OR progress stagnation.
        # --------------------------------------------------------------
        loop_detected = self._detect_loop(current_pos)

        progress_stalled = self._detect_progress_stall(
            current_pos,
            waypoint,
        )

        if self._recovery_remaining <= 0:
            if loop_detected:
                self.loop_recoveries += 1
                self.last_loop_detected = True
                self._begin_recovery("LOOP")

            elif progress_stalled:
                self.progress_recoveries += 1
                self.last_progress_stall = True
                self._begin_recovery("NO_PROGRESS")

        # --------------------------------------------------------------
        # Temporary recovery phase.
        # --------------------------------------------------------------
        if self._recovery_remaining > 0:
            recovery_action = self._local_astar_action(
                current_pos,
                waypoint,
                blocked_cells,
                valid_actions=valid_set,
            )

            if recovery_action is not None:
                self.last_astar_action = recovery_action
                self._recovery_remaining -= 1
                return recovery_action

            # No legal A* recovery action -> return control to DDQN.
            self._recovery_remaining = 0

        # --------------------------------------------------------------
        # Normal hybrid control:
        # DDQN remains primary, A* provides a small directional bias.
        # --------------------------------------------------------------
        return self._guided_ddqn_action(
            state,
            valid_set,
            epsilon=epsilon,
            astar_action=astar_action,
        )
