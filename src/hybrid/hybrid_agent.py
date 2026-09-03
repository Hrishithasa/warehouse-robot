import numpy as np


class HybridAgent:
    """
    Hybrid Agent: combines A* (global planning) with DDQN (local control).

    UPDATED for the 14-day plan / Member 2 role:
      - Action space is 8 directions only (N, NE, E, SE, S, SW, W, NW) —
        matching src/ddqn/network.py's QNetwork output. No "wait" action
        exists in this plan's spec, so waiting-specific reward logic has
        been removed (see calculate_reward()).
      - Ownership note: per the 14-day responsibility matrix, "Adaptive
        reward" is Member 1's LEAD, Member 2's INTEGRATION. The reward
        math below is a working placeholder built for pipeline testing —
        it should be synced with / replaced by Member 1's
        src/environment/rewards.py before Day 6, so there's one source
        of truth for reward values instead of two.
      - WaypointManager is now its own class (src/hybrid/waypoint_manager.py,
        Day 6 deliverable) rather than folded into HybridAgent. This class
        still exposes should_replan() as a lightweight deviation check;
        full waypoint-advance/invalid-waypoint logic belongs in
        WaypointManager.

    blue_print() -> DDQN state | should_replan() -> A* trigger |
    calculate_reward() -> contextual reward (placeholder) | reset() -> per-episode reset
    """

    REGIME_WEIGHTS = {
        "aisle":          {"collision": -80.0, "progress": 8.0, "deviation_alpha": 0.6, "replan": -5.0, "proximity": -10.0},
        "open":           {"collision": -50.0, "progress": 5.0, "deviation_alpha": 0.3, "replan": -5.0, "proximity": -5.0},
        "dense_obstacle": {"collision": -50.0, "progress": 3.0, "deviation_alpha": 0.2, "replan": -2.0, "proximity": -20.0},
    }

    def __init__(self, total_episodes=1000, deviation_threshold=2,
                 aisle_wall_ratio=0.4, dense_obstacle_count=2):
        self.CROP_SIZE = 7
        self.HALF = self.CROP_SIZE // 2
        self.STATE_DIM = 53  # placeholder — replaced by Member 1's final state (Day 5)
        self.deviation_threshold = deviation_threshold
        self.total_episodes = total_episodes
        self.current_episode = 0
        self.aisle_wall_ratio = aisle_wall_ratio
        self.dense_obstacle_count = dense_obstacle_count
        self.steps_since_plan = 0
        self.replans_this_episode = 0

    # ---------- STATE EXTRACTION ----------
    def blue_print(self, agent_pos, goal_pos, waypoint_pos, grid):
        """
        Builds the DDQN state: local patch + waypoint/goal direction.

        NOTE: this is a temporary/local state representation for pipeline
        testing (per Day 3 guidance: "use a temporary state, don't wait
        for the final state representation"). Replace with Member 1's
        src/environment/state_augmentation.py output on Day 5.
        """
        patch = self._crop_patch(grid, agent_pos)
        direction = np.array([
            (waypoint_pos[1] - agent_pos[1]) / grid.shape[1],
            (waypoint_pos[0] - agent_pos[0]) / grid.shape[0],
            (goal_pos[1] - agent_pos[1]) / grid.shape[1],
            (goal_pos[0] - agent_pos[0]) / grid.shape[0],
        ], dtype=np.float32)
        state = np.concatenate([patch.flatten(), direction])
        assert len(state) == self.STATE_DIM, f"State has {len(state)} numbers, expected {self.STATE_DIM}!"
        return state

    def _crop_patch(self, grid, agent_pos):
        """Crops a 7x7 window around agent_pos; out-of-bounds cells count as walls."""
        row, col = agent_pos
        half, H, W = self.HALF, *grid.shape
        patch = np.ones((self.CROP_SIZE, self.CROP_SIZE), dtype=grid.dtype)
        r0, r1 = max(0, row - half), min(H, row + half + 1)
        c0, c1 = max(0, col - half), min(W, col + half + 1)
        pr0, pc0 = r0 - (row - half), c0 - (col - half)
        patch[pr0:pr0 + (r1 - r0), pc0:pc0 + (c1 - c0)] = grid[r0:r1, c0:c1]
        return patch

    @staticmethod
    def _chebyshev(a, b):
        """King-move distance, shared by every distance check in this class."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    # ---------- REPLANNING (lightweight check; full logic -> WaypointManager) ----------
    def should_replan(self, agent_pos, waypoint_pos):
        """
        True if the robot has drifted past deviation_threshold from the
        expected waypoint. This is a lightweight signal only — per the
        Day 6 plan, DynamicReplanner (Member 1, src/astar/replanner.py)
        owns the actual "is the path blocked by a dynamic obstacle"
        decision. This method covers the simpler "drifted off-path" case.
        """
        self.steps_since_plan += 1
        if self._chebyshev(agent_pos, waypoint_pos) > self.deviation_threshold:
            self.replans_this_episode += 1
            self.steps_since_plan = 0
            return True
        return False

    # ---------- CONTEXTUAL REWARD (placeholder pending Member 1's rewards.py) ----------
    def detect_context(self, patch):
        """Classifies surroundings as 'dense_obstacle' / 'aisle' / 'open' from the local patch.
        Placeholder for Member 1's ContextClassifier (src/environment/context.py, Day 3-4)."""
        if np.sum(patch == 0.5) >= self.dense_obstacle_count:
            return "dense_obstacle"
        if np.sum(patch == 1) / patch.size > self.aisle_wall_ratio:
            return "aisle"
        return "open"

    def _distance_to_nearest_obstacle(self, patch):
        """Chebyshev distance to nearest dynamic obstacle in the patch, or None if none visible."""
        cells = np.argwhere(patch == 0.5)
        if len(cells) == 0:
            return None
        return min(self._chebyshev((r, c), (self.HALF, self.HALF)) for r, c in cells)

    def calculate_reward(self, prev_pos, new_pos, goal_pos, waypoint_pos, patch,
                          collision=False, reached_goal=False, replanned=False):
        """
        Placeholder contextual reward for pipeline testing.

        IMPORTANT: per the 14-day plan, Member 1 leads the actual reward
        formula (src/environment/rewards.py); this method exists so the
        DDQN training pipeline has *something* to train against before
        that lands. Once Member 1's reward module exists, this should
        either call into it directly or be removed in favor of it —
        avoid maintaining two separate reward implementations.

        No "wait" branch: with an 8-action space (N/NE/E/SE/S/SW/W/NW),
        the robot cannot choose to stay in place, so new_pos == prev_pos
        should not occur in normal operation (only via a collision bounce,
        which is already handled by the `collision` flag).
        """
        regime = self.detect_context(patch)
        w = self.REGIME_WEIGHTS[regime]

        if collision:
            return w["collision"]
        if reached_goal:
            return 200.0

        reward = -0.01  # step penalty

        old_dist = self._chebyshev(prev_pos, waypoint_pos)
        new_dist = self._chebyshev(new_pos, waypoint_pos)
        reward += w["progress"] if new_dist < old_dist else -w["progress"]
        reward -= w["deviation_alpha"] * new_dist

        obstacle_dist = self._distance_to_nearest_obstacle(patch)
        if obstacle_dist is not None and obstacle_dist <= 2:
            reward += w["proximity"] / max(obstacle_dist, 1)

        if replanned:
            reward += w["replan"]

        return reward

    # ---------- RESET ----------
    def reset(self, episode=None):
        """Resets per-episode counters; call at the start of every episode."""
        self.steps_since_plan = 0
        self.replans_this_episode = 0
        self.current_episode = episode if episode is not None else self.current_episode + 1
