"""
state_builder.py

Hybrid State Builder.
Extends Member 1's StateAugmenter to inject active waypoint data 
into the DDQN's observation state vector.
"""

import os
import sys

# Go up THREE levels from src/hybrid/state_builder.py to reach the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import numpy as np
from typing import Tuple, Optional

# Import Member 1's base StateAugmenter
from src.environment.state_augmentation import StateAugmenter

Position = Tuple[int, int]


class HybridStateBuilder:
    """
    Builds the augmented state vector for the hybrid agent,
    incorporating global waypoint coordinates alongside local sensor data.
    """
    def __init__(self, static_grid):
        # Initialize Member 1's underlying augmenter
        self.base_augmenter = StateAugmenter(static_grid=static_grid)

    def build_hybrid_state(
        self,
        robot_position: Position,
        goal_position: Position,
        workers: list,
        dynamic_robots: list,
        waypoint: Optional[Position],
        context: str,
        risk_level: str
    ) -> np.ndarray:
        """
        Builds the final state array, passing the active waypoint down 
        to Member 1's state builder logic.
        """
        state = self.base_augmenter.build_state(
            robot_position=robot_position,
            goal_position=goal_position,
            workers=workers,
            dynamic_robots=dynamic_robots,
            waypoint=waypoint,
            context=context,
            risk_level=risk_level
        )
        return state