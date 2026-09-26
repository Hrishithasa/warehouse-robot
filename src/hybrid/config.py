"""
config.py

Hyperparameters and configuration settings for the Hybrid Architecture.
"""

class HybridConfig:
    # Grid settings
    GRID_TYPE = "open"
    
    # Pathfinding parameters
    ALLOW_DIAGONAL = True
    
    # Safety and Replanning
    REPLAN_THRESHOLD = 1  # Replan immediately if a waypoint/path cell is blocked