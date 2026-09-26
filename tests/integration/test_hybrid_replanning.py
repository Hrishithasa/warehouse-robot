import os
import sys

# Go up THREE levels from tests/integration/test_hybrid_replanning.py to reach the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import Member 1's A* modules 
from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner

# Import your new Waypoint Manager
from src.hybrid.waypoint_manager import WaypointManager

def run_day6_checkpoint():
    print("--- Day 6: Dynamic A* + Replanning Checkpoint ---")
    
    # 1. Setup a simple 5x5 empty grid
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    
    planner = AStarPlanner(grid)
    replanner = DynamicReplanner(planner)
    manager = WaypointManager()
    
    start = (0, 0)
    goal = (0, 4)
    blocked_cells = set()
    
    # --- STEP 1: PLAN ---
    print("\n[1] PLAN")
    initial_path = planner.find_path(start, goal, blocked_cells)
    manager.set_path(initial_path)
    print(f"Initial A* Route: {manager.path}")
    
    # Simulate robot moving to the first waypoint
    robot_position = manager.get_current_waypoint()
    print(f"Robot moves to starting waypoint: {robot_position}")
    manager.advance_waypoint() # Target the next waypoint
    
    # --- STEP 2: ENCOUNTER OBSTACLE ---
    print("\n[2] ENCOUNTER OBSTACLE")
    # A dynamic obstacle suddenly moves onto the robot's next target cell
    obstacle_pos = (0, 1)
    blocked_cells.add(obstacle_pos)
    print(f"Dynamic obstacle appears at: {obstacle_pos}")
    print(f"Robot's current target waypoint is: {manager.get_current_waypoint()}")
    
    # --- STEP 3: DETECT BLOCKED ROUTE ---
    print("\n[3] DETECT BLOCKED ROUTE")
    # Utilizing Member 1's DynamicReplanner
    is_blocked = replanner.is_path_blocked(manager.path, blocked_cells)
    is_target_invalid = manager.is_waypoint_invalid(blocked_cells)
    
    print(f"Is the overall route blocked? {is_blocked}")
    print(f"Is the immediate target waypoint invalid? {is_target_invalid}")
    
    # --- STEP 4 & 5: REPLAN & PRODUCE NEW ROUTE ---
    if is_blocked or is_target_invalid:
        print("\n[4] REPLAN")
        print("Requesting new path from DynamicReplanner...")
        
        new_path = replanner.replan(
            current_position=robot_position,
            goal=goal,
            current_path=manager.path,
            blocked_cells=blocked_cells
        )
        
        print("\n[5] PRODUCE NEW ROUTE")
        manager.set_path(new_path)
        print(f"New A* Route: {manager.path}")
        
        # Verify it successfully routed around the obstacle at (0,1)
        if obstacle_pos not in manager.path:
            print("✅ Success! The new route safely avoided the dynamic obstacle.")
    
    print("\n--- End-of-day checkpoint complete! ---")


if __name__ == "__main__":
    run_day6_checkpoint()