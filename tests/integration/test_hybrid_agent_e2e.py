import os
import sys

# Go up THREE levels from tests/integration/test_hybrid_agent_e2e.py to reach project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.environment.warehouse_env import WarehouseEnv
from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.ddqn.agent import DDQNAgent, DDQNConfig
from src.hybrid.waypoint_manager import WaypointManager
from src.hybrid.state_builder import HybridStateBuilder
from src.hybrid.replanning import HybridReplannerWrapper
from src.hybrid.hybrid_agent import HybridAgent


def run_hybrid_e2e_test():
    print("--- Starting Hybrid Architecture E2E Integration Test ---")

    # 1. Initialize Environment & Planners
    env = WarehouseEnv(config="open")
    obs, info = env.reset()

    # Safely convert numpy grid to list if needed
    grid_data = env._static_grid.tolist() if hasattr(env._static_grid, 'tolist') else env._static_grid
    astar_planner = AStarPlanner(grid=grid_data)
    
    replanner = DynamicReplanner(planner=astar_planner)
    waypoint_manager = WaypointManager()
    
    # 2. Setup Hybrid State Builder and DDQN Agent
    state_builder = HybridStateBuilder(static_grid=env._static_grid)
    
    # Generate a dummy state to detect exact dimension for DDQN
    dummy_state = state_builder.build_hybrid_state(
        robot_position=env.robot_pos,
        goal_position=env.goal_pos,
        workers=env.workers,
        dynamic_robots=env.dynamic_robots,
        waypoint=None,
        context=info["context"],
        risk_level=info["risk_level"]
    )
    
    ddqn_config = DDQNConfig(state_dim=len(dummy_state))
    ddqn_agent = DDQNAgent(ddqn_config)

    # 3. Initialize the Hybrid Agent Brain
    hybrid_agent = HybridAgent(
        ddqn_agent=ddqn_agent,
        waypoint_manager=waypoint_manager,
        replanner=replanner,
        astar_planner=astar_planner
    )

    # 4. Run a short simulation loop
    print("\nRunning test episode with Hybrid Agent...")
    done = False
    step_count = 0
    max_steps = 20

    while not done and step_count < max_steps:
        # Collect current environment blocked cells (dynamic obstacles)
        blocked_cells = set(env.dynamic_robots + env.workers)

        # Build current hybrid state
        current_waypoint = waypoint_manager.get_current_waypoint()
        state = state_builder.build_hybrid_state(
            robot_position=env.robot_pos,
            goal_position=env.goal_pos,
            workers=env.workers,
            dynamic_robots=env.dynamic_robots,
            waypoint=current_waypoint,
            context=info["context"],
            risk_level=info["risk_level"]
        )

        # Get action from hybrid agent
        action = hybrid_agent.get_action(
            state=state,
            current_pos=env.robot_pos,
            goal_pos=env.goal_pos,
            blocked_cells=blocked_cells
        )

        # Step environment
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        info = next_info
        step_count += 1

    print(f"Test completed successfully over {step_count} steps!")
    print("--- Hybrid Agent E2E Integration Verified! ---")


if __name__ == "__main__":
    run_hybrid_e2e_test()