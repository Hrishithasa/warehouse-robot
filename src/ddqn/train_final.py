import numpy as np
import os
import sys


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)



from agent import DDQNAgent, DDQNConfig

# Import Member 1's finalized files
from src.environment.warehouse_env import WarehouseEnv
from src.environment.state_augmentation import StateAugmenter


def run_member_2_day_5_integration():
    print("--- Verifying Pipeline: Env -> State Builder -> DDQN -> Action -> Env ---")

    # 1. Initialize Member 1's Final Environment
    env = WarehouseEnv(config="open")
    
    # 2. Initialize Member 1's State Builder
    # Requires the static_grid generated inside WarehouseEnv
    state_builder = StateAugmenter(static_grid=env._static_grid)
    
    # Dynamically determine the final state dimension to avoid hardcoding
    # The expected output is 51 dimensions (25 + 2 + 2 + 12 + 4 + 3 + 3)
    dummy_state = state_builder.build_state(
        robot_position=(0, 0),
        goal_position=(1, 1),
        context="open",
        risk_level="low"
    )
    final_state_dim = len(dummy_state)
    print(f"Detected Final State Dimension: {final_state_dim}")

    # 3. Configure the DDQN Agent
    config = DDQNConfig(
        state_dim=final_state_dim, # Automatically adapts your QNetwork
        batch_size=32,
        replay_warm_up=100,        # Short warmup for the retraining experiment
        checkpoint_frequency=500
    )
    agent = DDQNAgent(config)
    
    # 4. Short Training Experiment (Retrain)
    num_episodes = 5
    global_step = 0
    
    print("\n--- Starting Retraining Experiment ---")
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        
        # Use Member 1's StateAugmenter to build the initial state
        state = state_builder.build_state(
            robot_position=env.robot_pos,
            goal_position=env.goal_pos,
            workers=env.workers,
            dynamic_robots=env.dynamic_robots,
            waypoint=None, 
            context=info["context"],
            risk_level=info["risk_level"]
        )
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Agent outputs 1 of the 8 Q-value mapped actions (0-7)
            action = agent.select_action(state)
            
            # Step the environment
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Build the next augmented state using the new environment data
            next_state = state_builder.build_state(
                robot_position=env.robot_pos,
                goal_position=env.goal_pos,
                workers=env.workers,
                dynamic_robots=env.dynamic_robots,
                waypoint=None,
                context=next_info["context"],
                risk_level=next_info["risk_level"]
            )
            
            # Store the transition and trigger a training step
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            global_step += 1
            steps += 1
            
        epsilon = agent.scheduler.get(agent.current_step)
        print(f"Episode {episode} | Reward: {episode_reward:7.2f} | Steps: {steps:3d} | Epsilon: {epsilon:.2f}")

    print("--- End-of-day checkpoint: DDQN successfully communicates using final state contract! ---")


if __name__ == "__main__":
    run_member_2_day_5_integration()