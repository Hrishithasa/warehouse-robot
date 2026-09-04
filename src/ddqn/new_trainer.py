"""
src/ddqn/trainer.py

Member 2 — Day 3 Deliverable: Training Loop.

Implements the core reinforcement learning loop:
- Environment reset
- State acquisition (using temporary HybridAgent.blue_print for pipeline testing)
- Action selection
- Environment step (handling Gymnasium API: terminated/truncated)
- Transition storage
- Double DQN training step
- Target network updates
"""

import numpy as np
import torch

class Trainer:
    """
    Orchestrates the DDQN training process, bridging the Environment and the Agent.
    """

    def __init__(self, env, agent, hybrid_agent, config):
        """
        Args:
            env: The WarehouseEnv instance from Member 1.
            agent: The DDQNAgent instance (Member 2).
            hybrid_agent: The HybridAgent instance (Member 2) containing blue_print state builder.
            config: Hyperparameter dictionary from config.py.
        """
        self.env = env
        self.agent = agent
        self.hybrid_agent = hybrid_agent
        
        # Hyperparameters
        self.batch_size = config.get("batch_size", 64)
        self.target_update_frequency = config.get("target_update_frequency", 500)
        self.replay_warmup = config.get("replay_warmup", 1000)
        
        self.total_steps = 0
        self.episodes_completed = 0

    def get_temporary_state(self, info):
        """
        Helper method to bridge Member 1's current simple observation
        with Member 2's required state dimension for DDQN testing.
        
        This will be replaced on Day 5 when Member 1's state_augmentation.py lands.
        """
        # We need a dummy grid for the blue_print crop function.
        # We can extract the raw static grid from the environment.
        grid = getattr(self.env, "_static_grid", None)
        if grid is None:
             grid = np.zeros((self.env.grid_size, self.env.grid_size))
             
        # Use the hybrid agent to build the local state representation
        state = self.hybrid_agent.blue_print(
            agent_pos=self.env.robot_pos,
            goal_pos=self.env.goal_pos,
            waypoint_pos=self.env.goal_pos, # Dummy waypoint (direct to goal) for now
            grid=grid
        )
        return state

    def train_episode(self):
        """
        Runs a single training episode.
        
        Returns:
            dict: Metrics for the episode (reward, steps, loss, etc.)
        """
        # Gymnasium API: reset returns (observation, info)
        obs, info = self.env.reset()
        
        # Build our temporary DDQN state vector
        current_state = self.get_temporary_state(info)
        
        episode_reward = 0.0
        episode_loss = 0.0
        train_steps_this_episode = 0
        done = False
        
        while not done:
            # 1. Select Action (Epsilon-Greedy)
            action = self.agent.select_action(current_state)
            
            # 2. Environment Step (Gymnasium API)
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            done = terminated or truncated
            
            # 3. Build Next State
            next_state = self.get_temporary_state(step_info)
            
            # 4. Store Transition
            self.agent.store_transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminated # Truncated timeouts shouldn't bootstrap as true terminal states
            )
            
            # 5. Train Step (Double DQN)
            if len(self.agent.replay_buffer) > self.replay_warmup:
                loss = self.agent.train_step(self.batch_size)
                if loss is not None:
                    episode_loss += loss
                    train_steps_this_episode += 1
                    
            # 6. Target Network Sync
            self.total_steps += 1
            if self.total_steps % self.target_update_frequency == 0:
                self.agent.update_target_network()
                
            # Prepare for next loop iteration
            current_state = next_state
            episode_reward += reward
            
        self.episodes_completed += 1
        
        avg_loss = episode_loss / max(1, train_steps_this_episode)
        
        return {
            "episode": self.episodes_completed,
            "reward": episode_reward,
            "steps": self.env._step_count,
            "loss": avg_loss,
            "epsilon": self.agent.epsilon,
            "terminated": terminated,
            "truncated": truncated
        }