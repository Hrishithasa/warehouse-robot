import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

from network import QNetwork, NUM_ACTIONS
from replay_buffer import ReplayBuffer

# ---------------------------------------------------------
# 1. Configurable Hyperparameters
# ---------------------------------------------------------
@dataclass
class DDQNConfig:
    state_dim: int
    num_actions: int = NUM_ACTIONS
    batch_size: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    
    target_update_frequency: int = 1000
    replay_warm_up: int = 5000
    buffer_capacity: int = 50000
    
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 50000
    
    checkpoint_frequency: int = 10000
    checkpoint_dir: str = "experiments/checkpoints/"

# ---------------------------------------------------------
# 2. Checkpoint & Epsilon Managers
# ---------------------------------------------------------
class CheckpointManager:
    def __init__(self, config: DDQNConfig):
        self.dir = config.checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(self, step: int, online_net, target_net, optimizer):
        filepath = os.path.join(self.dir, f"ddqn_step_{step}.pt")
        torch.save({
            'step': step,
            'online_state': online_net.state_dict(),
            'target_state': target_net.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load(self, filepath: str, online_net, target_net, optimizer) -> int:
        checkpoint = torch.load(filepath)
        online_net.load_state_dict(checkpoint['online_state'])
        target_net.load_state_dict(checkpoint['target_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Resumed from step {checkpoint['step']}")
        return checkpoint['step']

class EpsilonScheduler:
    def __init__(self, config: DDQNConfig):
        self.config = config

    def get(self, step: int) -> float:
        if step >= self.config.epsilon_decay_steps:
            return self.config.epsilon_min
        decay = math.exp(-1. * step / self.config.epsilon_decay_steps)
        return self.config.epsilon_min + (self.config.epsilon_start - self.config.epsilon_min) * decay

# ---------------------------------------------------------
# 3. Trainable DDQN Agent
# ---------------------------------------------------------
class DDQNAgent:
    def __init__(self, config: DDQNConfig, device=None):
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.online_net = QNetwork(config.state_dim, config.num_actions).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(capacity=config.buffer_capacity)
        
        self.scheduler = EpsilonScheduler(config)
        self.checkpointer = CheckpointManager(config)
        self.current_step = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.scheduler.get(self.current_step):
            return random.randrange(self.config.num_actions)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Executes a single DDQN learning step."""
        if len(self.buffer) < self.config.replay_warm_up:
            return 
            
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # DDQN Logic
        with torch.no_grad():
            best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_actions)
            expected_q = rewards + (self.config.gamma * next_q_values * (1 - dones))
            
        current_q = self.online_net(states).gather(1, actions)
        
        loss = nn.MSELoss()(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Sync Target Network
        if self.current_step % self.config.target_update_frequency == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        # Save Checkpoint
        if self.current_step > 0 and self.current_step % self.config.checkpoint_frequency == 0:
            self.checkpointer.save(self.current_step, self.online_net, self.target_net, self.optimizer)

        self.current_step += 1
        # ---------------------------------------------------------
# ---------------------------------------------------------
# Pipeline Verification Test (Run this to verify Day 4)
# ---------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    print("--- Starting DDQN Pipeline Verification ---")
    
    # 1. Setup config with a tiny state and small batch size
    test_config = DDQNConfig(
        state_dim=25,
        batch_size=4,            # <-- FIX: Added a small batch size
        checkpoint_frequency=5,  # Force save every 5 steps
        replay_warm_up=10        # Train after 10 samples
    )
    
    # 2. Initialize Agent
    agent = DDQNAgent(test_config)
    print("Agent initialized successfully!")
    
    # 3. Inject fake experiences into the replay buffer
    print("Populating replay buffer...")
    for _ in range(15):
        dummy_state = np.random.rand(25)
        dummy_next_state = np.random.rand(25)
        agent.buffer.push(dummy_state, 0, 1.0, dummy_next_state, False)
        
    # 4. Run training steps to trigger a save at step 5
    print("Running training loop...")
    for _ in range(6):
        agent.train_step()
        
    print(f"Current step after training: {agent.current_step}")
    
    # 5. Verify Loading
    print("--- Testing Checkpoint Resumption ---")
    target_file = os.path.join(test_config.checkpoint_dir, "ddqn_step_5.pt")
    
    if os.path.exists(target_file):
        agent.checkpointer.load(
            target_file, 
            agent.online_net, 
            agent.target_net, 
            agent.optimizer
        )
        print("✅ Pipeline Verified! Day 4 Checkpoint Complete.")
    else:
        print("❌ Error: Checkpoint file was not created.")