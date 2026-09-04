import numpy as np
from agent import DDQNAgent, DDQNConfig

class DummyEnvironment:
    """
    A temporary environment using a simple state representation 
    to test the DDQN pipeline before final integration.
    """
    def __init__(self, state_dim=25, num_actions=8):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.current_step = 0
        self.max_steps = 50

    def reset(self):
        """Implement: reset & state acquisition"""
        self.current_step = 0
        return np.random.rand(self.state_dim)

    def step(self, action):
        """Implement: environment step"""
        self.current_step += 1
        
        # Simulate a random next state and a random reward
        next_state = np.random.rand(self.state_dim)
        reward = 1.0 if np.random.rand() > 0.5 else -0.1
        done = self.current_step >= self.max_steps
        
        return next_state, reward, done

def run_training_infrastructure():
    print("--- Starting DDQN Training Infrastructure Test ---")
    
    # 1. Setup Config & Agent
    config = DDQNConfig(
        state_dim=25, 
        replay_warm_up=100, 
        batch_size=32,
        checkpoint_frequency=200
    )
    agent = DDQNAgent(config)
    env = DummyEnvironment(state_dim=config.state_dim, num_actions=config.num_actions)
    
    num_episodes = 5
    global_step = 0
    
    # 2. The Core Training Loop
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Implement: action selection
            action = agent.select_action(state)
            
            # Implement: environment step
            next_state, reward, done = env.step(action)
            
            # Implement: transition storage
            agent.buffer.push(state, action, reward, next_state, done)
            
            # Implement: training & target updates (handled inside agent)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            global_step += 1
            
        # Implement: logging
        epsilon = agent.scheduler.get(agent.current_step)
        print(f"Episode {episode} | Reward: {episode_reward:.2f} | Total Steps: {global_step} | Epsilon: {epsilon:.2f}")

    print("--- Training Infrastructure Test Complete ---")

if __name__ == "__main__":
    run_training_infrastructure()