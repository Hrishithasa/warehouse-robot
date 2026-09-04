"""
src/ddqn/agent.py

Member 2 — Day 2 Deliverable: DDQNAgent.

Implements:
    - Online network + target network
    - select_action()          (epsilon-greedy)
    - store_transition()       (delegates to ReplayBuffer)
    - train_step()              (Double DQN update — see note below)
    - update_target_network()
    - save() / load()
    - epsilon / epsilon_decay / epsilon_min

CRITICAL — Double DQN, not vanilla DQN:
    Vanilla DQN target:   r + gamma * max_a' Q_target(s', a')
    Double DQN target:    r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

    The ONLINE network selects which action is best at the next state;
    the TARGET network only evaluates that chosen action's value. This
    decouples "which action looks best" from "how good is it", which is
    what prevents DDQN's known overestimation bias. See train_step()
    below — this distinction is implemented exactly there.
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim

# FIXED: Using relative imports for module compatibility
from .network import QNetwork, NUM_ACTIONS
from .replay_buffer import ReplayBuffer


class DDQNAgent:
    def __init__(self, state_dim, num_actions=NUM_ACTIONS, hidden_size=128,
                 lr=1e-3, gamma=0.95,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 buffer_capacity=50_000, device=None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Online network: trained every step, used for action selection.
        self.online_network = QNetwork(state_dim, hidden_size, num_actions).to(self.device)

        # Target network: a periodically-synced copy, used only to
        # produce stable targets during training (see train_step()).
        self.target_network = QNetwork(state_dim, hidden_size, num_actions).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # never trained directly, only copied into

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------
    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon: pick a random action (explore).
        Otherwise: pick argmax Q-value from the ONLINE network (exploit).

        Args:
            state: array-like, shape (state_dim,)

        Returns:
            int: action index in [0, num_actions)
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # FIXED: Added unsqueeze(0) to create a proper batch dimension of (1, state_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state_t)
        return int(torch.argmax(q_values).item())

    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        """Delegates to the replay buffer built on Day 1."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def train_step(self, batch_size=64):
        """
        One Double DQN training update, sampled from the replay buffer.

        Returns:
            float: the loss value, or None if the buffer doesn't have
                   enough transitions yet to fill a batch.
        """
        if len(self.replay_buffer) < batch_size:
            return None  # not enough experience yet — skip this step

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # --- Current Q estimate for the action actually taken ---
        q_values = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Double DQN target ---
        with torch.no_grad():
            # Step 1: ONLINE network SELECTS the best next action.
            next_actions = self.online_network(next_states).argmax(dim=1)

            # Step 2: TARGET network EVALUATES that selected action.
            next_q_values = self.target_network(next_states) \
                .gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Zero out the bootstrap term for terminal transitions.
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        # FIXED: Replaced mse_loss with smooth_l1_loss (Huber loss) for gradient stability
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._decay_epsilon()
        return loss.item()

    def _decay_epsilon(self):
        """Shrinks epsilon toward epsilon_min after every training step."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def update_target_network(self):
        """Hard update: copies online network's weights into the target network.
        Call this periodically (e.g. every N steps — see config.py, Day 4)."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    # ------------------------------------------------------------------
    def save(self, path):
        """Saves online/target weights, optimizer state, and epsilon for resuming training."""
        torch.save({
            "online_state_dict": self.online_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
        }, path)

    def load(self, path):
        """Restores a checkpoint saved by save(). Loads onto self.device."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint["online_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]