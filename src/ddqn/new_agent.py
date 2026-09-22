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
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim

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

        self.online_network = QNetwork(state_dim, hidden_size, num_actions).to(self.device)
        self.target_network = QNetwork(state_dim, hidden_size, num_actions).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------
    def get_q_values(self, state):
        """
        Return the ONLINE-network Q-values for one state.

        This is a read-only inference helper for visualization/analysis.
        It does not change epsilon, network weights, replay memory, or
        action-selection behavior.

        Returns:
            list[float]: one Q-value per action.
        """
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state_t)

        return q_values.squeeze(0).detach().cpu().tolist()


    # ------------------------------------------------------------------
    def select_action_masked(self, state, valid_actions, epsilon=None):
        """
        Epsilon-greedy action selection restricted to physically valid actions.

        This is an inference/control helper. Training targets, replay memory,
        network architecture, and the existing select_action() method are
        unchanged.

        If valid_actions is empty, fall back to select_action() because the
        environment has no explicit "stay" action.
        """
        valid_actions = sorted(set(int(a) for a in valid_actions))

        if not valid_actions:
            return self.select_action(state)

        eps = self.epsilon if epsilon is None else float(epsilon)

        if random.random() < eps:
            return random.choice(valid_actions)

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state_t).squeeze(0)

        masked_q_values = q_values.clone()

        invalid_actions = [
            action
            for action in range(self.num_actions)
            if action not in valid_actions
        ]

        if invalid_actions:
            masked_q_values[invalid_actions] = float("-inf")

        return int(torch.argmax(masked_q_values).item())

    # ------------------------------------------------------------------
    def get_q_values(self, state):
        """
        Return the online-network Q-values for one state.

        This is inference-only. It does not modify the network, optimizer,
        replay buffer, epsilon, or training behavior.
        """
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state_t)

        return q_values.squeeze(0).detach().cpu().numpy()
    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.online_network(state_t)

        return int(torch.argmax(q_values).item())

    # ------------------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states) \
                .gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._decay_epsilon()
        return loss.item()

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path):
        torch.save({
            "online_state_dict": self.online_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint["online_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
