"""P-D3QN agent implementation (Algorithm 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .replay_buffer import PrioritizedReplayBuffer, Transition


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv_stream = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.feature(state)
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


@dataclass
class TrainStats:
    loss: float
    td_error: float
    beta: float


class PD3QNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_capacity: int = 5000,
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 50000,
        copy_step: int = 60,
        max_grad_norm: float = 5.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.copy_step = copy_step
        self.device = torch.device(device)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.max_grad_norm = max_grad_norm

        self.eval_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_capacity)
        self.train_counter = 0
        self.epsilon_start = epsilon_start

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.action_dim))
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.eval_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def _current_beta(self) -> float:
        frac = min(1.0, self.train_counter / max(1, self.beta_frames))
        return float(self.beta_start + frac * (self.beta_end - self.beta_start))

    def train_step(self, batch_size: int = 64) -> TrainStats | None:
        if len(self.buffer) < batch_size:
            return None
        beta = self._current_beta()
        samples, indices, weights = self.buffer.sample(batch_size, beta=beta)
        states = torch.tensor(np.stack([t.state for t in samples]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in samples], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([t.reward for t in samples], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in samples]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in samples], dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q_values = self.eval_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.eval_net(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * (1 - dones) * q_next

        td_error = target - q_values
        loss = (weights_t * td_error.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        self.train_counter += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if self.train_counter % self.copy_step == 0:
            self.update_target()

        return TrainStats(loss=float(loss.item()), td_error=float(td_error.abs().mean().item()), beta=beta)
