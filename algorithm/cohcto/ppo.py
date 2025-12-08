"""PPO agent for COHCTO (Algorithm 3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(state)
        value = self.value(state).squeeze(-1)
        return logits, value


@dataclass
class Transition:
    state: np.ndarray
    action: int
    logprob: float
    reward: float
    value: float
    done: bool


class RolloutBuffer:
    def __init__(self) -> None:
        self.data: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self.data.append(transition)

    def clear(self) -> None:
        self.data.clear()


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_t = torch.from_numpy(state).float().to(self.device)
        logits, value = self.model(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), float(logprob.detach().cpu()), float(value.detach().cpu())

    def store(self, transition: Transition) -> None:
        self.buffer.add(transition)

    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, batch_size: int = 64, epochs: int = 4) -> float:
        if not self.buffer.data:
            return 0.0
        states = torch.tensor(np.stack([t.state for t in self.buffer.data]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in self.buffer.data], dtype=torch.int64, device=self.device)
        old_logprobs = torch.tensor([t.logprob for t in self.buffer.data], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t.reward for t in self.buffer.data], dtype=torch.float32, device=self.device)
        values = torch.tensor([t.value for t in self.buffer.data], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in self.buffer.data], dtype=torch.float32, device=self.device)

        advantages, returns = self._compute_advantages(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(epochs):
            idx = np.arange(len(states))
            np.random.shuffle(idx)
            for start in range(0, len(idx), batch_size):
                batch_idx = idx[start : start + batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]

                logits, value_pred = self.model(batch_states)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - batch_old_logprobs).exp()
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_adv
                policy_loss = -(torch.min(ratio * batch_adv, clipped)).mean()
                value_loss = (batch_returns - value_pred).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        self.buffer.clear()
        return total_loss
