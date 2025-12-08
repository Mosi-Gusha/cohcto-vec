"""Prioritized experience replay buffer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.storage: List[Transition] = []
        self.priorities: List[float] = []
        self.pos = 0
        self.eps = 1e-3

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition, priority: float | None = None) -> None:
        max_prio = max(self.priorities, default=1.0)
        if priority is None:
            priority = max_prio
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
            self.priorities.append(priority)
        else:
            self.storage[self.pos] = transition
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if len(self.storage) == 0:
            raise ValueError("Buffer is empty")
        prios = np.array(self.priorities, dtype=np.float64)
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.storage), batch_size, p=probs)
        samples = [self.storage[i] for i in indices]
        total = len(self.storage)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray, eps: float = 1e-3) -> None:
        for idx, err in zip(indices, errors):
            prio = (abs(err) + eps) ** self.alpha
            self.priorities[int(idx)] = float(prio)
