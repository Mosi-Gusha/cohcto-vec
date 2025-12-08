"""GGRN: Graph + GRU based service demand criticality prediction (Algorithm 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Row-normalize adjacency with self-loops."""
    eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    a_hat = adj + eye
    deg = a_hat.sum(dim=1, keepdim=True).clamp(min=1.0)
    return a_hat / deg


@dataclass
class GraphSnapshot:
    """Single DAG snapshot for a time slot."""

    adj: torch.Tensor  # [num_tasks, num_tasks]
    x: torch.Tensor  # [num_tasks, num_task_types] one-hot


class GGRN(nn.Module):
    """
    Implements Algorithm 1 from the VEC paper.

    - GCN layers extract spatial dependencies from a DAG.
    - Global pooling turns node embeddings into a graph embedding.
    - GRU models temporal evolution across a sliding window.
    - FC head outputs task-type criticality; service criticality is aggregated.
    """

    def __init__(
        self,
        num_task_types: int,
        num_services: int,
        hidden_dim: int = 64,
        gcn_layers: int = 2,
        gru_hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.num_task_types = num_task_types
        self.num_services = num_services
        self.hidden_dim = hidden_dim
        self.gcn_layers = nn.ModuleList()
        input_dim = num_task_types
        for _ in range(gcn_layers):
            layer = nn.Linear(input_dim, hidden_dim, bias=True)
            nn.init.xavier_uniform_(layer.weight)
            self.gcn_layers.append(layer)
            input_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden or hidden_dim,
            batch_first=False,
        )
        self.fc_task = nn.Linear(gru_hidden or hidden_dim, num_task_types)

    def _encode_graph(self, snap: GraphSnapshot) -> torch.Tensor:
        """Apply stacked GCN layers then global mean pooling."""
        adj = normalize_adjacency(snap.adj)
        z = snap.x
        for layer in self.gcn_layers:
            z = adj @ z
            z = layer(z)
            z = F.relu(z)
        g_emb = z.mean(dim=0)  # [hidden_dim]
        return g_emb

    def forward(
        self, snapshots: Sequence[GraphSnapshot], task_to_service: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            snapshots: ordered list of graph snapshots in a window.
            task_to_service: mapping from task type idx -> service idx.
        Returns:
            task_scores: [num_task_types] criticality per task type.
            service_scores: [num_services] normalized service criticality.
        """
        if len(snapshots) == 0:
            raise ValueError("snapshots must be non-empty")
        device = snapshots[0].adj.device
        graph_embeddings = [self._encode_graph(s) for s in snapshots]
        seq = torch.stack(graph_embeddings, dim=0)  # [T, hidden_dim]
        seq = seq.unsqueeze(1)  # [T, 1, hidden_dim]
        outputs, _ = self.gru(seq)  # [T, 1, hidden_dim]
        h = outputs[-1, 0]  # last time slot hidden state
        task_scores = F.relu(self.fc_task(h))

        service_scores = torch.zeros(self.num_services, device=device)
        for task_idx, srv_idx in enumerate(task_to_service):
            service_scores[srv_idx] += task_scores[task_idx]
        # normalize to keep magnitudes stable
        service_scores = F.softmax(service_scores, dim=0) * self.num_services
        return task_scores, service_scores


class SyntheticGGRNDataset(torch.utils.data.Dataset):
    """
    Small synthetic dataset to warm up the GGRN on random DAG sequences.
    The target is the service demand frequency in the next slot.
    """

    def __init__(
        self,
        num_samples: int,
        window: int,
        num_task_types: int,
        num_services: int,
        task_to_service: Sequence[int],
        max_nodes: int = 8,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.num_samples = num_samples
        self.window = window
        self.num_task_types = num_task_types
        self.num_services = num_services
        self.task_to_service = list(task_to_service)
        self.max_nodes = max_nodes

    def __len__(self) -> int:
        return self.num_samples

    def _random_graph(self, base_probs: torch.Tensor) -> GraphSnapshot:
        num_nodes = int(torch.randint(3, self.max_nodes + 1, (1,), generator=self.rng))
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        for i in range(num_nodes - 1):
            j = int(torch.randint(i + 1, num_nodes, (1,), generator=self.rng))
            adj[i, j] = 1.0
        node_types = torch.multinomial(base_probs, num_samples=num_nodes, replacement=True, generator=self.rng)
        x = F.one_hot(node_types, num_classes=self.num_task_types).float()
        return GraphSnapshot(adj=adj, x=x)

    def __getitem__(self, idx: int) -> Tuple[List[GraphSnapshot], torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError
        # Create a slowly drifting service demand over the window so GRU learns temporal patterns.
        base_probs = torch.ones(self.num_task_types)
        base_probs = base_probs / base_probs.sum()
        snaps: List[GraphSnapshot] = []
        for _ in range(self.window):
            snaps.append(self._random_graph(base_probs))
            noise = torch.randn_like(base_probs) * 0.05
            base_probs = torch.softmax(base_probs + noise, dim=0)
        # True demand = count of services in the last graph.
        last_types = snaps[-1].x.argmax(dim=1)
        service_counts = torch.zeros(self.num_services, dtype=torch.float32)
        for t in last_types:
            service_counts[self.task_to_service[int(t)]] += 1.0
        target = service_counts / service_counts.sum().clamp(min=1.0)
        return snaps, target
