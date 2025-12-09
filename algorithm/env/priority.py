"""Topology priority allocation (Algorithm 2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx


@dataclass
class TaskSpec:
    """Task description inside an application DAG."""

    task_id: str
    service_id: int
    work: float  # CPU cycles or abstract cost
    size: float  # input size used to estimate transfer time
    parents: List[str]


@dataclass
class ApplicationSpec:
    app_id: str
    deadline: float
    tasks: List[TaskSpec]


@dataclass
class PriorityInfo:
    priority: float
    local_deadline: float


def _longest_path_cost(g: nx.DiGraph, duration: Dict[str, float]) -> Dict[str, float]:
    """Longest remaining cost from each node to any sink."""
    order = list(nx.topological_sort(g))
    longest: Dict[str, float] = {n: duration[n] for n in g.nodes}
    for node in reversed(order):
        child_costs = [longest[c] for c in g.successors(node)]
        if child_costs:
            longest[node] = duration[node] + max(child_costs)
    return longest


def assign_topology_priorities(
    apps: Iterable[ApplicationSpec],
    alpha_app: float = 0.4,
    alpha_work: float = 0.4,
    alpha_crit_path: float = 0.2,
) -> Dict[Tuple[str, str], PriorityInfo]:
    """
    Closer mapping to Algorithm 2 (formulas 31/32):
    - Application weight ω_aq ∝ 1/deadline (normalized).
    - Task priority blends: application weight, normalized work (ϕ), critical-path indicator Γ.
    - Local deadlines allocated by work share then tightened by priority.
    Returns:
        Mapping (app_id, task_id) -> PriorityInfo(priority, local_deadline)
    """
    apps = list(apps)
    raw_app_pri = {a.app_id: 1.0 / max(a.deadline, 1e-3) for a in apps}
    total_app_pri = sum(raw_app_pri.values())
    app_pri_norm = {k: v / total_app_pri for k, v in raw_app_pri.items()}

    result: Dict[Tuple[str, str], PriorityInfo] = {}
    for app in apps:
        g = nx.DiGraph()
        duration = {}
        for t in app.tasks:
            g.add_node(t.task_id)
            duration[t.task_id] = max(t.work, 1.0)
        for t in app.tasks:
            for parent in t.parents:
                g.add_edge(parent, t.task_id)

        if not nx.is_directed_acyclic_graph(g):
            raise ValueError(f"Application {app.app_id} graph must be a DAG")

        longest = _longest_path_cost(g, duration)
        max_path = max(longest.values())
        # Mark tasks on a critical path (Γ).
        crit_nodes: set[str] = set()
        if max_path > 0:
            for node in g.nodes:
                if abs(longest[node] - max_path) < 1e-6:
                    crit_nodes.add(node)

        work_sum = sum(duration.values())
        task_priority: Dict[str, float] = {}
        for node in g.nodes:
            work_share = duration[node] / work_sum if work_sum > 0 else 0.0
            gamma_cp = 1.0 if node in crit_nodes else 0.0
            task_priority[node] = (
                alpha_app * app_pri_norm[app.app_id]
                + alpha_work * work_share
                + alpha_crit_path * gamma_cp
            )

        # Local deadline allocation per (32): base on work share then tightened by priority.
        base_deadline = {
            node: app.deadline * (duration[node] / work_sum) if work_sum > 0 else app.deadline
            for node in g.nodes
        }
        pri_sum = sum(task_priority.values()) or 1.0
        raw_deadline = {
            node: base_deadline[node] * (1.0 - task_priority[node] / pri_sum)
            for node in g.nodes
        }
        # Enforce precedence: child deadline >= parent.
        for parent in nx.topological_sort(g):
            for child in g.successors(parent):
                raw_deadline[child] = max(raw_deadline[child], raw_deadline[parent])

        for node in g.nodes:
            result[(app.app_id, node)] = PriorityInfo(
                priority=task_priority[node],
                local_deadline=raw_deadline[node],
            )

    return result
