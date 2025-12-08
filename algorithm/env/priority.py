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
    alpha: float = 0.6,
    beta: float = 0.4,
) -> Dict[Tuple[str, str], PriorityInfo]:
    """
    Closer mapping to Algorithm 2:
    - Application weight proportional to 1/deadline.
    - Task priority blends critical path urgency (alpha) and fan-out importance (beta).
    - Local deadlines allocated along longest paths then tightened by precedence.
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
        # Task structural importance approximates formulas (31)/(32): combine longest-path share and out-degree.
        task_priority: Dict[str, float] = {}
        for node in g.nodes:
            share = (longest[node] / max_path) if max_path > 0 else 0.0
            fanout = g.out_degree(node)
            task_priority[node] = app_pri_norm[app.app_id] * (alpha * share + beta * (1 + fanout))

        # Local deadline allocation proportional to longest-path share.
        raw_deadline = {
            node: app.deadline * (longest[node] / max_path) if max_path > 0 else app.deadline
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
