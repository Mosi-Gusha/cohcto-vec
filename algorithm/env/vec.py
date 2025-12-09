"""Three-layer VEC environment + COHCTO glue code."""
from __future__ import annotations

import math
import random
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Sequence, Tuple, Optional

import networkx as nx
import numpy as np
import torch

from .ggrn import GGRN, GraphSnapshot
from .priority import ApplicationSpec, TaskSpec, assign_topology_priorities


@dataclass
class TaskInstance:
    app_id: str
    task_id: str
    service_id: int
    work: float
    size: float
    deadline: float
    priority: float
    vehicle_id: int
    parents: List[str] = field(default_factory=list)
    orig_parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


class LRUCache:
    """Minimal LRU cache for services."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.store: OrderedDict[int, None] = OrderedDict()

    def has(self, service: int) -> bool:
        if service in self.store:
            self.store.move_to_end(service)
            return True
        return False

    def put(self, service: int) -> None:
        if service in self.store:
            self.store.move_to_end(service)
            return
        self.store[service] = None
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)

    def remove(self, service: int) -> None:
        if service in self.store:
            self.store.pop(service, None)

    def state_vector(self, num_services: int) -> np.ndarray:
        vec = np.zeros(num_services, dtype=np.float32)
        for s in self.store:
            vec[s] = 1.0
        return vec


class VECEnvironment:
    """
    Simplified simulator for the COHCTO algorithm (Algorithm 3).
    - Uses Algorithm 1 (GGRN) to predict service criticality from synthetic DAG history.
    - Uses Algorithm 2 to allocate task priorities/local deadlines.
    - Provides a discrete action space: choose execution node and whether to actively cache.
    """

    def __init__(
        self,
        ggrn: GGRN,
        num_services: int = 6,
        num_task_types: int = 6,
        num_vehicles: int = 4,
        num_rsus: int = 2,
        window: int = 3,
        cache_capacity: int = 3,
        seed: int = 0,
        arena_size: float = 1.0,
        veh_speed: float = 0.05,
        coverage_radius: float = 0.5,
        reward_weights: Tuple[float, float, float, float, float] = (2.0, 1.5, 1.0, 1.0, 0.2),
        task_to_service: Optional[Sequence[int]] = None,
        lambda_cache: float = 0.6,
        rsu_failure_prob: float = 0.0,
        max_queue_time: float = 5.0,
    ) -> None:
        self.rng = random.Random(seed)
        self.ggrn = ggrn
        self.num_services = num_services
        self.num_task_types = num_task_types
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.window = window
        self.cache_capacity = cache_capacity
        self.task_to_service = list(task_to_service) if task_to_service is not None else list(range(num_task_types))
        self.arena_size = arena_size
        self.veh_speed = veh_speed
        self.coverage_radius = coverage_radius
        self.vehicle_pos = np.zeros((num_vehicles, 2), dtype=np.float32)
        self.rsu_pos = np.linspace(0.2, 0.8, num_rsus).reshape(-1, 1)
        self.rsu_pos = np.hstack([self.rsu_pos, np.full((num_rsus, 1), 0.5, dtype=np.float32)])

        self.vehicle_compute = np.full(num_vehicles, 4.0)  # GHz (abstract)
        self.rsu_compute = np.full(num_rsus, 8.0)
        self.cloud_compute = 20.0
        # Reward weights map to paper's Ij/Zj/Dj style: cache hit (w1), cost penalty (w2), success (w3),
        # drop penalty and optional energy scaling for the cost term.
        self.w_cache, self.w_cost, self.w_success, self.w_drop, self.w_energy_scale = reward_weights
        self.vehicle_cache: List[LRUCache] = [LRUCache(cache_capacity) for _ in range(num_vehicles)]
        self.rsu_cache: List[LRUCache] = [LRUCache(cache_capacity) for _ in range(num_rsus)]
        self.vehicle_queue_load = np.zeros(num_vehicles, dtype=np.float32)  # accumulated compute time in queue
        self.rsu_queue_load = np.zeros(num_rsus, dtype=np.float32)
        self.cloud_queue_load = 0.0
        self.service_freq = np.zeros(num_services, dtype=np.float32)
        self.service_image = np.linspace(0.8, 2.0, num_services, dtype=np.float32)  # MB per service
        self.graph_history: Deque[GraphSnapshot] = deque(maxlen=window)
        self.graph_targets: Deque[Tuple[GraphSnapshot, torch.Tensor]] = deque(maxlen=window * 3)
        self.task_location: Dict[str, int] = {}
        self.backhaul_queue = 0.0
        self.handoff_drop_rate = 0.7
        self.lambda_cache = lambda_cache
        self.rsu_failure_prob = rsu_failure_prob
        self.rsu_alive = np.ones(num_rsus, dtype=np.float32)
        self.max_queue_time = max_queue_time

        self.apps: List[ApplicationSpec] = []
        self.tasks_by_id: Dict[str, TaskInstance] = {}
        self.pending_tasks: List[TaskInstance] = []
        self.ready_queue: List[TaskInstance] = []
        self.current_step = 0
        self.max_steps = 20
        self._service_priority = np.ones(num_services, dtype=np.float32) / num_services

        num_targets = self.num_vehicles + self.num_rsus + 1  # +1 cloud
        # Joint decision: pick offload target, and optional proactive cache RSU (0 = no cache, 1..R).
        self.action_dim = num_targets * (self.num_rsus + 1)
        self.state_dim = (
            1  # time
            + 2  # deadline ratio + task size
            + 2  # parent/child counts
            + 3  # distance to nearest RSU, coverage ratio, cloud backhaul latency proxy
            + num_services  # task service one-hot
            + num_services  # predicted criticality
            + self.num_rsus * num_services  # RSU caches
            + self.num_vehicles * num_services  # vehicle caches
            + self.num_vehicles  # vehicle queue load
            + self.num_rsus  # RSU queue load
            + 1  # lambda_cache
            + self.num_rsus  # rsu alive flags
        )

    def reset(self) -> np.ndarray:
        self.vehicle_cache = [LRUCache(self.cache_capacity) for _ in range(self.num_vehicles)]
        self.rsu_cache = [LRUCache(self.cache_capacity) for _ in range(self.num_rsus)]
        self.vehicle_queue_load[:] = 0
        self.rsu_queue_load[:] = 0
        self.cloud_queue_load = 0.0
        self.service_freq[:] = 0
        self.current_step = 0
        self.task_location.clear()
        self.backhaul_queue = 0.0
        self.rsu_alive[:] = 1.0
        # Randomize vehicle positions in arena.
        self.vehicle_pos = np.random.rand(*self.vehicle_pos.shape).astype(np.float32) * self.arena_size
        self.vehicle_pos = np.clip(self.vehicle_pos, 0.0, self.arena_size)

        self.apps, tasks, snapshots = self._sample_tasks_with_graphs()
        self.tasks_by_id = {t.task_id: t for t in tasks}
        self.pending_tasks = tasks
        self.ready_queue = [t for t in tasks if not t.parents]
        self.ready_queue.sort(key=lambda t: t.priority, reverse=True)
        self.graph_history.clear()
        self.graph_targets.clear()
        for snap, target in snapshots:
            self.graph_history.append(snap)
            self.graph_targets.append((snap, target))
        self._update_service_priority()
        return self._build_state()

    def _sample_tasks_with_graphs(self) -> Tuple[List[ApplicationSpec], List[TaskInstance], List[Tuple[GraphSnapshot, torch.Tensor]]]:
        apps: List[ApplicationSpec] = []
        num_apps = self.rng.randint(2, 4)
        snapshots: List[Tuple[GraphSnapshot, torch.Tensor]] = []
        for app_idx in range(num_apps):
            deadline = self.rng.uniform(4.0, 8.0)
            num_tasks = self.rng.randint(3, 6)
            tasks: List[TaskSpec] = []
            for t_idx in range(num_tasks):
                task_id = f"t{app_idx}_{t_idx}"
                service_id = self.rng.randrange(self.num_services)
                work = self.rng.uniform(5.0, 15.0)
                size = self.rng.uniform(0.5, 2.0)
                parents = [f"t{app_idx}_{p}" for p in range(t_idx) if self.rng.random() < 0.3]
                tasks.append(TaskSpec(task_id, service_id, work, size, parents))
            apps.append(ApplicationSpec(app_id=f"a{app_idx}", deadline=deadline, tasks=tasks))

        priorities = assign_topology_priorities(apps)
        queue: List[TaskInstance] = []
        for app in apps:
            # Build adjacency for history snapshot.
            g = nx.DiGraph()
            for t in app.tasks:
                g.add_node(t.task_id, service=t.service_id)
            for t in app.tasks:
                for p in t.parents:
                    g.add_edge(p, t.task_id)

            for task in app.tasks:
                pinfo = priorities[(app.app_id, task.task_id)]
                veh_idx = self.rng.randrange(self.num_vehicles)
                queue.append(
                    TaskInstance(
                        app_id=app.app_id,
                        task_id=task.task_id,
                        service_id=task.service_id,
                        work=task.work,
                        size=task.size,
                        deadline=pinfo.local_deadline,
                        priority=pinfo.priority,
                        vehicle_id=veh_idx,
                        parents=task.parents,
                        orig_parents=list(task.parents),
                        children=[c for c in g.successors(task.task_id)],
                    )
                )
            # Convert DAG to GraphSnapshot for GGRN history.
            node_ids = list(g.nodes)
            id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            adj = torch.zeros((len(node_ids), len(node_ids)), dtype=torch.float32)
            node_types = torch.zeros(len(node_ids), dtype=torch.int64)
            for nid in node_ids:
                node_types[id_to_idx[nid]] = g.nodes[nid]["service"]
            for u, v in g.edges:
                adj[id_to_idx[u], id_to_idx[v]] = 1.0
            x = torch.nn.functional.one_hot(node_types % self.num_task_types, num_classes=self.num_task_types).float()
            snap = GraphSnapshot(adj=adj, x=x)
            target = torch.zeros(self.num_services, dtype=torch.float32)
            for svc in node_types:
                target[int(svc)] += 1.0
            target = target / target.sum().clamp(min=1.0)
            snapshots.append((snap, target))

        # Highest priority first for initial ready queue.
        queue.sort(key=lambda t: t.priority, reverse=True)
        return apps, queue, snapshots

    def _update_service_priority(self) -> None:
        if not self.graph_history:
            return
        with torch.no_grad():
            _, srv_scores = self.ggrn(list(self.graph_history), self.task_to_service)
            self._service_priority = srv_scores.cpu().numpy()

    def _decode_action(self, action: int) -> Tuple[int, Optional[int]]:
        num_targets = self.num_vehicles + self.num_rsus + 1
        per_target = self.num_rsus + 1
        target = action // per_target
        cache_slot = action % per_target  # 0 = no cache, >0 cache at rsu_idx
        cache_rsu = cache_slot - 1 if cache_slot > 0 else None
        return target, cache_rsu

    def _cache_hit(self, service: int, target: int) -> bool:
        # targets: vehicles [0..V-1], RSUs [V..V+R-1], cloud last
        if target < self.num_vehicles:
            return self.vehicle_cache[target].has(service)
        elif target < self.num_vehicles + self.num_rsus:
            rsu_idx = target - self.num_vehicles
            if self.rsu_alive[rsu_idx] < 0.5:
                return False
            return self.rsu_cache[rsu_idx].has(service)
        return False

    def _place_cache(self, service: int, cache_rsu: Optional[int]) -> None:
        if cache_rsu is not None and 0 <= cache_rsu < self.num_rsus and self.rsu_alive[cache_rsu] >= 0.5:
            self._rsu_put_with_priority(cache_rsu, service, self._service_priority[service])

    def _rsu_put_with_priority(self, rsu_idx: int, service: int, priority: float) -> None:
        cache = self.rsu_cache[rsu_idx]
        if cache.has(service):
            return
        # Evict the lowest-priority cached service if capacity full.
        if len(cache.store) >= cache.capacity:
            if cache.store:
                # pick min priority among cached services
                min_svc = min(cache.store.keys(), key=lambda s: self._service_priority[s])
                cache.remove(min_svc)
        cache.put(service)

    def _bundle_prefetch(self, rsu_idx: int, bundle_size: int = 1) -> None:
        """Proactively cache top critical services not already cached."""
        if bundle_size <= 0 or self.rsu_alive[rsu_idx] < 0.5:
            return
        cached = set(self.rsu_cache[rsu_idx].store.keys())
        sorted_services = np.argsort(-self._service_priority)
        added = 0
        for svc in sorted_services:
            if svc in cached:
                continue
            self._rsu_put_with_priority(rsu_idx, int(svc), self._service_priority[int(svc)])
            added += 1
            if added >= bundle_size:
                break

    def _next_ready_task(self) -> Optional[TaskInstance]:
        if not self.ready_queue:
            return None
        return self.ready_queue.pop(0)

    def _build_state(self) -> np.ndarray:
        if not self.ready_queue:
            return np.zeros(self.state_dim, dtype=np.float32)
        task = self.ready_queue[0]
        time_feat = np.array([self.current_step / max(self.max_steps - 1, 1)], dtype=np.float32)
        deadline_ratio = np.array([task.deadline / 10.0], dtype=np.float32)
        size_feat = np.array([task.size / 3.0], dtype=np.float32)
        topo_feat = np.array(
            [
                len(task.parents) / max(len(self.tasks_by_id) + 1, 1),
                len(task.children) / max(len(self.tasks_by_id) + 1, 1),
            ],
            dtype=np.float32,
        )
        dist_rsu = self._nearest_rsu_distance(task.vehicle_id)
        coverage_ratio = np.array([min(1.0, dist_rsu / max(self.coverage_radius, 1e-3))], dtype=np.float32)
        backhaul = np.array([self.backhaul_queue], dtype=np.float32)
        net_feat = np.array([dist_rsu, coverage_ratio[0], backhaul[0]], dtype=np.float32)
        service_one_hot = np.zeros(self.num_services, dtype=np.float32)
        service_one_hot[task.service_id] = 1.0
        rsu_cache_vecs = [c.state_vector(self.num_services) for c in self.rsu_cache]
        rsu_state = (
            np.concatenate(rsu_cache_vecs, axis=0).astype(np.float32)
            if rsu_cache_vecs
            else np.array([], dtype=np.float32)
        )
        veh_cache_vecs = [c.state_vector(self.num_services) for c in self.vehicle_cache]
        veh_state = (
            np.concatenate(veh_cache_vecs, axis=0).astype(np.float32)
            if veh_cache_vecs
            else np.array([], dtype=np.float32)
        )
        vehicle_util = self.vehicle_queue_load / (self.vehicle_queue_load.max() + 1e-3)
        rsu_util = self.rsu_queue_load / (self.rsu_queue_load.max() + 1e-3)
        state = np.concatenate(
            [
                time_feat,
                deadline_ratio,
                size_feat,
                topo_feat,
                net_feat,
                service_one_hot,
                self._service_priority.astype(np.float32),
                rsu_state,
                veh_state,
                vehicle_util,
                rsu_util,
                np.array([self.lambda_cache], dtype=np.float32),
                self.rsu_alive.astype(np.float32),
            ]
        )
        return state

    def _nearest_rsu_distance(self, veh_idx: int) -> float:
        veh_xy = self.vehicle_pos[veh_idx]
        dist = float(np.linalg.norm(self.rsu_pos - veh_xy, axis=1).min())
        return dist

    def _link_rate(self, target: int, veh_idx: int) -> float:
        """Distance-based uplink rate (MB/s)."""
        base_bw = 12.0  # MB/s peak
        if target < self.num_vehicles:
            return float(base_bw)  # local execution, no uplink
        veh_xy = self.vehicle_pos[veh_idx]
        rsu_idx = target - self.num_vehicles if target < self.num_vehicles + self.num_rsus else np.argmin(
            np.linalg.norm(self.rsu_pos - veh_xy, axis=1)
        )
        dist = float(np.linalg.norm(self.rsu_pos[rsu_idx] - veh_xy))
        rate = base_bw * math.exp(-3.0 * dist)
        # Coverage loss when far from RSU.
        if dist > self.coverage_radius:
            rate *= 0.3
        return max(0.5, rate)

    def _handoff_drop(self, veh_idx: int, target: int) -> Tuple[bool, float]:
        """Returns (dropped, outage_delay) based on coverage."""
        if target < self.num_vehicles:
            return False, 0.0
        dist = self._nearest_rsu_distance(veh_idx)
        if dist > self.coverage_radius * 1.5:
            # High probability drop when far outside coverage.
            if self.rng.random() < self.handoff_drop_rate:
                return True, 0.2
        if dist > self.coverage_radius * 1.2:
            return False, 0.05
        return False, 0.0

    def _parent_transfer_delay(self, task: TaskInstance, target: int) -> float:
        """Data dependency transfer if parents executed elsewhere."""
        delay = 0.0
        for pid in task.orig_parents:
            src = self.task_location.get(pid)
            if src is None or src == target:
                continue
            # Cross-node transfer penalty.
            delay += 0.02 if (src < self.num_vehicles and target < self.num_vehicles + self.num_rsus) else 0.05
        return delay

    def _queue_and_delay(self, task: TaskInstance, target: int) -> Tuple[float, float, float, float, bool]:
        """Return (total_delay, energy, tx_time, deploy_delay, dropped) with queueing time included."""
        if target < self.num_vehicles:
            compute_power = self.vehicle_compute[target]
            uplink_bw = self._link_rate(target, task.vehicle_id)
            queue_wait = self.vehicle_queue_load[target]
            backhaul = 0.0
        elif target < self.num_vehicles + self.num_rsus:
            rsu_idx = target - self.num_vehicles
            compute_power = self.rsu_compute[rsu_idx]
            uplink_bw = self._link_rate(target, task.vehicle_id)
            queue_wait = self.rsu_queue_load[rsu_idx]
            backhaul = 0.0
            if self.rsu_alive[rsu_idx] < 0.5:
                return float("inf"), float("inf"), 0.0, 0.0, True
        else:
            compute_power = self.cloud_compute
            uplink_bw = self._link_rate(target, task.vehicle_id)
            queue_wait = self.cloud_queue_load
            backhaul = self.backhaul_queue
        tx_time = task.size / uplink_bw
        compute_time = task.work / compute_power
        deploy_delay = self.service_image[task.service_id] / uplink_bw if not self._cache_hit(task.service_id, target) else 0.0
        dep_delay = self._parent_transfer_delay(task, target)
        dropped, outage_delay = self._handoff_drop(task.vehicle_id, target)
        total_delay = tx_time + queue_wait + compute_time + backhaul + deploy_delay + dep_delay + outage_delay
        if queue_wait > self.max_queue_time:
            dropped = True
        energy = 0.1 * compute_time + 0.05 * task.size + 0.02 * deploy_delay
        # Update queues to reflect enqueued work.
        if target < self.num_vehicles:
            self.vehicle_queue_load[target] += compute_time
        elif target < self.num_vehicles + self.num_rsus:
            self.rsu_queue_load[target - self.num_vehicles] += compute_time
        else:
            self.cloud_queue_load += compute_time
            self.backhaul_queue += deploy_delay + tx_time
        return total_delay, energy, tx_time, deploy_delay, dropped

    def _drain_queues(self, timeslot: float = 1.0) -> None:
        """Advance time by reducing queued workloads."""
        self.vehicle_queue_load = np.maximum(0.0, self.vehicle_queue_load - timeslot)
        self.rsu_queue_load = np.maximum(0.0, self.rsu_queue_load - timeslot)
        self.cloud_queue_load = max(0.0, self.cloud_queue_load - timeslot)
        # Move vehicles randomly to emulate mobility.
        delta = (np.random.rand(*self.vehicle_pos.shape) - 0.5) * 2 * self.veh_speed
        self.vehicle_pos = np.clip(self.vehicle_pos + delta, 0.0, self.arena_size)

    def _maybe_fail_rsus(self) -> None:
        """Simulate RSU failures; failed RSU clears cache and stops service."""
        for idx in range(self.num_rsus):
            if self.rsu_alive[idx] < 0.5:
                # Allow simple recovery chance to rejoin service.
                if self.rsu_failure_prob > 0 and self.rng.random() < (self.rsu_failure_prob * 0.1):
                    self.rsu_alive[idx] = 1.0
                continue
            if self.rsu_failure_prob > 0 and self.rng.random() < self.rsu_failure_prob:
                self.rsu_alive[idx] = 0.0
                self.rsu_cache[idx] = LRUCache(self.cache_capacity)
                self.rsu_queue_load[idx] = 0.0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if not self.ready_queue:
            return self._build_state(), 0.0, True, {}
        self._maybe_fail_rsus()
        target, cache_rsu = self._decode_action(action)
        task = self._next_ready_task()
        if task is None:
            return self._build_state(), 0.0, True, {}
        self.service_freq[task.service_id] += 1

        cache_hit = self._cache_hit(task.service_id, target)
        delay, energy, tx_time, deploy_delay, dropped = self._queue_and_delay(task, target)
        success = 1.0 if (delay <= task.deadline and not dropped) else 0.0

        # Active caching at RSU based on predicted criticality and observed frequency.
        if target >= self.num_vehicles and target < self.num_vehicles + self.num_rsus:
            rsu_idx = target - self.num_vehicles
            freq_ratio = self.service_freq[task.service_id] / max(1.0, self.service_freq.sum())
            priority = self.lambda_cache * self._service_priority[task.service_id] + (1.0 - self.lambda_cache) * freq_ratio
            if priority > 0.2 and self.rsu_alive[rsu_idx] >= 0.5:
                self._rsu_put_with_priority(rsu_idx, task.service_id, priority)
                if priority > 0.6:
                    self._bundle_prefetch(rsu_idx, bundle_size=1)
                # Cooperative push to another RSU when highly critical.
                if priority > 0.5 and self.num_rsus > 1:
                    alt = 0 if rsu_idx != 0 else 1
                    if self.rsu_alive[alt] >= 0.5:
                        self._rsu_put_with_priority(alt, task.service_id, priority)

        # Proactive caching to a chosen RSU (joint decision).
        self._place_cache(task.service_id, cache_rsu)

        # Reward aligned to paper's Ij/Zj/Dj decomposition.
        cache_term = self.w_cache * (1.0 if cache_hit else 0.0)
        delay_norm = delay / max(task.deadline, 1e-3)
        cost_term = self.w_cost * (delay_norm + self.w_energy_scale * energy)
        success_term = self.w_success * success
        drop_term = self.w_drop * (1.0 if dropped else 0.0)
        reward = cache_term - cost_term + success_term - drop_term

        # Mark task completion and move children into ready queue when parents done.
        for child_id in task.children:
            child = self.tasks_by_id.get(child_id)
            if child is None:
                continue
            # Remove this parent from child's parent list to track completion.
            if task.task_id in child.parents:
                child.parents.remove(task.task_id)
            if len(child.parents) == 0 and child not in self.ready_queue:
                self.ready_queue.append(child)
        # Record execution location for data dependency transfer.
        self.task_location[task.task_id] = target
        # Resort ready queue by priority to approximate global priority scheduling.
        self.ready_queue.sort(key=lambda t: t.priority, reverse=True)
        # Remove task from pending/tasks dict.
        if task.task_id in self.tasks_by_id:
            self.tasks_by_id.pop(task.task_id)
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)

        self.current_step += 1
        self._drain_queues(timeslot=1.0)
        # Update service priority every step with a fresh snapshot of remaining tasks to keep GGRN temporal.
        self._append_current_graph_snapshot()
        self._update_service_priority()
        done = self.current_step >= self.max_steps or (not self.ready_queue and not self.pending_tasks)
        # Age backhaul queue.
        self.backhaul_queue = max(0.0, self.backhaul_queue - 0.05)
        return self._build_state(), float(reward), done, {
            "delay": delay,
            "energy": energy,
            "success": success,
            "cache_hit": cache_hit,
            "tx_time": tx_time,
            "deploy_delay": deploy_delay,
            "dropped": dropped,
            "reward_terms": {
                "cache": cache_term,
                "cost": cost_term,
                "success": success_term,
                "drop": drop_term,
            },
        }

    def get_graph_data(self) -> List[Tuple[GraphSnapshot, torch.Tensor]]:
        """Expose recent graph snapshots and targets for online GGRN fine-tuning."""
        return list(self.graph_targets)

    def _append_current_graph_snapshot(self) -> None:
        """
        Build a lightweight DAG snapshot from remaining tasks to feed GGRN temporal window.
        Target uses current service frequency (normalized) as demand proxy.
        """
        g = nx.DiGraph()
        for t in self.pending_tasks:
            g.add_node(t.task_id, service=t.service_id)
        for t in self.pending_tasks:
            for p in t.parents:
                if g.has_node(p):
                    g.add_edge(p, t.task_id)
        if not g.nodes:
            return
        node_ids = list(g.nodes)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        adj = torch.zeros((len(node_ids), len(node_ids)), dtype=torch.float32)
        node_types = torch.zeros(len(node_ids), dtype=torch.int64)
        for nid in node_ids:
            node_types[id_to_idx[nid]] = g.nodes[nid]["service"]
        for u, v in g.edges:
            adj[id_to_idx[u], id_to_idx[v]] = 1.0
        x = torch.nn.functional.one_hot(node_types % self.num_task_types, num_classes=self.num_task_types).float()
        snap = GraphSnapshot(adj=adj, x=x)
        target = torch.zeros(self.num_services, dtype=torch.float32)
        if self.service_freq.sum() > 0:
            target = torch.from_numpy(self.service_freq / self.service_freq.sum())
        else:
            for svc in node_types:
                target[int(svc)] += 1.0
            target = target / target.sum().clamp(min=1.0)
        self.graph_history.append(snap)
        self.graph_targets.append((snap, target))
        # Trim to window
        while len(self.graph_history) > self.window:
            self.graph_history.popleft()
