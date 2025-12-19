"""Unified CLI + trainers for COHCTO and PD3QN (shared VEC environment)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import yaml

from .env.vec import VECEnvironment
from .env.ggrn import GGRN, GraphSnapshot, SyntheticGGRNDataset
from .cohcto.ppo import PPOAgent, Transition as PPOTransition
from .pd3qn.agent import PD3QNAgent
from .pd3qn.replay_buffer import Transition as DQNTransition


# ---------------------- config helpers ---------------------- #
def load_config(algo: str, config_path: str | None = None) -> dict:
    base = Path(__file__).resolve().parent / "configs"
    name = "cohcto" if algo.startswith("cohcto") else "pd3qn"
    cfg_path = Path(config_path) if config_path else base / f"{name}.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # Allow CLI variant to force hierarchical
    if algo == "cohcto-hier":
        data["hierarchical"] = True
    return data


def pretrain_ggrn(env_cfg: dict, ggrn_cfg: dict, device: str) -> GGRN:
    num_task_types = env_cfg.get("num_task_types", 6)
    num_services = env_cfg.get("num_services", 6)
    window = ggrn_cfg.get("window", env_cfg.get("window", 3))
    samples = ggrn_cfg.get("samples", 200)
    epochs = ggrn_cfg.get("epochs", 5)
    task_to_service = list(env_cfg.get("task_to_service") or range(num_task_types))
    ggrn = GGRN(
        num_task_types=num_task_types,
        num_services=num_services,
        hidden_dim=ggrn_cfg.get("gcn_hidden_dim", 64),
        gru_hidden=ggrn_cfg.get("gru_hidden_dim", None),
    )
    dataset_path = ggrn_cfg.get("dataset_path")

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    if dataset_path:
        path = Path(dataset_path)
        if path.exists():
            loaded = torch.load(path)
            dataset = _ListDataset(loaded)
            print(f"[GGRN] loaded dataset from {path} ({len(dataset)} samples)")
        else:
            print(f"[GGRN] dataset_path {path} not found, falling back to synthetic data")
            dataset = SyntheticGGRNDataset(
                num_samples=samples,
                window=window,
                num_task_types=num_task_types,
                num_services=num_services,
                task_to_service=task_to_service,
                seed=0,
            )
    else:
        dataset = SyntheticGGRNDataset(
            num_samples=samples,
            window=window,
            num_task_types=num_task_types,
            num_services=num_services,
            task_to_service=task_to_service,
            seed=0,
        )

    validate_folds = int(ggrn_cfg.get("validate_folds", 0))
    opt = torch.optim.Adam(ggrn.parameters(), lr=ggrn_cfg.get("lr", 1e-3))
    loss_fn = torch.nn.MSELoss()
    ggrn.to(device)
    def _run_epoch(ds, tag: str = ""):
        total_loss = 0.0
        for snaps, target in ds:
            snaps = [GraphSnapshot(adj=s.adj.to(device), x=s.x.to(device)) for s in snaps]
            target = target.to(device)
            _, pred = ggrn(snaps, task_to_service)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[GGRN]{tag} loss {total_loss/len(ds):.4f}")

    for epoch in range(epochs):
        _run_epoch(dataset, tag=f" epoch {epoch+1}/{epochs}")
    if validate_folds > 1 and len(dataset) >= validate_folds:
        data_list = list(dataset)
        fold_size = len(data_list) // validate_folds
        with torch.no_grad():
            for k in range(validate_folds):
                start = k * fold_size
                end = start + fold_size
                val_slice = data_list[start:end]
                if len(val_slice) == 0:
                    continue
                total = 0.0
                for snaps, target in val_slice:
                    snaps = [GraphSnapshot(adj=s.adj.to(device), x=s.x.to(device)) for s in snaps]
                    target = target.to(device)
                    _, pred = ggrn(snaps, task_to_service)
                    total += loss_fn(pred, target).item()
                print(f"[GGRN] fold {k+1}/{validate_folds} val_loss {total/len(val_slice):.4f}")
    return ggrn


def build_env(env_cfg: dict, ggrn: GGRN) -> VECEnvironment:
    return VECEnvironment(
        ggrn=ggrn,
        num_services=env_cfg.get("num_services", 6),
        num_task_types=env_cfg.get("num_task_types", 6),
        num_vehicles=env_cfg.get("num_vehicles", 4),
        num_rsus=env_cfg.get("num_rsus", 2),
        window=env_cfg.get("window", 3),
        cache_capacity=env_cfg.get("cache_capacity", 3),
        arena_size=env_cfg.get("arena_size", 1.0),
        veh_speed=env_cfg.get("veh_speed", 0.05),
        coverage_radius=env_cfg.get("coverage_radius", 0.5),
        reward_weights=tuple(env_cfg.get("reward_weights", (2.0, 1.0, 0.1, 0.2, 1.0))),
        task_to_service=env_cfg.get("task_to_service"),
        max_steps=env_cfg.get("max_steps", 20),
        veh_speed_range_kmh=env_cfg.get("veh_speed_range_kmh"),
        vehicle_compute_range=env_cfg.get("vehicle_compute_range"),
        rsu_compute_range=env_cfg.get("rsu_compute_range"),
        task_work_range=env_cfg.get("task_work_range"),
        task_size_range=env_cfg.get("task_size_range"),
        app_deadline_range=env_cfg.get("app_deadline_range"),
        vehicle_cache_capacity=env_cfg.get("vehicle_cache_capacity"),
        rsu_cache_capacity=env_cfg.get("rsu_cache_capacity"),
    )


# ---------------------- COHCTO training ---------------------- #
def train_cohcto(cfg: dict) -> None:
    device = cfg.get("device", "cpu")
    episodes = cfg.get("episodes", 50)
    online_ggrn = cfg.get("online_ggrn", True)
    ggrn = pretrain_ggrn(cfg["env"], cfg.get("ggrn", {}), device=device)
    env = build_env(cfg["env"], ggrn)

    algo_cfg = cfg.get("algo", {})
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        gamma=algo_cfg.get("gamma", 0.95),
        lam=algo_cfg.get("lam", 0.9),
        lr=algo_cfg.get("lr", 3e-4),
        actor_lr=algo_cfg.get("actor_lr"),
        critic_lr=algo_cfg.get("critic_lr"),
        clip_eps=algo_cfg.get("clip_eps", 0.2),
        entropy_coef=algo_cfg.get("entropy_coef", 0.01),
        value_coef=algo_cfg.get("value_coef", 0.5),
    )

    base_dir = Path(__file__).resolve().parent
    log_path = Path(cfg.get("log_dir") or (base_dir / "logs" / "cohcto"))
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_file = log_path / "cohcto_metrics.csv"
    if not metrics_file.exists():
        with metrics_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "avg_delay",
                    "avg_energy",
                    "hit_rate",
                    "success_rate",
                    "reward",
                    "steps",
                    "r_cache",
                    "r_cost",
                    "r_success",
                    "r_drop",
                ]
            )

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        delays: List[float] = []
        energies: List[float] = []
        hits = 0
        cache_tasks = 0
        total_tasks = 0
        successes = 0
        r_cache = r_cost = r_success = r_drop = 0.0
        while not done:
            action, logprob, value = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(PPOTransition(state, action, logprob, reward, value, done))
            state = next_state
            ep_reward += reward
            steps += 1
            delays.append(info.get("delay", 0.0))
            energies.append(info.get("energy", 0.0))
            hits += 1 if info.get("cache_hit") else 0
            target = info.get("target")
            if target is not None and target < env.num_vehicles + env.num_rsus:
                cache_tasks += 1
            successes += 1 if info.get("success") else 0
            total_tasks += 1
            terms = info.get("reward_terms") or {}
            r_cache += terms.get("cache", 0.0)
            r_cost += terms.get("cost", 0.0)
            r_success += terms.get("success", 0.0)
            r_drop += terms.get("drop", 0.0)
        mini_batch = cfg.get("mini_batch_size", cfg.get("batch_size", 128))
        loss = agent.update(batch_size=mini_batch, epochs=5)
        avg_delay = sum(delays) / max(len(delays), 1)
        avg_energy = sum(energies) / max(len(energies), 1)
        hit_rate = hits / max(cache_tasks, 1)
        success_rate = successes / max(total_tasks, 1)
        with metrics_file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ep + 1,
                    avg_delay,
                    avg_energy,
                    hit_rate,
                    success_rate,
                    ep_reward,
                    steps,
                    r_cache / max(steps, 1),
                    r_cost / max(steps, 1),
                    r_success / max(steps, 1),
                    r_drop / max(steps, 1),
                ]
            )
        print(
            "[COHCTO] "
            f"ep {ep+1}/{episodes} reward {ep_reward:.2f} steps {steps} "
            f"delay {avg_delay:.3f} energy {avg_energy:.3f} hit {hit_rate:.2f} succ {success_rate:.2f} "
            f"loss {loss:.4f}"
        )
        if online_ggrn and (ep + 1) % 5 == 0:
            finetune_ggrn(ggrn, env.get_graph_data(), list(range(env.num_task_types)), device=device, cfg=cfg.get("ggrn", {}))
    try_plot_cohcto(metrics_file)


def finetune_ggrn(
    ggrn: GGRN,
    graph_data: Iterable[Tuple[GraphSnapshot, torch.Tensor]],
    task_to_service: Sequence[int],
    device: str,
    cfg: dict,
) -> None:
    data = list(graph_data)
    if not data:
        return
    ggrn.to(device)
    opt = torch.optim.Adam(ggrn.parameters(), lr=cfg.get("online_lr", 5e-4))
    loss_fn = torch.nn.MSELoss()
    steps = cfg.get("online_steps", 1)
    for _ in range(steps):
        total = 0.0
        for snap, target in data:
            snap = GraphSnapshot(adj=snap.adj.to(device), x=snap.x.to(device))
            target = target.to(device)
            _, pred = ggrn([snap], task_to_service)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(data)
        print(f"[GGRN-online] loss {avg:.4f}")


def try_plot_cohcto(csv_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    df.plot(x="episode", y="avg_delay", ax=axes[0, 0], title="Average Delay")
    df.plot(x="episode", y="avg_energy", ax=axes[0, 1], title="Average Energy")
    df.plot(x="episode", y="hit_rate", ax=axes[1, 0], title="Cache Hit Rate")
    df.plot(x="episode", y="success_rate", ax=axes[1, 1], title="Application Success Rate")
    plt.tight_layout()
    fig.savefig(csv_path.parent / "cohcto_metrics.png")
    plt.close(fig)


# ---------------------- PD3QN training ---------------------- #
def train_pd3qn(cfg: dict) -> None:
    device = cfg.get("device", "cpu")
    episodes = cfg.get("episodes", 200)
    batch_size = cfg.get("batch_size", 64)
    ggrn = pretrain_ggrn(cfg["env"], cfg.get("ggrn", {}), device=device)
    env = build_env(cfg["env"], ggrn)

    algo_cfg = cfg.get("algo", {})
    agent = PD3QNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        buffer_capacity=algo_cfg.get("buffer_capacity", 5000),
        gamma=algo_cfg.get("gamma", 0.99),
        lr=algo_cfg.get("lr", 1e-3),
        epsilon_start=algo_cfg.get("epsilon_start", 1.0),
        epsilon_end=algo_cfg.get("epsilon_end", 0.05),
        epsilon_decay=algo_cfg.get("epsilon_decay", 0.995),
        beta_start=algo_cfg.get("beta_start", 0.4),
        beta_end=algo_cfg.get("beta_end", 1.0),
        beta_frames=algo_cfg.get("beta_frames", 50000),
        copy_step=algo_cfg.get("copy_step", 60),
        max_grad_norm=algo_cfg.get("max_grad_norm", 5.0),
        device=device,
    )

    base_dir = Path(__file__).resolve().parent
    log_path = Path(cfg.get("log_dir") or (base_dir / "logs" / "pd3qn"))
    log_path.mkdir(parents=True, exist_ok=True)
    metrics_file = log_path / "pd3qn_vec_metrics.csv"
    if not metrics_file.exists():
        with metrics_file.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "avg_delay",
                    "avg_energy",
                    "hit_rate",
                    "success_rate",
                    "reward",
                    "steps",
                    "epsilon",
                    "beta",
                    "loss",
                    "td_error",
                    "r_cache",
                    "r_cost",
                    "r_success",
                    "r_drop",
                ]
            )

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        last_stats = None
        delays: List[float] = []
        energies: List[float] = []
        hits = 0
        cache_tasks = 0
        total_tasks = 0
        successes = 0
        done = False
        r_cache = r_cost = r_success = r_drop = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(
                DQNTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
            )
            state = next_state
            ep_reward += reward
            delays.append(info.get("delay", 0.0))
            energies.append(info.get("energy", 0.0))
            hits += 1 if info.get("cache_hit") else 0
            target = info.get("target")
            if target is not None and target < env.num_vehicles + env.num_rsus:
                cache_tasks += 1
            successes += 1 if info.get("success") else 0
            total_tasks += 1
            steps += 1
            last_stats = agent.train_step(batch_size=batch_size) or last_stats
            terms = info.get("reward_terms") or {}
            r_cache += terms.get("cache", 0.0)
            r_cost += terms.get("cost", 0.0)
            r_success += terms.get("success", 0.0)
            r_drop += terms.get("drop", 0.0)

        beta = last_stats.beta if last_stats else agent._current_beta()
        loss_val = last_stats.loss if last_stats else 0.0
        td_err = last_stats.td_error if last_stats else 0.0
        avg_delay = sum(delays) / max(len(delays), 1)
        avg_energy = sum(energies) / max(len(energies), 1)
        hit_rate = hits / max(cache_tasks, 1)
        success_rate = successes / max(total_tasks, 1)
        with metrics_file.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ep + 1,
                    avg_delay,
                    avg_energy,
                    hit_rate,
                    success_rate,
                    ep_reward,
                    steps,
                    agent.epsilon,
                    beta,
                    loss_val,
                    td_err,
                    r_cache / max(steps, 1),
                    r_cost / max(steps, 1),
                    r_success / max(steps, 1),
                    r_drop / max(steps, 1),
                ]
            )
        print(
            f"[P-D3QN:VEC] "
            f"ep {ep+1:04d}/{episodes} reward {ep_reward:.2f} steps {steps} "
            f"delay {avg_delay:.3f} energy {avg_energy:.3f} hit {hit_rate:.2f} succ {success_rate:.2f} "
            f"eps {agent.epsilon:.3f} beta {beta:.3f} loss {loss_val:.4f}"
        )
    try_plot_pd3qn(metrics_file)


def try_plot_pd3qn(csv_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    df.plot(x="episode", y="avg_delay", ax=axes[0, 0], title="Average Delay")
    df.plot(x="episode", y="avg_energy", ax=axes[0, 1], title="Average Energy")
    df.plot(x="episode", y="hit_rate", ax=axes[1, 0], title="Cache Hit Rate")
    df.plot(x="episode", y="success_rate", ax=axes[1, 1], title="Application Success Rate")
    plt.tight_layout()
    fig.savefig(csv_path.parent / f"{csv_path.stem}.png")
    plt.close(fig)


# ---------------------- CLI ---------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Unified trainer for COHCTO and PD3QN (VEC environment)")
    parser.add_argument(
        "--algo",
        type=str,
        default="cohcto",
        choices=["cohcto", "pd3qn"],
        help="Which algorithm to run",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config (optional)")
    args = parser.parse_args()

    cfg = load_config(args.algo, args.config)
    if args.algo.startswith("cohcto"):
        train_cohcto(cfg)
    else:
        train_pd3qn(cfg)


if __name__ == "__main__":
    main()
