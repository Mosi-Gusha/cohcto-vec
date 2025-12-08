"""Unified offloading research package.

Includes:
- COHCTO (proposed) with GGRN, topology priority, PPO / hierarchical PPO.
- P-D3QN baseline running on the shared VEC environment.
"""

# Lazy imports to avoid preloading offload.train when running `python -m offload.train`.
def train_cohcto(cfg):
    from .train import train_cohcto as _train_cohcto

    return _train_cohcto(cfg)


def train_pd3qn(cfg):
    from .train import train_pd3qn as _train_pd3qn

    return _train_pd3qn(cfg)


__all__ = ["train_cohcto", "train_pd3qn"]
