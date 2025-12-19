"""Plot COHCTO and PD3QN metrics on a single figure."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare COHCTO vs PD3QN metrics in one plot")
    parser.add_argument(
        "--cohcto",
        type=str,
        default=None,
        help="Path to COHCTO metrics CSV (default: logs/cohcto/cohcto_metrics.csv)",
    )
    parser.add_argument(
        "--pd3qn",
        type=str,
        default=None,
        help="Path to PD3QN metrics CSV (default: logs/pd3qn/pd3qn_vec_metrics.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default: logs/compare_metrics.png)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cohcto_path = Path(args.cohcto) if args.cohcto else base_dir / "logs" / "cohcto" / "cohcto_metrics.csv"
    pd3qn_path = Path(args.pd3qn) if args.pd3qn else base_dir / "logs" / "pd3qn" / "pd3qn_vec_metrics.csv"
    out_path = Path(args.out) if args.out else base_dir / "logs" / "compare_metrics.png"

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover - best effort
        raise SystemExit(f"Missing plotting dependencies: {exc}") from exc

    if not cohcto_path.exists():
        raise SystemExit(f"COHCTO metrics not found: {cohcto_path}")
    if not pd3qn_path.exists():
        raise SystemExit(f"PD3QN metrics not found: {pd3qn_path}")

    df_c = pd.read_csv(cohcto_path)
    df_p = pd.read_csv(pd3qn_path)

    metrics = [
        ("avg_delay", "Average Delay"),
        ("avg_energy", "Average Energy"),
        ("hit_rate", "Cache Hit Rate"),
        ("success_rate", "Application Success Rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (col, title) in zip(axes.flatten(), metrics):
        if col not in df_c.columns or col not in df_p.columns:
            raise SystemExit(f"Missing column '{col}' in one of the CSV files")
        ax.plot(df_p["episode"], df_p[col], label="PD3QN")
        ax.plot(df_c["episode"], df_c[col], label="COHCTO")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(col)
        ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
