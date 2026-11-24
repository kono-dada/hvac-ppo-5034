"""Main ablation: clip range, GAE lambda and number of epochs, one at a time.

Every configuration changes exactly one of the three hyperparameters from the
SB3 default (clip_range 0.2, gae_lambda 0.95, n_epochs 10); everything else is
as in hvac.train.DEFAULT_PPO. The predictions for these runs are written down in
docs/report/zh/03_ablation.md and must be fixed before anything but the
baseline is run.

Usage:  python scripts/run_ablation.py [--workers 10] [--only baseline clip_0.05 ...]
        python scripts/run_ablation.py --list
Output: runs/ablation/<config>/seed<k>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hvac.train import DEFAULT_PPO, run_many

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "ablation"

SEEDS = range(5)
# 100 PPO updates of 2048 steps. On the baseline this is where the fast part of
# learning is over (see docs/report/zh/03_ablation.md, section 3.1).
STEPS = 204_800
EVAL_EVERY = 8_192  # every 4 updates -> 26 points per learning curve
EVAL_DAYS = 100  # the curve uses the same 100 test days as the final evaluation

# The default value of each hyperparameter is the shared "baseline" configuration.
GRID: dict[str, tuple[str, list[float]]] = {
    "clip_range": ("clip", [0.05, 0.1, 0.3, 0.5]),
    "gae_lambda": ("lambda", [0.8, 0.9, 0.99, 1.0]),
    "n_epochs": ("epochs", [1, 3, 30]),
}


def configs() -> dict[str, dict]:
    """name -> PPO overrides, baseline first."""
    out: dict[str, dict] = {"baseline": {}}
    for key, (short, values) in GRID.items():
        for value in values:
            out[f"{short}_{value:g}"] = {key: value}
    return out


def build_jobs(only: list[str] | None = None, seeds=SEEDS) -> list[dict]:
    jobs = []
    for name, overrides in configs().items():
        if only and name not in only:
            continue
        for seed in seeds:
            jobs.append(
                {
                    "run_dir": OUT / name / f"seed{seed}",
                    "ppo_overrides": overrides,
                    "total_timesteps": STEPS,
                    "seed": seed,
                    "eval_every": EVAL_EVERY,
                    "n_curve_days": EVAL_DAYS,
                    "n_final_days": EVAL_DAYS,
                }
            )
    # Run time grows with the number of epochs; start the long runs first so
    # that no worker is left alone with them at the end.
    jobs.sort(key=lambda j: -j["ppo_overrides"].get("n_epochs", DEFAULT_PPO["n_epochs"]))
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--only", nargs="*", help="configuration names; default: all")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="use seeds 0..N-1")
    parser.add_argument("--list", action="store_true", help="print the configurations and exit")
    args = parser.parse_args()

    unknown = set(args.only or []) - set(configs())
    if unknown:
        parser.error(f"unknown configuration(s): {sorted(unknown)}; choose from {list(configs())}")
    if args.list:
        for name, overrides in configs().items():
            print(f"{name:14s} {overrides or '(SB3 defaults)'}")
    else:
        run_many(build_jobs(args.only, range(args.seeds)), workers=args.workers)
