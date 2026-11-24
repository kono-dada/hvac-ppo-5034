"""Pre-experiments: is the environment learnable, and which design choices matter?

Every run uses SB3's default PPO hyperparameters. Clip range, GAE lambda and the
number of epochs are deliberately NOT varied here: they are the subject of the
main ablation, whose outcome we want to predict before we look at it.

Usage:  python scripts/run_preexperiments.py [--workers 8]
Output: runs/pre/<variant>/seed<k>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hvac.train import run_many

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "pre"
SEEDS = range(5)
STEPS = 500_000

# name -> (environment overrides, extra train_ppo arguments)
VARIANTS: dict[str, tuple[dict, dict]] = {
    "baseline": ({}, {}),
    # P1: how the day ends
    "end_truncate": ({"end_of_day": "truncate"}, {}),
    # P2: price of discomfort (kWh per K*h)
    "weight_0.5": ({"comfort_weight": 0.5}, {}),
    "weight_8": ({"comfort_weight": 8.0}, {}),
    # P3: can the agent see the wall temperature?
    "no_mass_obs": ({"observe_mass": False}, {}),
    # P4: size of the reward numbers
    "reward_x0.1": ({"reward_scale": 0.1}, {}),
    "reward_vecnorm": ({}, {"normalize_reward": True}),
    # P5: is 500k steps enough?
    "baseline_long": ({}, {"total_timesteps": 1_500_000}),
}


def build_jobs() -> list[dict]:
    jobs = []
    for name, (env_overrides, extra) in VARIANTS.items():
        for seed in SEEDS:
            job = {
                "run_dir": OUT / name / f"seed{seed}",
                "env_overrides": env_overrides,
                "total_timesteps": STEPS,
                "seed": seed,
            }
            job.update(extra)
            jobs.append(job)
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_many(build_jobs(), workers=args.workers)
