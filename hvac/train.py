"""Train PPO (Stable-Baselines3) on the room environment and log what we report.

Each run writes into its own directory:
    spec.json        everything needed to reproduce the run
    progress.csv     SB3's own per-update diagnostics (approx_kl, clip_fraction, ...)
    eval_curve.csv   deterministic evaluation on fixed days during training
    final_eval.json  final deterministic evaluation, per day and averaged
    model.zip        the trained policy
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from hvac.env import RoomConfig, RoomThermalEnv
from hvac.evaluation import eval_seeds, evaluate_model, mean_metrics

# SB3 defaults, written out so the baseline is visible in one place. The 2048
# samples per update are collected by 8 parallel copies of the room (8 x 256).
DEFAULT_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}
N_ENVS = 8
INFO_KEYS = ("episode_energy_kwh", "episode_discomfort_kh", "episode_violation_steps")


def train_ppo(
    run_dir: str | Path,
    env_overrides: dict[str, Any] | None = None,
    ppo_overrides: dict[str, Any] | None = None,
    total_timesteps: int = 300_000,
    seed: int = 0,
    eval_every: int = 20_480,
    n_curve_days: int = 30,
    n_final_days: int = 100,
    normalize_reward: bool = False,
    diagnostics: bool = False,
) -> dict[str, float]:
    """`diagnostics=True` trains with hvac.diagnostics.DiagnosticPPO, which follows exactly
    the same trajectory but also writes diag_epochs.csv and diag_rounds.csv."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize

    if diagnostics:
        from hvac.diagnostics import DiagnosticPPO as PPO  # noqa: N814

    torch.set_num_threads(1)  # runs are parallelised across processes instead
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = RoomConfig(**(env_overrides or {}))
    ppo_kwargs = {**DEFAULT_PPO, **(ppo_overrides or {})}
    spec = {
        "env": asdict(config),
        "ppo": ppo_kwargs,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "n_envs": N_ENVS,
        "normalize_reward": normalize_reward,
    }
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2))

    vec_env = make_vec_env(
        lambda: RoomThermalEnv(config),
        n_envs=N_ENVS,
        seed=seed,
        monitor_kwargs={"info_keywords": INFO_KEYS},
    )
    if normalize_reward:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, gamma=ppo_kwargs["gamma"])
    model = PPO("MlpPolicy", vec_env, seed=seed, device="cpu", verbose=0, **ppo_kwargs)
    model.set_logger(configure(str(run_dir), ["csv"]))

    curve_seeds = eval_seeds(n_curve_days)
    curve_rows: list[dict[str, float]] = []

    def evaluate_now(timesteps: int) -> None:
        row = {"timesteps": timesteps, **mean_metrics(evaluate_model(model, config, curve_seeds))}
        buf = model.ep_info_buffer or []
        # Same quantities as experienced during training (stochastic policy, random days).
        row["train_return"] = float(np.mean([e["r"] for e in buf])) if buf else float("nan")
        row["train_energy_kwh"] = (
            float(np.mean([e["episode_energy_kwh"] for e in buf])) if buf else float("nan")
        )
        row["train_discomfort_kh"] = (
            float(np.mean([e["episode_discomfort_kh"] for e in buf])) if buf else float("nan")
        )
        row["policy_std"] = float(model.policy.log_std.exp().mean().item())
        curve_rows.append(row)

    class EvalCurve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = 0

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            if self.num_timesteps >= self.next_eval:
                evaluate_now(self.num_timesteps)
                self.next_eval += eval_every

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=EvalCurve(), log_interval=1)
    evaluate_now(model.num_timesteps)
    elapsed = time.time() - start

    with open(run_dir / "eval_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    final_rows = evaluate_model(model, config, eval_seeds(n_final_days))
    final = mean_metrics(final_rows)
    final["train_seconds"] = elapsed
    (run_dir / "final_eval.json").write_text(json.dumps({"mean": final, "per_day": final_rows}, indent=2))
    model.save(run_dir / "model")
    if diagnostics:
        model.save_diagnostics(run_dir)
    vec_env.close()
    return final


def _run_one(job: dict[str, Any]) -> tuple[str, dict[str, float]]:
    return str(job["run_dir"]), train_ppo(**job)


def run_many(jobs: list[dict[str, Any]], workers: int = 8, skip_done: bool = True) -> dict[str, dict]:
    """Run several `train_ppo(**job)` calls in parallel processes."""
    results: dict[str, dict] = {}
    todo = []
    for job in jobs:
        done = Path(job["run_dir"]) / "final_eval.json"
        if skip_done and done.exists():
            results[str(job["run_dir"])] = json.loads(done.read_text())["mean"]
        else:
            todo.append(job)
    if todo:
        with get_context("spawn").Pool(min(workers, len(todo))) as pool:
            for run_dir, final in pool.imap_unordered(_run_one, todo):
                results[run_dir] = final
                print(f"done {run_dir}: {json.dumps({k: round(v, 3) for k, v in final.items()})}", flush=True)
    return results
