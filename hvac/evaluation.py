"""Evaluation on a fixed set of days, so every controller faces identical weather."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hvac.controllers import solve_oracle
from hvac.env import RoomConfig, RoomThermalEnv

EVAL_SEED_START = 100_000  # training uses small seeds; evaluation days never overlap
TRAJECTORY_KEYS = ("t_air", "t_mass", "t_out", "hvac_action", "energy_kwh", "discomfort_kh")


def eval_seeds(n_days: int) -> list[int]:
    return [EVAL_SEED_START + i for i in range(n_days)]


def _summarise(env: RoomThermalEnv, t_out_mean: float, rewards: list[float], last_info: dict) -> dict:
    occupied_steps = int(env.occupied[: env.cfg.steps_per_episode].sum())
    return {
        "t_out_mean": t_out_mean,
        "return": float(np.sum(rewards)),
        "energy_kwh": last_info["episode_energy_kwh"],
        "discomfort_kh": last_info["episode_discomfort_kh"],
        "violation_pct": 100.0 * last_info["episode_violation_steps"] / occupied_steps,
    }


def run_episode(
    env: RoomThermalEnv,
    seed: int,
    policy: Callable[[RoomThermalEnv, np.ndarray], float] | None = None,
    actions: np.ndarray | None = None,
    oracle: bool = False,
    record: bool = False,
) -> dict[str, Any]:
    """Play one day. Give exactly one of `policy`, `actions`, or `oracle=True`."""
    obs, info = env.reset(seed=seed)
    t_out_mean = info["t_out_mean"]
    if oracle:
        actions = solve_oracle(env)
    rewards: list[float] = []
    traj: dict[str, list[float]] = {key: [] for key in TRAJECTORY_KEYS}
    for k in range(env.cfg.steps_per_episode):
        a = actions[k] if actions is not None else policy(env, obs)
        obs, r, _, _, info = env.step(np.array([a], dtype=np.float32))
        rewards.append(r)
        if record:
            for key in TRAJECTORY_KEYS:
                traj[key].append(info[key])
    out = _summarise(env, t_out_mean, rewards, info)
    out["seed"] = seed
    if record:
        out["trajectory"] = {key: np.array(val) for key, val in traj.items()}
        out["trajectory"]["hour"] = env.hours[:-1].copy()
    return out


def evaluate_controller(controller, config: RoomConfig, seeds: list[int]) -> list[dict]:
    """Rule-based controller (has `.act(env)`), or the string "oracle"."""
    env = RoomThermalEnv(config)
    if controller == "oracle":
        return [run_episode(env, s, oracle=True) for s in seeds]
    return [run_episode(env, s, policy=lambda e, _obs: controller.act(e)) for s in seeds]


def evaluate_model(model, config: RoomConfig, seeds: list[int], deterministic: bool = True) -> list[dict]:
    """Evaluate an SB3 model on all days at once (one batched forward pass per step)."""
    envs = [RoomThermalEnv(config) for _ in seeds]
    resets = [env.reset(seed=s) for env, s in zip(envs, seeds)]
    obs = np.stack([r[0] for r in resets])
    t_means = [r[1]["t_out_mean"] for r in resets]
    rewards: list[list[float]] = [[] for _ in seeds]
    infos: list[dict] = [{} for _ in seeds]
    for _ in range(config.steps_per_episode):
        actions, _ = model.predict(obs, deterministic=deterministic)
        next_obs = []
        for i, env in enumerate(envs):
            o, r, _, _, infos[i] = env.step(actions[i])
            rewards[i].append(r)
            next_obs.append(o)
        obs = np.stack(next_obs)
    out = []
    for i, s in enumerate(seeds):
        row = _summarise(envs[i], t_means[i], rewards[i], infos[i])
        row["seed"] = s
        out.append(row)
    return out


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    keys = ("return", "energy_kwh", "discomfort_kh", "violation_pct")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
