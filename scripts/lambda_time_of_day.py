"""Where in the day does each GAE lambda setting spend energy and lose comfort?

Supports the lambda paragraph of docs/report/zh/03_ablation.md, section 3.6.
For every lambda (all 10 seeds), replays the trained policies on the 100 test
days and splits energy and discomfort by time of day. Also reports the room
temperature during the occupied hours, to show how close to the comfort band
edge each setting keeps the room.

Usage:  python scripts/lambda_time_of_day.py
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from hvac.controllers import tuned_thermostat
from hvac.env import RoomConfig, RoomThermalEnv
from hvac.evaluation import eval_seeds, run_episode

ROOT_RUNS = "runs/ablation"
CONFIGS = [("lambda_0.8", 0.8), ("lambda_0.9", 0.9), ("baseline", 0.95), ("lambda_0.99", 0.99), ("lambda_1", 1.0)]
SEEDS = range(10)


def replay(model, cfg: RoomConfig, days: list[int]) -> dict[str, np.ndarray]:
    """Per-step energy, discomfort and air temperature, shape (days, 96), batched over days."""
    envs = [RoomThermalEnv(cfg) for _ in days]
    obs = np.stack([env.reset(seed=s)[0] for env, s in zip(envs, days)])
    n = cfg.steps_per_episode
    out = {k: np.zeros((len(days), n)) for k in ("energy", "discomfort", "t_air", "t_out_mean")}
    for k in range(n):
        actions, _ = model.predict(obs, deterministic=True)
        nxt = []
        for i, env in enumerate(envs):
            o, _, _, _, info = env.step(actions[i])
            out["energy"][i, k] = info["energy_kwh"]
            out["discomfort"][i, k] = info["discomfort_kh"]
            out["t_air"][i, k] = info["t_air"]
            nxt.append(o)
        obs = np.stack(nxt)
    out["t_out_mean"] = np.array([env.t_out[:n].mean() for env in envs])
    return out


if __name__ == "__main__":
    cfg = RoomConfig()
    days = eval_seeds(100)
    hour_end = (np.arange(cfg.steps_per_episode) + 1) * cfg.dt_hours  # time at the end of each step
    occ = (hour_end > cfg.occupied_start_hour) & (hour_end <= cfg.occupied_end_hour)
    early = hour_end <= cfg.occupied_start_hour
    first2h = occ & (hour_end <= cfg.occupied_start_hour + 2)

    def split(r: dict[str, np.ndarray]) -> list[float]:
        cold, hot = r["t_out_mean"] < 10.0, r["t_out_mean"] >= 22.0
        return [
            r["energy"][:, early].sum(1).mean(),
            r["energy"][:, occ].sum(1).mean(),
            r["energy"][:, hour_end > cfg.occupied_end_hour].sum(1).mean(),
            r["discomfort"][:, first2h].sum(1).mean(),
            r["discomfort"][:, occ & ~first2h].sum(1).mean(),
            np.abs(r["t_air"][:, occ] - 22.5).mean(),
            r["discomfort"][cold].sum(1).mean(),
            r["discomfort"][hot].sum(1).mean(),
        ]

    def show(label: str, m) -> None:
        print(f"{label:>17s} | {m[0]:.2f} {m[1]:.2f} {m[2]:.2f} | {m[3]:.3f} {m[4]:.3f} | {m[5]:.2f} | {m[6]:.3f} / {m[7]:.3f}")

    print("                  | energy kWh: before 8:00, 8-18, after 18 | discomfort K.h: 8-10, 10-18 "
          "| mean |T_air - 22.5| occupied | discomfort cold days (<10 C) / hot days (>=22 C)")
    for label, ctrl in (("oracle", "oracle"), ("tuned thermostat", tuned_thermostat())):
        eps = [run_episode(RoomThermalEnv(cfg), s, oracle=True, record=True) if ctrl == "oracle" else
               run_episode(RoomThermalEnv(cfg), s, policy=lambda e, _o: ctrl.act(e), record=True) for s in days]
        r = {"energy": np.stack([e["trajectory"]["energy_kwh"] for e in eps]),
             "discomfort": np.stack([e["trajectory"]["discomfort_kh"] for e in eps]),
             "t_air": np.stack([e["trajectory"]["t_air"] for e in eps]),
             "t_out_mean": np.array([e["t_out_mean"] for e in eps])}
        show(label, split(r))
    for name, lam in CONFIGS:
        rows = []
        for s in SEEDS:
            model = PPO.load(f"{ROOT_RUNS}/{name}/seed{s}/model", device="cpu")
            rows.append(split(replay(model, cfg, days)))
        show(f"PPO lambda {lam:.2f}", np.mean(rows, 0))
