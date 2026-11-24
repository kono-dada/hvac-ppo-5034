"""Where does the comfort/energy conflict sit in this environment?

Uses only the perfect-foresight optimum (no learning), so the answer is about
the environment itself and not about how well PPO happens to train.

Usage:  python scripts/oracle_tradeoff.py
"""

from __future__ import annotations

import numpy as np

from hvac.controllers import OffController
from hvac.env import RoomConfig, RoomThermalEnv
from hvac.evaluation import eval_seeds, evaluate_controller, mean_metrics, run_episode

SEEDS = eval_seeds(100)


def oracle(**overrides) -> dict[str, float]:
    return mean_metrics(evaluate_controller("oracle", RoomConfig(**overrides), SEEDS))


if __name__ == "__main__":
    off = mean_metrics(evaluate_controller(OffController(), RoomConfig(), SEEDS))
    print(f"HVAC off: energy {off['energy_kwh']:.2f} kWh, discomfort {off['discomfort_kh']:.2f} K.h\n")

    print("A. Optimum as the price of discomfort changes (default room)")
    print("   weight   energy kWh   discomfort K.h   violated steps %")
    for w in (0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 8.0):
        m = oracle(comfort_weight=w)
        print(f"   {w:6.3f}   {m['energy_kwh']:10.2f}   {m['discomfort_kh']:14.2f}   {m['violation_pct']:16.1f}")

    print("\nB. Optimum at weight 2 as the heat pump gets smaller")
    print("   capacity W   energy kWh   discomfort K.h   violated steps %")
    for q in (3000.0, 2000.0, 1500.0, 1000.0, 750.0):
        m = oracle(q_hvac_max=q)
        print(f"   {q:10.0f}   {m['energy_kwh']:10.2f}   {m['discomfort_kh']:14.2f}   {m['violation_pct']:16.1f}")

    print("\nC. Optimum at weight 2 when comfort is required around the clock")
    cfg = RoomConfig(occupied_start_hour=0.0, occupied_end_hour=24.0)
    env = RoomThermalEnv(cfg)
    energy, early, late = [], [], []
    for seed in SEEDS:
        ep = run_episode(env, seed, oracle=True, record=True)
        hour = ep["trajectory"]["hour"]
        energy.append(ep["energy_kwh"])
        early.append(ep["trajectory"]["discomfort_kh"][hour < 3.0].sum())
        late.append(ep["trajectory"]["discomfort_kh"][hour >= 3.0].sum())
    print(f"   energy {np.mean(energy):.2f} kWh, discomfort {np.mean(early) + np.mean(late):.2f} K.h "
          f"(00-03h: {np.mean(early):.2f}, 03-24h: {np.mean(late):.2f})")
