"""Tune the rule-based thermostat on days that are neither training nor test days.

The thermostat in hvac.controllers has three knobs: how many hours before 8:00
it switches on, how far inside the comfort band it aims (margin), and how hard
it reacts (gain). The values first used in the report (5 h, 0.5 K, 2) were never
tuned. This script grid-searches them on 200 tuning days and reports the chosen
controller on the 100 test days that every controller and every PPO run is
evaluated on.

Usage:  python scripts/tune_thermostat.py
"""

from __future__ import annotations

import itertools

from hvac.controllers import TUNED_THERMOSTAT, ThermostatController
from hvac.env import RoomConfig
from hvac.evaluation import eval_seeds, evaluate_controller, mean_metrics

TUNE_SEEDS = [300_000 + i for i in range(200)]  # PPO trains on small seeds, test days are 100_000..
PRESTART = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
MARGIN = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
GAIN = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]


def score(ctrl: ThermostatController, cfg: RoomConfig, seeds: list[int]) -> dict[str, float]:
    m = mean_metrics(evaluate_controller(ctrl, cfg, seeds))
    m["cost"] = m["energy_kwh"] + cfg.comfort_weight * m["discomfort_kh"]
    return m


if __name__ == "__main__":
    cfg = RoomConfig()
    rows = []
    for p, m, g in itertools.product(PRESTART, MARGIN, GAIN):
        rows.append((score(ThermostatController(prestart_hours=p, margin=m, gain=g), cfg, TUNE_SEEDS)["cost"], p, m, g))
    rows.sort()
    print(f"grid: {len(rows)} settings, tuned on {len(TUNE_SEEDS)} days")
    for c, p, m, g in rows[:5]:
        print(f"  tuning-day cost {c:.3f}: prestart {p} h, margin {m} K, gain {g}")
    best = rows[0][1:]
    on_edge = [name for name, val, grid in zip(("prestart", "margin", "gain"), best, (PRESTART, MARGIN, GAIN))
               if val in (grid[0], grid[-1])]
    print("  best setting on the edge of the grid:", on_edge or "no")
    stored = (TUNED_THERMOSTAT["prestart_hours"], TUNED_THERMOSTAT["margin"], TUNED_THERMOSTAT["gain"])
    print(f"  hvac.controllers.TUNED_THERMOSTAT = {stored} -> {'matches' if stored == best else 'DIFFERS FROM'} the grid optimum")

    test = eval_seeds(100)
    print("\non the 100 test days:")
    for name, ctrl in [
        ("untuned (prestart 5 h, margin 0.5, gain 2)", ThermostatController(prestart_hours=5)),
        ("tuned   " + str(dict(zip(("prestart_hours", "margin", "gain"), best))), ThermostatController(*best[:1], margin=best[1], gain=best[2])),
    ]:
        m = score(ctrl, cfg, test)
        print(f"  {name}: energy {m['energy_kwh']:.2f} kWh, discomfort {m['discomfort_kh']:.2f} K.h, "
              f"violated {m['violation_pct']:.1f}%, cost {m['cost']:.2f}")
