"""Reference controllers that need no learning.

They give the PPO results a scale: "off" and the thermostats are what a
building would do without RL, and the LP oracle is the best any controller
could possibly do on a given day if it knew the whole day's weather in advance.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from hvac.env import RoomThermalEnv


class OffController:
    """Never runs the HVAC."""

    name = "off"

    def act(self, env: RoomThermalEnv) -> float:
        return 0.0


class ThermostatController:
    """Proportional thermostat with an optional early start.

    While active it heats when the air is below `comfort_low + margin` and cools
    when it is above `comfort_high - margin`. It is active from `prestart_hours`
    before occupancy until occupancy ends; `always_on=True` keeps it active all
    day. `prestart_hours=0` is the purely reactive thermostat.
    """

    def __init__(
        self,
        prestart_hours: float = 0.0,
        always_on: bool = False,
        margin: float = 0.5,
        gain: float = 2.0,
    ):
        self.prestart_hours = prestart_hours
        self.always_on = always_on
        self.margin = margin
        self.gain = gain
        if always_on:
            self.name = "thermostat_24h"
        else:
            self.name = f"thermostat_prestart_{prestart_hours:g}h"

    def act(self, env: RoomThermalEnv) -> float:
        cfg = env.cfg
        hour = env.hours[env.k] % 24.0
        active = self.always_on or (
            cfg.occupied_start_hour - self.prestart_hours <= hour < cfg.occupied_end_hour
        )
        if not active:
            return 0.0
        t_air = env.x[0]
        heat = self.gain * (cfg.comfort_low + self.margin - t_air)
        cool = self.gain * (t_air - (cfg.comfort_high - self.margin))
        if heat > 0.0:
            return float(min(heat, 1.0))
        if cool > 0.0:
            return float(-min(cool, 1.0))
        return 0.0


# The report's reference thermostat. Chosen by scripts/tune_thermostat.py on 200
# days that are neither training nor test days. The class defaults above
# (margin 0.5, gain 2) are the untuned values used in the pre-experiments.
TUNED_THERMOSTAT = {"prestart_hours": 2.5, "margin": 1.0, "gain": 0.75}


def tuned_thermostat() -> ThermostatController:
    ctrl = ThermostatController(**TUNED_THERMOSTAT)
    ctrl.name = "thermostat_tuned"
    return ctrl


def solve_oracle(env: RoomThermalEnv) -> np.ndarray:
    """Best possible action sequence for the day currently loaded in `env`.

    Call right after `env.reset()`. The model is linear and the reward is
    piecewise linear, so the perfect-foresight optimum is a linear program:

        minimise   sum_k  e_max*(heat_k + cool_k) + w * dt_h * slack_k
        subject to slack_k >= comfort_low  - T_air[k+1]     (occupied steps)
                   slack_k >= T_air[k+1] - comfort_high     (occupied steps)
                   0 <= heat_k, cool_k <= 1,  slack_k >= 0

    with action_k = heat_k - cool_k. This is an upper bound on performance, not
    a fair competitor: it sees the future weather, which the agent cannot.
    """
    cfg = env.cfg
    n = cfg.steps_per_episode
    a_mat, b_mat = env.A, env.B

    # T_air[k+1] = free[k] + sum_j gain[k, j] * action_j
    free = np.zeros(n)
    gain = np.zeros((n, n))
    x = env.x.copy()
    sens = np.zeros((2, n))  # d x / d action_j
    for k in range(n):
        u = np.array([env.t_out[k], 0.0, env.q_solar[k], env.q_internal[k]])
        x = a_mat @ x + b_mat @ u
        sens = a_mat @ sens
        sens[:, k] += b_mat[:, 1] * cfg.q_hvac_max
        free[k] = x[0]
        gain[k] = sens[0]

    occ = np.flatnonzero(env.occupied[:n])
    m = len(occ)
    # Variables: heat (n), cool (n), slack (m).
    cost = np.concatenate(
        [
            np.full(2 * n, cfg.energy_per_step_max_kwh),
            np.full(m, cfg.comfort_weight * cfg.dt_hours),
        ]
    )
    g = gain[occ]
    eye = np.eye(m)
    #  comfort_low - free - g(h - c) <= slack   ->  -g h + g c - slack <= free - low
    #  free + g(h - c) - comfort_high <= slack  ->   g h - g c - slack <= high - free
    a_ub = np.block([[-g, g, -eye], [g, -g, -eye]])
    b_ub = np.concatenate([free[occ] - cfg.comfort_low, cfg.comfort_high - free[occ]])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, None)] * m
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"oracle LP failed: {res.message}")
    return res.x[:n] - res.x[n : 2 * n]
