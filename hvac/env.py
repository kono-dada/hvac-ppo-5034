"""A single-room thermal environment for HVAC control (Gymnasium API).

Physics: a 2R2C "grey-box" model. Two temperatures describe the room:

    T_air   fast node: indoor air + furniture (what people feel)
    T_mass  slow node: walls / floor slab (stores a lot of heat)

    C_air  * dT_air /dt = UA_ao*(T_out - T_air) + UA_am*(T_mass - T_air)
                          + Q_hvac + f_air*Q_solar + Q_internal
    C_mass * dT_mass/dt = UA_am*(T_air - T_mass) + UA_mo*(T_out - T_mass)
                          + (1 - f_air)*Q_solar

One episode is one day (midnight to midnight) in 15-minute steps. Every episode
draws a new random day: anything from a cold winter day to a hot summer day.
The agent picks one number per step in [-1, 1]: negative = cooling, positive =
heating, 0 = off. The reward trades electricity use against thermal discomfort
of the occupants (only counted while the room is occupied).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.linalg import expm


@dataclass(frozen=True)
class RoomConfig:
    """All knobs of the environment, in SI units unless stated otherwise."""

    # --- time -------------------------------------------------------------
    dt_seconds: float = 900.0  # 15 min control step
    steps_per_episode: int = 96  # one day
    # How the day ends. "terminate": the task really is over at midnight, nothing
    # after it counts. "truncate": the day is cut off but "would have continued";
    # SB3 then adds its value estimate of the midnight state to the last reward.
    end_of_day: str = "terminate"

    # --- thermal model (a ~30 m2 office room) ------------------------------
    c_air: float = 1.0e6  # J/K, air + furniture
    c_mass: float = 1.0e7  # J/K, walls and floor slab
    ua_air_out: float = 30.0  # W/K, windows + infiltration
    ua_air_mass: float = 400.0  # W/K, air <-> wall surfaces
    ua_mass_out: float = 50.0  # W/K, conduction through the envelope
    solar_to_air: float = 0.3  # share of solar gain heating the air directly

    # --- HVAC (reversible heat pump) ---------------------------------------
    q_hvac_max: float = 3000.0  # W thermal, same for heating and cooling
    cop: float = 3.0  # thermal W delivered per electric W

    # --- comfort -----------------------------------------------------------
    comfort_low: float = 21.0  # degC, occupied hours (BOPTEST convention)
    comfort_high: float = 24.0  # degC
    occupied_start_hour: float = 8.0
    occupied_end_hour: float = 18.0
    violation_tolerance: float = 0.1  # K; smaller excursions are not counted as a violated step

    # --- reward: r = -(energy_kWh + comfort_weight * discomfort_Kh) * scale -
    comfort_weight: float = 2.0  # kWh of electricity one K*h of discomfort is worth
    reward_scale: float = 1.0

    # --- random day generator ----------------------------------------------
    t_out_mean_range: tuple[float, float] = (-2.0, 32.0)  # daily mean, degC
    t_out_amp_range: tuple[float, float] = (3.0, 6.0)  # day/night swing, K
    t_out_peak_hour: float = 15.0
    t_out_noise_rho: float = 0.95  # AR(1) weather noise, per step
    t_out_noise_sigma: float = 0.3  # K, innovation std per step
    q_solar_peak: float = 1000.0  # W through the windows at clear-sky noon
    clearness_range: tuple[float, float] = (0.2, 1.0)  # cloud factor per day
    q_internal_occupied: float = 400.0  # W, people + equipment
    q_internal_idle: float = 50.0  # W, standby loads

    # --- initial state at midnight -----------------------------------------
    # T_mass(0) = intercept + slope * T_out_mean + U(-spread, spread)
    # T_air(0)  = T_mass(0) + air_offset + U(-0.5, 0.5)
    # Fitted to where the room ends up at midnight when the same kind of day
    # repeats under a sensible controller (see scripts/fit_initial_state.py).
    init_mass_intercept: float = 16.3
    init_mass_slope: float = 0.33
    init_mass_spread: float = 1.0
    init_air_offset: float = -0.4

    # --- observation -------------------------------------------------------
    observe_mass: bool = True  # False hides T_mass (partially observable variant)

    @property
    def dt_hours(self) -> float:
        return self.dt_seconds / 3600.0

    @property
    def p_elec_max_kw(self) -> float:
        return self.q_hvac_max / self.cop / 1000.0

    @property
    def energy_per_step_max_kwh(self) -> float:
        return self.p_elec_max_kw * self.dt_hours


def discretize(cfg: RoomConfig) -> tuple[np.ndarray, np.ndarray]:
    """Exact zero-order-hold discretisation of the 2R2C model.

    Returns (A, B) with  x[k+1] = A @ x[k] + B @ u[k],
    x = [T_air, T_mass],  u = [T_out, Q_hvac, Q_solar, Q_internal].
    Inputs are held constant during a step, so no integration error is made.
    """
    a = np.array(
        [
            [-(cfg.ua_air_out + cfg.ua_air_mass) / cfg.c_air, cfg.ua_air_mass / cfg.c_air],
            [cfg.ua_air_mass / cfg.c_mass, -(cfg.ua_air_mass + cfg.ua_mass_out) / cfg.c_mass],
        ]
    )
    b = np.array(
        [
            [cfg.ua_air_out / cfg.c_air, 1.0 / cfg.c_air, cfg.solar_to_air / cfg.c_air, 1.0 / cfg.c_air],
            [cfg.ua_mass_out / cfg.c_mass, 0.0, (1.0 - cfg.solar_to_air) / cfg.c_mass, 0.0],
        ]
    )
    aug = np.zeros((6, 6))
    aug[:2, :2] = a
    aug[:2, 2:] = b
    m = expm(aug * cfg.dt_seconds)
    return m[:2, :2], m[:2, 2:]


class RoomThermalEnv(gym.Env):
    """One room, one heat pump, one day per episode.

    Observation (all roughly in [-1, 1]):
        0  (T_air  - 22.5) / 10
        1  (T_mass - 22.5) / 10      (omitted when observe_mass=False)
        2  (T_out  - 15) / 15
        3  Q_solar / q_solar_peak
        4  occupied flag for the coming step (0 or 1)
        5  sin(2*pi*hour/24)
        6  cos(2*pi*hour/24)

    Action: Box(-1, 1, (1,)). Q_hvac = action * q_hvac_max.

    Reward per step:
        -(energy_kWh + comfort_weight * discomfort_Kh) * reward_scale
    where discomfort_Kh = occupied * distance of T_air from the comfort band
    (in K) * step length (in h), evaluated on the temperature the step ends at.

    The episode ends after one day. By default this is a true termination
    (`end_of_day="terminate"`): the objective is the cost of this one day, and the
    time of day is part of the observation, so the agent can know the end is near.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: RoomConfig | None = None, **overrides: Any):
        super().__init__()
        self.cfg = replace(config or RoomConfig(), **overrides)
        if self.cfg.end_of_day not in ("terminate", "truncate"):
            raise ValueError(f"end_of_day must be 'terminate' or 'truncate', got {self.cfg.end_of_day!r}")
        self.A, self.B = discretize(self.cfg)

        n_obs = 7 if self.cfg.observe_mass else 6
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        n = self.cfg.steps_per_episode
        self.hours = np.arange(n + 1) * self.cfg.dt_hours
        hour_of_day = self.hours % 24.0
        self.occupied = (hour_of_day >= self.cfg.occupied_start_hour) & (
            hour_of_day < self.cfg.occupied_end_hour
        )
        self.q_internal = np.where(
            self.occupied, self.cfg.q_internal_occupied, self.cfg.q_internal_idle
        )

        # Filled in by reset().
        self.t_out = np.zeros(n + 1)
        self.q_solar = np.zeros(n + 1)
        self.x = np.zeros(2)
        self.x0 = np.zeros(2)
        self.k = 0
        self._totals: dict[str, float] = {}

    # ------------------------------------------------------------------ day
    def _sample_day(self) -> float:
        """Draw the weather of one day. Returns the daily mean outdoor temperature."""
        cfg, rng = self.cfg, self.np_random
        n = cfg.steps_per_episode

        t_mean = rng.uniform(*cfg.t_out_mean_range)
        amp = rng.uniform(*cfg.t_out_amp_range)
        noise = np.zeros(n + 1)
        stationary_std = cfg.t_out_noise_sigma / np.sqrt(1.0 - cfg.t_out_noise_rho**2)
        noise[0] = rng.normal(0.0, stationary_std)
        shocks = rng.normal(0.0, cfg.t_out_noise_sigma, size=n)
        for i in range(n):
            noise[i + 1] = cfg.t_out_noise_rho * noise[i] + shocks[i]
        phase = 2.0 * np.pi * (self.hours - cfg.t_out_peak_hour) / 24.0
        self.t_out = t_mean + amp * np.cos(phase) + noise

        clearness = rng.uniform(*cfg.clearness_range)
        sun = np.clip(np.sin(np.pi * ((self.hours % 24.0) - 6.0) / 12.0), 0.0, None)
        self.q_solar = cfg.q_solar_peak * clearness * sun
        return float(t_mean)

    # ------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg, rng = self.cfg, self.np_random

        t_mean = self._sample_day()
        t_mass = (
            cfg.init_mass_intercept
            + cfg.init_mass_slope * t_mean
            + rng.uniform(-cfg.init_mass_spread, cfg.init_mass_spread)
        )
        t_air = t_mass + cfg.init_air_offset + rng.uniform(-0.5, 0.5)
        self.x = np.array([t_air, t_mass], dtype=np.float64)
        self.x0 = self.x.copy()
        self.k = 0
        self._totals = {"energy_kwh": 0.0, "discomfort_kh": 0.0, "violation_steps": 0.0}
        return self._obs(), {"t_out_mean": t_mean}

    def step(self, action):
        cfg = self.cfg
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        k = self.k

        u = np.array([self.t_out[k], a * cfg.q_hvac_max, self.q_solar[k], self.q_internal[k]])
        self.x = self.A @ self.x + self.B @ u
        t_air = self.x[0]

        energy_kwh = abs(a) * cfg.energy_per_step_max_kwh
        deviation = max(cfg.comfort_low - t_air, t_air - cfg.comfort_high, 0.0)
        occupied = bool(self.occupied[k])
        discomfort_kh = deviation * cfg.dt_hours if occupied else 0.0
        violated = occupied and deviation > cfg.violation_tolerance
        reward = -(energy_kwh + cfg.comfort_weight * discomfort_kh) * cfg.reward_scale

        self._totals["energy_kwh"] += energy_kwh
        self._totals["discomfort_kh"] += discomfort_kh
        self._totals["violation_steps"] += float(violated)

        self.k += 1
        done = self.k >= cfg.steps_per_episode
        terminated = done and cfg.end_of_day == "terminate"
        truncated = done and cfg.end_of_day == "truncate"
        info = {
            "t_air": float(t_air),
            "t_mass": float(self.x[1]),
            "t_out": float(self.t_out[k]),
            "hvac_action": a,
            "occupied": occupied,
            "energy_kwh": energy_kwh,
            "discomfort_kh": discomfort_kh,
            "violated": violated,
        }
        if done:
            info["episode_energy_kwh"] = self._totals["energy_kwh"]
            info["episode_discomfort_kh"] = self._totals["discomfort_kh"]
            info["episode_violation_steps"] = self._totals["violation_steps"]
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self) -> np.ndarray:
        cfg, k = self.cfg, self.k
        hour = self.hours[k] % 24.0
        angle = 2.0 * np.pi * hour / 24.0
        obs = [(self.x[0] - 22.5) / 10.0]
        if cfg.observe_mass:
            obs.append((self.x[1] - 22.5) / 10.0)
        obs += [
            (self.t_out[k] - 15.0) / 15.0,
            self.q_solar[k] / cfg.q_solar_peak,
            float(self.occupied[k]),
            np.sin(angle),
            np.cos(angle),
        ]
        return np.asarray(obs, dtype=np.float32)
