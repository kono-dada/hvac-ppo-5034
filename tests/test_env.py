import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hvac.controllers import OffController, ThermostatController, solve_oracle
from hvac.env import RoomConfig, RoomThermalEnv, discretize
from hvac.evaluation import evaluate_controller, mean_metrics, run_episode


def test_gymnasium_api():
    check_env(RoomThermalEnv(), skip_render_check=True)


def test_sb3_api():
    from stable_baselines3.common.env_checker import check_env as sb3_check

    sb3_check(RoomThermalEnv())


def test_same_seed_same_day():
    a, b = RoomThermalEnv(), RoomThermalEnv()
    obs_a, _ = a.reset(seed=7)
    obs_b, _ = b.reset(seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(a.t_out, b.t_out)
    obs_c, _ = b.reset(seed=8)
    assert not np.array_equal(obs_a, obs_c)


@pytest.mark.parametrize("mode", ["terminate", "truncate"])
def test_episode_ends_after_one_day(mode):
    env = RoomThermalEnv(end_of_day=mode)
    env.reset(seed=0)
    for k in range(96):
        _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        last = k == 95
        assert terminated == (last and mode == "terminate")
        assert truncated == (last and mode == "truncate")
    assert "episode_energy_kwh" in info


def test_violation_tolerance_ignores_hairline_excursions():
    cfg = RoomConfig()
    oracle = evaluate_controller("oracle", cfg, list(range(900, 910)))
    assert mean_metrics(oracle)["violation_pct"] < 1.0


def test_discretisation_matches_fine_euler():
    cfg = RoomConfig()
    a_d, b_d = discretize(cfg)
    x = np.array([20.0, 18.0])
    u = np.array([5.0, 2000.0, 300.0, 400.0])
    exact = a_d @ x + b_d @ u
    y, h = x.copy(), 0.5
    for _ in range(int(cfg.dt_seconds / h)):
        d_air = (
            cfg.ua_air_out * (u[0] - y[0]) + cfg.ua_air_mass * (y[1] - y[0])
            + u[1] + cfg.solar_to_air * u[2] + u[3]
        ) / cfg.c_air
        d_mass = (
            cfg.ua_air_mass * (y[0] - y[1]) + cfg.ua_mass_out * (u[0] - y[1])
            + (1 - cfg.solar_to_air) * u[2]
        ) / cfg.c_mass
        y = y + h * np.array([d_air, d_mass])
    np.testing.assert_allclose(exact, y, atol=1e-3)


def test_no_inputs_relaxes_to_outdoor_temperature():
    cfg = RoomConfig()
    a_d, b_d = discretize(cfg)
    x = np.array([30.0, 10.0])
    for _ in range(96 * 60):
        x = a_d @ x + b_d @ np.array([12.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(x, [12.0, 12.0], atol=1e-6)


def test_heating_warms_cooling_cools():
    temps = {}
    for a in (-1.0, 0.0, 1.0):
        env = RoomThermalEnv()
        env.reset(seed=3)
        for _ in range(8):
            _, _, _, _, info = env.step(np.array([a], dtype=np.float32))
        temps[a] = info["t_air"]
    assert temps[-1.0] < temps[0.0] < temps[1.0]


def test_reward_matches_reported_metrics():
    env = RoomThermalEnv(comfort_weight=3.0, reward_scale=0.5)
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(96):
        _, r, _, _, info = env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
        total += r
    expected = -(info["episode_energy_kwh"] + 3.0 * info["episode_discomfort_kh"]) * 0.5
    assert total == pytest.approx(expected)


def test_no_discomfort_outside_occupied_hours():
    env = RoomThermalEnv()
    env.reset(seed=2)
    for k in range(96):
        _, _, _, _, info = env.step(np.array([-1.0], dtype=np.float32))  # freeze the room
        if not info["occupied"]:
            assert info["discomfort_kh"] == 0.0
    assert info["episode_discomfort_kh"] > 0.0


def test_full_power_for_a_day_uses_rated_energy():
    env = RoomThermalEnv()
    env.reset(seed=0)
    for _ in range(96):
        _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["episode_energy_kwh"] == pytest.approx(24.0 * env.cfg.p_elec_max_kw)


def test_oracle_plan_replays_and_beats_rules():
    cfg = RoomConfig()
    seeds = list(range(500, 520))
    oracle = evaluate_controller("oracle", cfg, seeds)
    for ctrl in (OffController(), ThermostatController(), ThermostatController(always_on=True)):
        rule = evaluate_controller(ctrl, cfg, seeds)
        for o, r in zip(oracle, rule):
            assert o["return"] >= r["return"] - 1e-6
    assert mean_metrics(oracle)["return"] > mean_metrics(evaluate_controller(OffController(), cfg, seeds))["return"]


def test_oracle_objective_equals_simulated_cost():
    env = RoomThermalEnv()
    env.reset(seed=11)
    plan = solve_oracle(env)
    assert np.all(np.abs(plan) <= 1.0 + 1e-9)
    replay = run_episode(RoomThermalEnv(), 11, actions=plan)
    perturbed = run_episode(RoomThermalEnv(), 11, actions=np.clip(plan + 0.05, -1, 1))
    assert replay["return"] >= perturbed["return"]
