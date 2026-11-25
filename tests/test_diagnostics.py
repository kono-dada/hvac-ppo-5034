import numpy as np
import pytest
import torch as th

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from hvac.diagnostics import DiagnosticPPO  # noqa: E402
from hvac.env import RoomConfig, RoomThermalEnv  # noqa: E402
from hvac.train import DEFAULT_PPO  # noqa: E402


def _train(cls, clip_range=0.2, n_epochs=3, rounds=3, seed=7, log_dir=None):
    th.set_num_threads(1)
    cfg = RoomConfig()
    env = make_vec_env(lambda: RoomThermalEnv(cfg), n_envs=2, seed=seed)
    kwargs = {**DEFAULT_PPO, "n_steps": 128, "clip_range": clip_range, "n_epochs": n_epochs}
    model = cls("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **kwargs)
    if log_dir is not None:
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(str(log_dir), ["csv"]))
    model.learn(total_timesteps=rounds * 256)
    env.close()
    return model


@pytest.mark.parametrize("clip_range,n_epochs", [(0.05, 3), (0.5, 3), (0.2, 1)])
def test_diagnostics_do_not_change_training(clip_range, n_epochs, tmp_path):
    import pandas as pd
    from scripts.analyze_clip_diagnostics import _same_nested

    plain = _train(PPO, clip_range, n_epochs, log_dir=tmp_path / "plain")
    numpy_rng, torch_rng = np.random.get_state(), th.get_rng_state().clone()
    diag = _train(DiagnosticPPO, clip_range, n_epochs, log_dir=tmp_path / "diag")
    assert _same_nested(plain.policy.state_dict(), diag.policy.state_dict())
    assert _same_nested(plain.policy.optimizer.state_dict(), diag.policy.optimizer.state_dict())
    assert th.equal(torch_rng, th.get_rng_state())
    numpy_diag = np.random.get_state()
    assert numpy_rng[0] == numpy_diag[0] and np.array_equal(numpy_rng[1], numpy_diag[1])
    assert numpy_rng[2:] == numpy_diag[2:]
    a, b = (pd.read_csv(tmp_path / p / "progress.csv", dtype=str, keep_default_na=False) for p in ("plain", "diag"))
    cols = sorted(set(a.columns) - {"time/fps", "time/time_elapsed"})
    assert a[cols].equals(b[cols])
    assert len(diag.diag_rounds) == 3
    assert len(diag.diag_epochs) == 3 * n_epochs


def test_gradient_split_and_counts():
    diag = _train(DiagnosticPPO, clip_range=0.05, n_epochs=5)
    rows = diag.diag_epochs
    # g_c = g_u - g_z up to float error
    assert max(r["identity_residual_max"] for r in rows) < 1e-4
    for r in rows:
        # zeroed + against-the-advantage = everything SB3 calls clipped (up to A == 0 exactly)
        assert (r["zeroed_frac"] + r["against_frac"] + r["boundary_roundoff_frac"]
                + r["surrogate_tie_frac"]) == pytest.approx(r["clip_fraction"], abs=1e-6)
        hist = [r[k] for k in ("u_lt_m1", "u_m1_0", "u_0_05", "u_05_08", "u_08_1", "u_gt_1")]
        assert sum(hist) == pytest.approx(1.0)
        assert r["kl_exact"] >= 0
    # the first minibatch of a round has ratio 1, so a small clip range must clip something later on
    assert max(r["zeroed_frac"] for r in rows) > 0
    first = [r for r in rows if r["epoch"] == 1]
    assert all(np.isfinite(r["consistency_c"]) for r in first)


def test_mean_net_matches_policy_mean():
    diag = _train(DiagnosticPPO, rounds=1)
    obs = th.randn(16, diag.observation_space.shape[0])
    with th.no_grad():
        expected = diag.policy.get_distribution(obs).distribution.mean.flatten()
        assert th.allclose(diag._mean_net()(obs).flatten(), expected)


def test_across_round_measures():
    diag = _train(DiagnosticPPO, n_epochs=1, rounds=11)
    rounds = diag.diag_rounds
    assert np.isnan(rounds[0]["cos_prev_round_displacement"])
    assert all(np.isnan(r["param_efficiency_10"]) for r in rounds[:9])
    for r in rounds[9:]:
        assert 0 < r["param_efficiency_10"] <= 1 + 1e-6  # triangle inequality
        # not bounded by 1: the per-round shifts are measured on each round's own observations
        assert 0 < r["mu_efficiency_10"] < np.inf


def test_rejects_wrong_sb3_version(monkeypatch):
    import stable_baselines3
    monkeypatch.setattr(stable_baselines3, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="copies PPO.train"):
        DiagnosticPPO("MlpPolicy", None)


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.2, 0.3, 0.5])
def test_exact_float32_clip_boundary_gradient(eps):
    from hvac.diagnostics import _clip_gradient_parts

    lo, hi = th.tensor(1 - eps), th.tensor(1 + eps)
    ratios = th.stack([th.nextafter(lo, th.tensor(-th.inf)), lo,
                       th.nextafter(lo, th.tensor(th.inf)), th.tensor(1.0),
                       th.nextafter(hi, th.tensor(-th.inf)), hi,
                       th.nextafter(hi, th.tensor(th.inf))]).repeat(2).requires_grad_()
    advantages = th.tensor([1.0] * 7 + [-1.0] * 7)
    with th.no_grad():
        parts = _clip_gradient_parts(advantages, ratios, eps)
    surrogate = advantages * ratios
    clipped = advantages * th.clamp(ratios, 1 - eps, 1 + eps)
    gu = th.autograd.grad(-surrogate.mean(), ratios, retain_graph=True)[0]
    gz = th.autograd.grad(-(surrogate * parts["weight"]).mean(), ratios, retain_graph=True)[0]
    gc = th.autograd.grad(-th.min(surrogate, clipped).mean(), ratios)[0]
    assert th.equal(gc, gu - gz)
    for i in (1, 5, 8, 12):  # exact float32 endpoints, for both advantage signs
        assert parts["weight"][i] < 1  # never completely zero a clamp/min endpoint
        assert gc[i] == gu[i] * (1 - parts["weight"][i])
    counted = sum(parts[k] for k in ("zeroed", "against", "boundary_roundoff", "surrogate_tie"))
    assert th.equal(counted, (th.abs(ratios.detach() - 1) > eps).float())


def test_consistency_and_normalized_variance():
    from hvac.diagnostics import _consistency
    vectors = [th.tensor([1.0, 2.0]), th.tensor([3.0, -2.0])]
    consistency, _ = _consistency(vectors)
    g = th.stack(vectors)
    mean_sq = g.mean(0).square().sum()
    second_moment = g.square().sum(1).mean()
    variance = (g - g.mean(0)).square().sum(1).mean()
    assert consistency == pytest.approx((mean_sq / second_moment).item())
    assert 1 - consistency == pytest.approx((variance / second_moment).item())


def test_gradient_probe_preserves_training_and_rng(tmp_path):
    from functools import partial
    from hvac.gradient_probe import ClipGradientProbePPO
    from scripts.analyze_clip_diagnostics import _same_nested

    plain = _train(PPO, 0.5)
    rng = th.get_rng_state().clone()
    probed = _train(partial(ClipGradientProbePPO, probe_rounds=(1, 3), probe_every=2), 0.5)
    assert _same_nested(plain.policy.state_dict(), probed.policy.state_dict())
    assert _same_nested(plain.policy.optimizer.state_dict(), probed.policy.optimizer.state_dict())
    assert th.equal(rng, th.get_rng_state())
    probed.save_diagnostics(tmp_path)
    assert len(probed.gradient_epochs) == 9
    assert len(probed.adam_probes) == 18
    assert all(0 <= r["extra_samples_frac"] <= 1 for r in probed.gradient_epochs)


def test_shadow_adam_same_clip_matches_actual_step():
    from functools import partial
    from hvac.gradient_probe import ClipGradientProbePPO

    model = _train(partial(ClipGradientProbePPO, compare_range=0.5,
                           probe_rounds=(1, 3), probe_every=2), 0.5)
    rows = model.adam_probes
    assert rows
    np.testing.assert_allclose([r["param_step_02"] for r in rows],
                               [r["param_step_05"] for r in rows], rtol=2e-4, atol=1e-7)
    np.testing.assert_allclose([r["mu_step_rms_02"] for r in rows],
                               [r["mu_step_rms_05"] for r in rows], rtol=2e-3, atol=2e-7)


def test_gaussian_kl_mean_and_sigma_decomposition():
    import pandas as pd
    from scripts.analyze_clip_diagnostics import kl_parts

    rounds = pd.DataFrame({
        "sigma_start": [1.0, 1.0, 2.0],
        "sigma_end": [1.0, 2.0, 1.0],
        "mean_shift_rms_end": [0.0, 0.4, 0.3],
    })
    mean_part, std_part = kl_parts(rounds)
    old_std = th.tensor(rounds["sigma_start"].to_numpy(), dtype=th.float64)
    new_std = th.tensor(rounds["sigma_end"].to_numpy(), dtype=th.float64)
    shift = th.tensor(rounds["mean_shift_rms_end"].to_numpy(), dtype=th.float64) * old_std
    expected = th.distributions.kl_divergence(
        th.distributions.Normal(th.zeros_like(old_std), old_std),
        th.distributions.Normal(shift, new_std),
    ).numpy()
    np.testing.assert_allclose(mean_part + std_part, expected, rtol=1e-12, atol=1e-12)


def test_missing_runs_fail_reproduction_check(tmp_path):
    from scripts.analyze_clip_diagnostics import reproduction_check
    assert not reproduction_check(tmp_path / "diag", tmp_path / "abl")


def test_runner_repairs_missing_diagnostics(tmp_path, monkeypatch):
    from scripts import run_clip_diagnostics as runner
    monkeypatch.setattr(runner, "OUT", tmp_path)
    run = tmp_path / "baseline" / "seed0"
    run.mkdir(parents=True)
    (run / "final_eval.json").write_text("{}")
    assert len(runner.build_jobs(["baseline"], [0])) == 1
    for f in ("spec.json", "progress.csv", "eval_curve.csv", "model.zip", "diag_epochs.csv", "diag_rounds.csv"):
        (run / f).touch()
    assert runner.build_jobs(["baseline"], [0]) == []
    assert len(runner.build_jobs(["baseline"], [0], force=True)) == 1
