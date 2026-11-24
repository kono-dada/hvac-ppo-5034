"""Tables and figures for the pre-experiments.

Usage:  python scripts/analyze_preexperiments.py
Reads:  runs/pre/
Writes: tasks/task-20261124/report/figures/*.png and summary.csv next to them
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from hvac.controllers import OffController, ThermostatController
from hvac.env import RoomConfig, RoomThermalEnv
from hvac.evaluation import eval_seeds, evaluate_controller, mean_metrics, run_episode

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "pre"
FIG = ROOT / "tasks" / "task-20261124" / "report" / "figures"
REF_WEIGHT = RoomConfig().comfort_weight  # all variants are compared at this price of discomfort
CURVE_DAYS = 30  # must match n_curve_days in hvac.train.train_ppo

LABELS = {
    "baseline": "baseline",
    "end_truncate": "day end = truncate",
    "weight_0.5": "comfort weight 0.5",
    "weight_8": "comfort weight 8",
    "no_mass_obs": "wall temperature hidden",
    "reward_x0.1": "reward x 0.1",
    "reward_vecnorm": "reward normalised (VecNormalize)",
    "baseline_long": "baseline, 1.5M steps",
}


def load_runs(variant: str) -> list[dict]:
    out = []
    for run in sorted((RUNS / variant).glob("seed*")):
        final = run / "final_eval.json"
        if not final.exists():
            continue
        out.append(
            {
                "dir": run,
                "final": json.loads(final.read_text()),
                "curve": pd.read_csv(run / "eval_curve.csv"),
                "progress": pd.read_csv(run / "progress.csv"),
            }
        )
    return out


def cost(energy, discomfort):
    return energy + REF_WEIGHT * discomfort


def reference_table(cfg: RoomConfig, seeds: list[int]) -> pd.DataFrame:
    controllers = [
        ("HVAC off", OffController()),
        ("thermostat, starts at 8:00", ThermostatController(prestart_hours=0)),
        ("thermostat, starts at 6:00", ThermostatController(prestart_hours=2)),
        ("thermostat, starts at 5:00", ThermostatController(prestart_hours=3)),
        ("thermostat, on 24 h", ThermostatController(always_on=True)),
        ("oracle (knows the whole day)", "oracle"),
    ]
    rows = []
    for name, ctrl in controllers:
        m = mean_metrics(evaluate_controller(ctrl, cfg, seeds))
        rows.append(
            {
                "controller": name,
                "energy_kwh": m["energy_kwh"],
                "discomfort_kh": m["discomfort_kh"],
                "violation_pct": m["violation_pct"],
                "cost": cost(m["energy_kwh"], m["discomfort_kh"]),
            }
        )
    return pd.DataFrame(rows)


def variant_table() -> pd.DataFrame:
    rows = []
    for variant in LABELS:
        runs = load_runs(variant)
        if not runs:
            continue
        e = np.array([r["final"]["mean"]["energy_kwh"] for r in runs])
        d = np.array([r["final"]["mean"]["discomfort_kh"] for r in runs])
        v = np.array([r["final"]["mean"]["violation_pct"] for r in runs])
        c = cost(e, d)
        # Stability: how much the evaluation cost still moves over the last 40% of training.
        wobble = []
        for r in runs:
            curve = r["curve"]
            tail = curve[curve["timesteps"] >= 0.6 * curve["timesteps"].max()]
            wobble.append(cost(tail["energy_kwh"], tail["discomfort_kh"]).std())
        rows.append(
            {
                "variant": LABELS[variant],
                "seeds": len(runs),
                "energy_kwh": e.mean(),
                "energy_sd": e.std(ddof=1),
                "discomfort_kh": d.mean(),
                "discomfort_sd": d.std(ddof=1),
                "violation_pct": v.mean(),
                "cost": c.mean(),
                "cost_sd": c.std(ddof=1),
                "cost_min": c.min(),
                "cost_max": c.max(),
                "late_wobble": float(np.mean(wobble)),
            }
        )
    return pd.DataFrame(rows)


def plot_learning_curves(refs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("cost", "Daily cost = energy + 2 x discomfort", (3.0, 9.0)),
        ("energy_kwh", "Energy use (kWh per day)", (3.0, 9.0)),
        ("discomfort_kh", "Discomfort (K·h per day)", (0.0, 1.0)),
    ]
    colors = {"baseline_long": "C0", "end_truncate": "C1", "no_mass_obs": "C2", "reward_vecnorm": "C3"}
    for ax, (key, title, ylim) in zip(axes, panels):
        for variant, color in colors.items():
            runs = load_runs(variant)
            if not runs:
                continue
            for i, r in enumerate(runs):
                curve = r["curve"]
                y = cost(curve["energy_kwh"], curve["discomfort_kh"]) if key == "cost" else curve[key]
                ax.plot(curve["timesteps"] / 1e3, y, color=color, alpha=0.55, lw=1.0,
                        label=LABELS[variant] if i == 0 else None)
        ref_key = "cost" if key == "cost" else key
        for name, style in (("oracle (knows the whole day)", ":"), ("thermostat, starts at 5:00", "--")):
            val = refs.loc[refs["controller"] == name, ref_key].item()
            ax.axhline(val, color="k", ls=style, lw=1.2, label=name)
        ax.set_title(title)
        ax.set_xlabel("training steps (thousands)")
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 1500)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Deterministic evaluation during training on 30 fixed days (one line per seed; "
                 "reference lines use the same 30 days)")
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png", dpi=150)
    plt.close(fig)


def plot_day_examples(cfg: RoomConfig, seeds: list[int]) -> None:
    runs = load_runs("baseline")
    finals = [cost(r["final"]["mean"]["energy_kwh"], r["final"]["mean"]["discomfort_kh"]) for r in runs]
    median_run = runs[int(np.argsort(finals)[len(finals) // 2])]
    model = PPO.load(median_run["dir"] / "model", device="cpu")
    env = RoomThermalEnv(cfg)
    means = {s: env.reset(seed=s)[1]["t_out_mean"] for s in seeds}
    picks = [
        ("cold day", min(seeds, key=lambda s: abs(means[s] - 1.0))),
        ("mild day", min(seeds, key=lambda s: abs(means[s] - 15.0))),
        ("hot day", min(seeds, key=lambda s: abs(means[s] - 30.0))),
    ]
    thermostat = ThermostatController(prestart_hours=0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True)
    for col, (label, seed) in enumerate(picks):
        episodes = {
            "oracle": run_episode(env, seed, oracle=True, record=True),
            "thermostat 8:00": run_episode(env, seed, policy=lambda e, _o: thermostat.act(e), record=True),
            "PPO (baseline)": run_episode(
                env, seed, policy=lambda _e, o: float(model.predict(o, deterministic=True)[0][0]), record=True
            ),
        }
        top, bottom = axes[0, col], axes[1, col]
        hours = episodes["oracle"]["trajectory"]["hour"] + cfg.dt_hours
        top.fill_between([cfg.occupied_start_hour, cfg.occupied_end_hour], cfg.comfort_low, cfg.comfort_high,
                         color="green", alpha=0.15, lw=0, label="comfort band (occupied)")
        top.plot(hours, episodes["oracle"]["trajectory"]["t_out"], color="grey", lw=1, label="outdoor")
        for (name, ep), color in zip(episodes.items(), ("k", "C1", "C0")):
            tr = ep["trajectory"]
            top.plot(hours, tr["t_air"], color=color, lw=1.6,
                     label=f"{name}: {ep['energy_kwh']:.1f} kWh, {ep['discomfort_kh']:.2f} K·h")
            bottom.step(hours, tr["hvac_action"], color=color, lw=1.4, where="pre", label=name)
        top.set_title(f"{label} (mean outdoor {episodes['oracle']['t_out_mean']:.0f} °C)")
        top.legend(fontsize=7, loc="best")
        top.grid(alpha=0.3)
        bottom.axhline(0, color="grey", lw=0.6)
        bottom.set_ylim(-1.05, 1.05)
        bottom.set_xlabel("hour of day")
        bottom.set_xticks(range(0, 25, 4))
        bottom.grid(alpha=0.3)
    axes[0, 0].set_ylabel("temperature (°C)")
    axes[1, 0].set_ylabel("action (-1 cool ... +1 heat)")
    fig.tight_layout()
    fig.savefig(FIG / "day_examples.png", dpi=150)
    plt.close(fig)


def plot_weight_tradeoff(refs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for variant, color in (("weight_0.5", "C4"), ("baseline", "C0"), ("weight_8", "C3")):
        runs = load_runs(variant)
        e = [r["final"]["mean"]["energy_kwh"] for r in runs]
        d = [r["final"]["mean"]["discomfort_kh"] for r in runs]
        label = LABELS[variant] if variant != "baseline" else "comfort weight 2 (baseline)"
        ax.scatter(d, e, color=color, s=40, label=label)
    for name, marker in (("oracle (knows the whole day)", "*"), ("thermostat, starts at 5:00", "s"),
                         ("thermostat, starts at 8:00", "D"), ("thermostat, on 24 h", "^")):
        row = refs[refs["controller"] == name].iloc[0]
        ax.scatter(row["discomfort_kh"], row["energy_kwh"], color="k", marker=marker, s=70, label=name)
    ax.set_xlabel("discomfort (K·h per day)")
    ax.set_ylabel("energy (kWh per day)")
    ax.set_title("Final policies: one dot per seed, 100 test days")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "weight_tradeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = RoomConfig()
    seeds = eval_seeds(100)
    refs = reference_table(cfg, seeds)
    # The training curves are evaluated on the first 30 of these days, which are not
    # equally hard on average, so their reference lines must use the same 30 days.
    curve_refs = reference_table(cfg, eval_seeds(CURVE_DAYS))
    table = variant_table()
    pd.set_option("display.width", 200)
    print("reference controllers, 100 test days")
    print(refs.round(2).to_string(index=False))
    print(f"\nreference controllers, the {CURVE_DAYS} days used for training curves")
    print(curve_refs.round(2).to_string(index=False))
    print("\nPPO variants, final policy, 100 test days")
    print(table.round(2).to_string(index=False))
    refs.to_csv(FIG / "reference_controllers.csv", index=False)
    curve_refs.to_csv(FIG / "reference_controllers_curve_days.csv", index=False)
    table.to_csv(FIG / "summary.csv", index=False)
    plot_learning_curves(curve_refs)
    plot_day_examples(cfg, seeds)
    plot_weight_tradeoff(refs)
