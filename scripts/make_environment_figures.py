"""Figures and numbers for report section 1 (the environment). No learning involved.

Usage:  python scripts/make_environment_figures.py
Writes: docs/report/figures/env_*.png
Prints: every number quoted in docs/report/zh/01_environment.md
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from hvac.controllers import OffController, ThermostatController
from hvac.env import RoomConfig, RoomThermalEnv, discretize
from hvac.evaluation import eval_seeds, evaluate_controller, mean_metrics, run_episode

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures"
SEEDS = eval_seeds(100)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- numbers
def print_model_numbers(cfg: RoomConfig) -> None:
    a = np.array(
        [
            [-(cfg.ua_air_out + cfg.ua_air_mass) / cfg.c_air, cfg.ua_air_mass / cfg.c_air],
            [cfg.ua_air_mass / cfg.c_mass, -(cfg.ua_air_mass + cfg.ua_mass_out) / cfg.c_mass],
        ]
    )
    taus = sorted(-1.0 / np.linalg.eigvals(a).real / 3600.0)
    ua_total = cfg.ua_air_out + 1.0 / (1.0 / cfg.ua_air_mass + 1.0 / cfg.ua_mass_out)
    print("MODEL")
    print(f"  time constants: {taus[0]:.2f} h and {taus[1]:.1f} h")
    print(f"  overall heat loss coefficient: {ua_total:.1f} W/K")
    print(f"  heat needed to hold 21 C against -2 C outdoors: {ua_total * 23 / 1000:.2f} kW (heat pump: {cfg.q_hvac_max / 1000:.0f} kW)")
    print(f"  electricity per K of indoor-outdoor difference: {ua_total / cfg.cop:.1f} W")
    print(f"  max electric power {cfg.p_elec_max_kw:.2f} kW, max energy per step {cfg.energy_per_step_max_kwh:.2f} kWh")
    a_d, b_d = discretize(cfg)
    x = np.zeros(2)
    for k in range(1, 13):
        x = a_d @ x + b_d @ np.array([0.0, cfg.q_hvac_max, 0.0, 0.0])
        if k in (1, 4, 12):
            print(f"  full power for {k * 15:3d} min: air +{x[0]:.2f} K, walls +{x[1]:.2f} K (relative to leaving it off)")


def print_controller_numbers(cfg: RoomConfig) -> None:
    print("\nREFERENCE CONTROLLERS, 100 test days (cost = energy + 2 x discomfort)")
    controllers = [
        ("HVAC off", OffController()),
        ("thermostat from 8:00", ThermostatController(prestart_hours=0)),
        ("thermostat from 6:00", ThermostatController(prestart_hours=2)),
        ("thermostat from 5:00", ThermostatController(prestart_hours=3)),
        ("thermostat from 3:00", ThermostatController(prestart_hours=5)),
        ("thermostat 24 h", ThermostatController(always_on=True)),
        ("oracle", "oracle"),
    ]
    for name, ctrl in controllers:
        m = mean_metrics(evaluate_controller(ctrl, cfg, SEEDS))
        cost = m["energy_kwh"] + cfg.comfort_weight * m["discomfort_kh"]
        print(f"  {name:22s} energy {m['energy_kwh']:5.2f}  discomfort {m['discomfort_kh']:6.2f}  violated {m['violation_pct']:5.1f}%  cost {cost:6.2f}")

    env = RoomThermalEnv(cfg)
    rng = np.random.default_rng(0)
    rows = [
        run_episode(env, s, policy=lambda _e, _o: float(np.clip(rng.normal(), -1.0, 1.0)))
        for s in SEEDS
    ]
    m = mean_metrics(rows)
    print(f"  {'random N(0,1) actions':22s} energy {m['energy_kwh']:5.2f}  discomfort {m['discomfort_kh']:6.2f}  "
          f"violated {m['violation_pct']:5.1f}%  cost {m['energy_kwh'] + cfg.comfort_weight * m['discomfort_kh']:6.2f}")


def print_oracle_numbers(cfg: RoomConfig) -> None:
    env = RoomThermalEnv(cfg)
    days = []
    for s in SEEDS:
        ep = run_episode(env, s, oracle=True, record=True)
        tr = ep["trajectory"]
        act, t_air = tr["hvac_action"], tr["t_air"]
        occ = env.occupied[: cfg.steps_per_episode]
        on = np.flatnonzero(np.abs(act) > 0.05)
        days.append(
            {
                "t_mean": ep["t_out_mean"],
                "energy": ep["energy_kwh"],
                "first_on": on[0] * cfg.dt_hours if len(on) else np.nan,
                "heats": bool((act > 0.05).any()),
                "cools": bool((act < -0.05).any()),
                "edge": float(np.mean((np.abs(t_air[occ] - cfg.comfort_low) < 0.05) | (np.abs(t_air[occ] - cfg.comfort_high) < 0.05))),
                "before_8": float(tr["energy_kwh"][tr["hour"] < cfg.occupied_start_hour].sum()),
            }
        )
    t_mean = np.array([d["t_mean"] for d in days])
    print("\nORACLE BEHAVIOUR BY DAILY MEAN OUTDOOR TEMPERATURE")
    for lo, hi in ((-2, 5), (5, 12), (12, 19), (19, 26), (26, 32)):
        sel = [d for d, t in zip(days, t_mean) if lo <= t < hi]
        first = np.array([d["first_on"] for d in sel])
        print(f"  {lo:3d}..{hi:2d} C  n={len(sel):2d}  energy {np.mean([d['energy'] for d in sel]):5.2f} kWh  "
              f"first switch-on {np.nanmean(first):5.2f} h (range {np.nanmin(first):.2f}-{np.nanmax(first):.2f})  "
              f"heat only {sum(d['heats'] and not d['cools'] for d in sel):2d}  cool only {sum(d['cools'] and not d['heats'] for d in sel):2d}  "
              f"both {sum(d['heats'] and d['cools'] for d in sel):2d}  neither {sum(not d['heats'] and not d['cools'] for d in sel):2d}")
    print(f"  occupied steps where the oracle's air temperature is within 0.05 K of a band edge: {100 * np.mean([d['edge'] for d in days]):.0f}%")
    print(f"  oracle energy used before 8:00: {np.mean([d['before_8'] for d in days]):.2f} of {np.mean([d['energy'] for d in days]):.2f} kWh")


def print_one_node_comparison() -> None:
    print("\nONE-NODE ROOMS VS OURS (cost of the best thermostat start time vs the oracle)")
    variants = {
        "two nodes (ours)": RoomConfig(),
        "one light node (walls store no heat)": RoomConfig(c_mass=1e3),
        "one heavy node (air locked to walls)": RoomConfig(ua_air_mass=1e6),
    }
    for name, cfg in variants.items():
        env = RoomThermalEnv(cfg)
        cold = [s for s in SEEDS if env.reset(seed=s)[1]["t_out_mean"] < 5.0]
        t8 = np.mean([run_episode(env, s, policy=lambda _e, _o: 0.0, record=True)["trajectory"]["t_air"][31] for s in cold])
        a_d, b_d = discretize(cfg)
        rise = (a_d @ np.zeros(2) + b_d @ np.array([0.0, cfg.q_hvac_max, 0.0, 0.0]))[0]
        costs = {}
        for p in (0, 1, 2, 3, 5):
            m = mean_metrics(evaluate_controller(ThermostatController(prestart_hours=p), cfg, SEEDS))
            costs[p] = m["energy_kwh"] + cfg.comfort_weight * m["discomfort_kh"]
        o = mean_metrics(evaluate_controller("oracle", cfg, SEEDS))
        print(f"  {name:38s} air after 15 min full power +{rise:.2f} K | air at 8:00 on cold days, HVAC off {t8:5.1f} C | "
              f"oracle cost {o['energy_kwh'] + cfg.comfort_weight * o['discomfort_kh']:.2f} (discomfort {o['discomfort_kh']:.2f}) | "
              f"thermostat costs by head start " + ", ".join(f"{p}h {c:.2f}" for p, c in costs.items()))


def print_margin_price(cfg: RoomConfig) -> None:
    """What one extra kelvin of safety margin costs, measured by narrowing the band for the oracle."""
    env = RoomThermalEnv(cfg)
    means = {s: env.reset(seed=s)[1]["t_out_mean"] for s in SEEDS}
    cold = [s for s in SEEDS if means[s] < 5.0]
    hot = [s for s in SEEDS if means[s] >= 26.0]
    hours = cfg.occupied_end_hour - cfg.occupied_start_hour

    def energy(config: RoomConfig, seeds: list[int]) -> float:
        return mean_metrics(evaluate_controller("oracle", config, seeds))["energy_kwh"]

    heat = (energy(replace(cfg, comfort_low=cfg.comfort_low + 1.0), cold) - energy(cfg, cold)) / hours
    cool = (energy(replace(cfg, comfort_high=cfg.comfort_high - 1.0), hot) - energy(cfg, hot)) / hours
    print("\nPRICE OF ONE KELVIN OF SAFETY MARGIN (oracle, band narrowed by 1 K)")
    print(f"  cold days (<5 C, n={len(cold)}): {heat:.3f} kWh per K.h -> a violation costs {cfg.comfort_weight / heat:.0f}x as much")
    print(f"  hot days (>=26 C, n={len(hot)}): {cool:.3f} kWh per K.h -> a violation costs {cfg.comfort_weight / cool:.0f}x as much")


def print_weather_numbers(cfg: RoomConfig) -> None:
    env = RoomThermalEnv(cfg)
    lows, highs = [], []
    for s in SEEDS:
        env.reset(seed=s)
        lows.append(env.t_out.min())
        highs.append(env.t_out.max())
    sd = cfg.t_out_noise_sigma / np.sqrt(1.0 - cfg.t_out_noise_rho**2)
    print("\nWEATHER, 100 test days")
    print(f"  coldest instant {min(lows):.1f} C, hottest instant {max(highs):.1f} C; noise standard deviation {sd:.2f} K")


# ---------------------------------------------------------------- figures
def fig_schematic(cfg: RoomConfig) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.45, 6)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 10, 6, color="#e8f1fb", zorder=0))
    ax.text(0.25, 5.55, r"室外  $T_{out}$", fontsize=13, color="#1f4e79", va="center")
    ax.add_patch(Rectangle((2.3, 0.9), 6.4, 3.9, facecolor="white", edgecolor="#555555", lw=2.5, zorder=1))
    ax.text(5.5, 4.52, "房间", fontsize=11, color="#555555", ha="center")

    def box(x, y, w, h, text, face):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", facecolor=face, edgecolor="#333333", lw=1.5, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, zorder=4)

    def arrow(p, q, color="#333333", style="<->", lw=2.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=18, color=color, lw=lw, zorder=5))

    box(3.0, 2.35, 1.9, 1.3, "室内空气\n" + r"$T_a$" + "（变化快）", "#fff2cc")
    box(6.2, 2.35, 1.9, 1.3, "墙体与楼板\n" + r"$T_m$" + "（变化慢）", "#d9d2c5")

    arrow((4.98, 3.0), (6.12, 3.0))
    ax.text(5.55, 3.2, r"$U_{am}$", ha="center", fontsize=12)
    ax.text(5.55, 2.62, f"{cfg.ua_air_mass:.0f} W/K", ha="center", fontsize=9.5)

    arrow((0.9, 3.0), (2.92, 3.0))
    ax.text(1.55, 3.2, r"$U_{ao}$", ha="center", fontsize=12)
    ax.text(1.2, 2.5, f"窗户与渗风\n{cfg.ua_air_out:.0f} W/K", ha="center", fontsize=9.5, va="center")

    arrow((8.18, 3.0), (9.6, 3.0))
    ax.text(9.25, 3.2, r"$U_{mo}$", ha="center", fontsize=12)
    ax.text(9.3, 2.5, f"外墙传导\n{cfg.ua_mass_out:.0f} W/K", ha="center", fontsize=9.5, va="center")

    red = "#c00000"
    ax.add_patch(FancyBboxPatch((3.0, 1.1), 1.9, 0.62, boxstyle="round,pad=0.06", facecolor="#fde0dd", edgecolor=red, lw=2, zorder=3))
    ax.text(3.95, 1.41, r"热泵  $Q_{hvac}$", ha="center", va="center", fontsize=12, color=red, zorder=4)
    arrow((3.95, 1.8), (3.95, 2.27), color=red, style="<->", lw=2.5)
    ax.text(5.1, 1.41, r"$= a \times$" + f"{cfg.q_hvac_max:.0f} W，  " + r"$a\in[-1,1]$" + " 由智能体决定", fontsize=11, color=red, va="center")

    sun = "#e69f00"
    ax.text(5.55, 5.6, r"太阳辐射  $Q_{sun}$", ha="center", fontsize=12, color=sun)
    arrow((5.3, 5.4), (4.3, 3.75), color=sun, style="->")
    arrow((5.8, 5.4), (6.9, 3.75), color=sun, style="->")
    ax.text(4.35, 4.25, f"{100 * cfg.solar_to_air:.0f}%", fontsize=10, color=sun, ha="right")
    ax.text(6.8, 4.25, f"{100 * (1 - cfg.solar_to_air):.0f}%", fontsize=10, color=sun, ha="left")

    ax.text(2.55, 4.15, r"人员与设备  $Q_{int}$", fontsize=11, color="#2e7d32", ha="left")
    arrow((3.1, 4.0), (3.5, 3.75), color="#2e7d32", style="->")
    fig.tight_layout()
    fig.savefig(FIG / "env_schematic.png", dpi=150)
    plt.close(fig)


def fig_step_response(cfg: RoomConfig) -> None:
    a_d, b_d = discretize(cfg)
    t_out, start, on_from, on_to, total = 15.0, 15.0, 1.0, 3.0, 10.0
    n = int(total / cfg.dt_hours)
    x = np.array([start, start])
    hours, air, mass = [0.0], [x[0]], [x[1]]
    for k in range(n):
        h = k * cfg.dt_hours
        q = cfg.q_hvac_max if on_from <= h < on_to else 0.0
        x = a_d @ x + b_d @ np.array([t_out, q, 0.0, 0.0])
        hours.append(h + cfg.dt_hours)
        air.append(x[0])
        mass.append(x[1])
    hours, air, mass = np.array(hours), np.array(air), np.array(mass)
    i_off = int(on_to / cfg.dt_hours)
    print("\nSTEP RESPONSE FIGURE (room and outdoors both start at 15 C, full heating from hour 1 to hour 3)")
    print(f"  at switch-off: air {air[i_off]:.1f} C, walls {mass[i_off]:.1f} C; one hour later: air {air[i_off + 4]:.1f} C, walls {mass[i_off + 4]:.1f} C; "
          f"at hour 10: air {air[-1]:.1f} C, walls {mass[-1]:.1f} C")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axvspan(on_from, on_to, color="#fde0dd", label="热泵满功率制热")
    ax.plot(hours, air, color="C1", lw=2.2, label=r"空气温度 $T_a$")
    ax.plot(hours, mass, color="#6b5b45", lw=2.2, label=r"墙体温度 $T_m$")
    ax.axhline(t_out, color="grey", lw=1, ls=":", label="室外温度")
    ax.set_xlabel("时间（小时）")
    ax.set_ylabel("温度（°C）")
    ax.set_xlim(0, total)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "env_step_response.png", dpi=150)
    plt.close(fig)


def fig_days(cfg: RoomConfig) -> None:
    env = RoomThermalEnv(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cmap = plt.get_cmap("coolwarm")
    lo, hi = cfg.t_out_mean_range
    for seed in range(12):
        _, info = env.reset(seed=seed)
        color = cmap((info["t_out_mean"] - lo) / (hi - lo))
        axes[0].plot(env.hours, env.t_out, color=color, lw=1.4)
        axes[1].plot(env.hours, env.q_solar, color=color, lw=1.4)
    for ax in axes:
        ax.axvspan(cfg.occupied_start_hour, cfg.occupied_end_hour, color="green", alpha=0.08, label="有人（8–18 点）")
        ax.set_xlabel("一天中的时刻")
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.3)
    axes[0].axhspan(cfg.comfort_low, cfg.comfort_high, color="green", alpha=0.25, label="舒适区 21–24 °C")
    axes[0].set_ylabel("室外温度（°C）")
    axes[0].set_title("12 个随机抽到的日子：室外温度")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].set_ylabel("太阳辐射得热（W）")
    axes[1].set_title("同样 12 天：透过窗户的太阳辐射")
    fig.tight_layout()
    fig.savefig(FIG / "env_days.png", dpi=150)
    plt.close(fig)


def fig_controllers(cfg: RoomConfig) -> None:
    env = RoomThermalEnv(cfg)
    means = {s: env.reset(seed=s)[1]["t_out_mean"] for s in SEEDS}
    picks = [
        ("冷天", min(SEEDS, key=lambda s: abs(means[s] - 1.0))),
        ("热天", min(SEEDS, key=lambda s: abs(means[s] - 30.0))),
    ]
    thermostat = ThermostatController(prestart_hours=0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    print("\nCONTROLLER FIGURE")
    for col, (label, seed) in enumerate(picks):
        episodes = {
            "不开机": (run_episode(env, seed, policy=lambda _e, _o: 0.0, record=True), "grey", "--"),
            "温控器，8 点开机": (run_episode(env, seed, policy=lambda e, _o: thermostat.act(e), record=True), "C1", "-"),
            "理论最优": (run_episode(env, seed, oracle=True, record=True), "k", "-"),
        }
        top, bottom = axes[0, col], axes[1, col]
        oracle = episodes["理论最优"][0]
        hours = oracle["trajectory"]["hour"] + cfg.dt_hours
        top.fill_between([cfg.occupied_start_hour, cfg.occupied_end_hour], cfg.comfort_low, cfg.comfort_high,
                         color="green", alpha=0.18, lw=0, label="有人时的舒适区（8–18 点）")
        top.plot(hours, oracle["trajectory"]["t_out"], color="C0", lw=1, alpha=0.7, label="室外温度")
        for name, (ep, color, ls) in episodes.items():
            tr = ep["trajectory"]
            top.plot(hours, tr["t_air"], color=color, ls=ls, lw=1.8, label=f"空气温度：{name}")
            if name != "不开机":
                bottom.step(hours, tr["hvac_action"], color=color, lw=1.5, where="pre")
            print(f"  {label} (mean outdoor {ep['t_out_mean']:.1f} C) {name}: energy {ep['energy_kwh']:.2f}, discomfort {ep['discomfort_kh']:.2f}, air at 8:00 {tr['t_air'][31]:.1f} C")
        top.plot(hours, oracle["trajectory"]["t_mass"], color="k", ls=":", lw=1.3, label="墙体温度：理论最优")
        otr = oracle["trajectory"]
        first_on = np.flatnonzero(np.abs(otr["hvac_action"]) > 0.05)[0] * cfg.dt_hours
        print(f"  {label} oracle first switches on at {first_on:.2f} h")
        for end_hour in (9.0, 17.75):
            k = int(round(end_hour / cfg.dt_hours)) - 1
            print(f"  {label} oracle, step ending {end_hour:5.2f} h: action {otr['hvac_action'][k]:+.2f}, air {otr['t_air'][k]:.2f} C, "
                  f"walls {otr['t_mass'][k]:.2f} C, outdoor {otr['t_out'][k]:.1f} C, sun {env.q_solar[k]:.0f} W")
        top.set_title(f"{label}（室外日均 {oracle['t_out_mean']:.0f} °C）")
        top.set_xlim(0, 24)
        top.grid(alpha=0.3)
        bottom.axhline(0, color="grey", lw=0.6)
        bottom.set_ylim(-1.08, 1.08)
        bottom.set_xlabel("一天中的时刻")
        bottom.set_xticks(range(0, 25, 4))
        bottom.grid(alpha=0.3)
    axes[0, 0].set_ylabel("温度（°C）")
    axes[1, 0].set_ylabel("动作 a（−1 制冷，+1 制热）")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG / "env_controllers.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    config = RoomConfig()
    print_model_numbers(config)
    print_weather_numbers(config)
    print_controller_numbers(config)
    print_oracle_numbers(config)
    print_one_node_comparison()
    print_margin_price(config)
    fig_schematic(config)
    fig_step_response(config)
    fig_days(config)
    fig_controllers(config)
