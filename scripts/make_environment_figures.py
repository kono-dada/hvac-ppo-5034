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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from hvac import plotting as ps
from hvac.controllers import OffController, ThermostatController, tuned_thermostat
from hvac.env import RoomConfig, RoomThermalEnv, discretize
from hvac.evaluation import eval_seeds, evaluate_controller, mean_metrics, run_episode

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures"
SEEDS = eval_seeds(100)

ps.use_style()


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
        ("thermostat, tuned", tuned_thermostat()),
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
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.35, 6.1)
    ax.axis("off")
    ax.set_aspect("auto")

    ax.add_patch(FancyBboxPatch((0.05, 0.4), 9.9, 5.65, boxstyle="round,pad=0,rounding_size=0.25",
                                facecolor=ps.COOL_SOFT, edgecolor="none", alpha=0.55, zorder=0))
    ax.text(0.35, 5.7, "室外", fontsize=13, weight="semibold", color=ps.COOL, va="center")
    ax.text(0.35, 5.33, r"温度 $T_{out}$ 随天气变化", fontsize=9.5, color=ps.COOL, va="center")

    ax.add_patch(FancyBboxPatch((2.3, 0.75), 6.4, 4.05, boxstyle="round,pad=0,rounding_size=0.18",
                                facecolor="white", edgecolor="#57606a", lw=2.2, zorder=1))
    ax.text(2.5, 4.55, "房间", fontsize=11, weight="semibold", color="#57606a", va="center")

    def node(x, y, w, h, title, sub, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.14", facecolor=face,
                                    edgecolor=edge, lw=1.8, zorder=3))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12.5, weight="semibold", zorder=4)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9.5, color=ps.MUTED, zorder=4)

    def arrow(p, q, color=ps.INK, style="<|-|>", lw=1.8, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, zorder=5,
                                     shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))

    def pill(x, y, text, sub=None, color=ps.INK):
        ax.text(x, y, text, ha="center", va="center", fontsize=10, color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.6", fc="white", ec="#afb8c1", lw=0.9))
        if sub:
            ax.text(x, y + 0.36, sub, ha="center", va="bottom", fontsize=10, color=ps.INK, zorder=6)

    node(3.0, 2.3, 1.95, 1.35, r"室内空气  $T_a$", "热容小 · 变化快", "#fdebd9", ps.AIR)
    node(6.1, 2.3, 1.95, 1.35, r"墙体与楼板  $T_m$", "热容大 · 变化慢", "#efe6dc", ps.MASS)

    arrow((4.98, 2.98), (6.07, 2.98))
    pill(5.52, 2.98, f"{cfg.ua_air_mass:.0f} W/K", r"$U_{am}$")
    arrow((0.75, 2.98), (2.97, 2.98))
    pill(1.6, 2.98, f"{cfg.ua_air_out:.0f} W/K", r"$U_{ao}$  窗户与渗风")
    arrow((8.08, 2.98), (9.8, 2.98))
    pill(9.27, 2.98, f"{cfg.ua_mass_out:.0f} W/K", r"$U_{mo}$" + "\n外墙传导")

    # heat pump: heating up, cooling down
    ax.add_patch(FancyBboxPatch((3.0, 0.98), 1.95, 0.72, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor="white", edgecolor=ps.INK, lw=1.6, zorder=3))
    ax.text(3.975, 1.34, r"热泵  $Q_{hvac}$", ha="center", va="center", fontsize=12, weight="semibold", zorder=4)
    arrow((3.72, 1.74), (3.72, 2.26), color=ps.HEAT, style="-|>", lw=2.2)
    arrow((4.23, 2.26), (4.23, 1.74), color=ps.COOL, style="-|>", lw=2.2)
    ax.text(3.62, 2.0, "制热", ha="right", va="center", fontsize=9, color=ps.HEAT)
    ax.text(4.33, 2.0, "制冷", ha="left", va="center", fontsize=9, color=ps.COOL)
    ax.text(5.2, 1.48, r"$Q_{hvac} = a \times$" + f" {cfg.q_hvac_max:.0f} W", fontsize=11.5, va="center", zorder=4)
    ax.text(5.2, 1.08, r"$a \in [-1, 1]$：智能体每 15 分钟选一次", fontsize=9.5, color=ps.MUTED, va="center")

    # sun
    sx, sy = 5.52, 5.45
    ax.add_patch(plt.Circle((sx, sy), 0.2, color=ps.SUN, zorder=4))
    for ang in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([sx + 0.28 * np.cos(ang), sx + 0.4 * np.cos(ang)], [sy + 0.28 * np.sin(ang), sy + 0.4 * np.sin(ang)],
                color=ps.SUN, lw=1.6, zorder=4, solid_capstyle="round")
    ax.text(sx + 0.55, sy, r"太阳辐射  $Q_{sun}$", fontsize=11.5, color=ps.SUN, va="center", weight="semibold")
    arrow((sx - 0.25, sy - 0.4), (4.35, 3.72), color=ps.SUN, style="-|>", rad=0.1)
    arrow((sx + 0.25, sy - 0.4), (6.75, 3.72), color=ps.SUN, style="-|>", rad=-0.1)
    ax.text(4.72, 4.45, f"{100 * cfg.solar_to_air:.0f}%", fontsize=10, color=ps.SUN, ha="right", weight="semibold")
    ax.text(6.35, 4.45, f"{100 * (1 - cfg.solar_to_air):.0f}%", fontsize=10, color=ps.SUN, ha="left", weight="semibold")

    ax.text(2.5, 4.12, r"人员与设备  $Q_{int}$", fontsize=10.5, color=ps.INTERNAL, ha="left", va="center")
    arrow((3.05, 3.98), (3.4, 3.7), color=ps.INTERNAL, style="-|>")
    ps.save(fig, FIG / "env_schematic.png")


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

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.axvspan(on_from, on_to, color=ps.HEAT_SOFT, lw=0, zorder=0)
    ax.text((on_from + on_to) / 2, 23.05, "热泵满功率制热", ha="center", va="center", fontsize=9.5, color=ps.HEAT)
    ax.axhline(t_out, color=ps.MUTED, lw=1, ls=(0, (2, 2)))
    ax.plot(hours, air, color=ps.AIR, lw=2.6)
    ax.plot(hours, mass, color=ps.MASS, lw=2.6)

    def mark(i, text, dx, dy, color, ha="left"):
        ax.plot(hours[i], air[i] if color == ps.AIR else mass[i], "o", ms=5.5, color=color, mec="white", mew=1.2, zorder=5)
        y = air[i] if color == ps.AIR else mass[i]
        ax.annotate(text, (hours[i], y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=color, ha=ha,
                    va="center", arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=0, shrinkB=4))

    mark(i_off, f"关机时 {air[i_off]:.1f} °C", 16, 4, ps.AIR)
    mark(i_off + 4, f"1 小时后 {air[i_off + 4]:.1f} °C", 18, 14, ps.AIR)
    mark(i_off, f"墙体只到 {mass[i_off]:.1f} °C", 14, -16, ps.MASS)

    ax.text(1.5, 20.7, r"空气 $T_a$", color=ps.AIR, fontsize=10.5, weight="semibold", ha="right")
    ax.text(7.0, 17.0, r"空气与墙体逐渐趋同，缓慢冷却", color=ps.MUTED, fontsize=9, ha="left")
    ax.text(5.2, 16.25, r"墙体 $T_m$", color=ps.MASS, fontsize=10.5, weight="semibold", ha="left", va="top")
    ax.text(total - 0.1, t_out - 0.12, f"室外 {t_out:.0f} °C", color=ps.MUTED, fontsize=9, ha="right", va="top")
    ax.set_xlabel("时间（小时）")
    ax.set_ylabel("温度（°C）")
    ax.set_xlim(0, total)
    ax.set_ylim(14.2, 23.5)
    ax.grid(axis="x", visible=False)
    ps.save(fig, FIG / "env_step_response.png")


def fig_days(cfg: RoomConfig) -> None:
    env = RoomThermalEnv(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"wspace": 0.22})
    cmap = LinearSegmentedColormap.from_list("days", ["#2b5c9e", "#6aa3c8", "#e9b44c", "#d9542b", "#9e1f1f"])
    lo, hi = cfg.t_out_mean_range
    norm = Normalize(lo, hi)
    for seed in range(12):
        _, info = env.reset(seed=seed)
        color = cmap(norm(info["t_out_mean"]))
        axes[0].plot(env.hours, env.t_out, color=color, lw=1.4, alpha=0.95)
        axes[1].plot(env.hours, env.q_solar, color=color, lw=1.4, alpha=0.95)
    for ax in axes:
        ax.axvspan(cfg.occupied_start_hour, cfg.occupied_end_hour, color=ps.BG_SOFT, lw=0, zorder=0)
        ax.axvline(cfg.occupied_start_hour, color=ps.FAINT, lw=0.8, zorder=0)
        ax.axvline(cfg.occupied_end_hour, color=ps.FAINT, lw=0.8, zorder=0)
        ax.set_xlabel("一天中的时刻")
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlim(0, 24)
        ax.grid(axis="x", visible=False)
    axes[0].axhspan(cfg.comfort_low, cfg.comfort_high, color=ps.COMFORT_SOFT, lw=0, zorder=0.5)
    axes[0].text(23.7, (cfg.comfort_low + cfg.comfort_high) / 2, f"舒适区 {cfg.comfort_low:.0f}–{cfg.comfort_high:.0f} °C",
                 color=ps.COMFORT, fontsize=8.5, va="center", ha="right", weight="semibold", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for ax in axes:
        ax.text((cfg.occupied_start_hour + cfg.occupied_end_hour) / 2, 0.015, "有人（8–18 点）", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8.5, color=ps.MUTED)
    axes[0].set_ylabel("室外温度（°C）")
    ps.panel_title(axes[0], "a", "室外温度")
    axes[1].set_ylabel("太阳辐射得热（W）")
    ps.panel_title(axes[1], "b", "透过窗户的太阳辐射")
    cb = fig.colorbar(ScalarMappable(norm, cmap), ax=axes, fraction=0.025, pad=0.02, aspect=25)
    cb.set_label("当天室外平均温度（°C）")
    cb.outline.set_visible(False)
    ps.header(fig, "随机抽到的 12 天", "同一种颜色在两幅图里是同一天", top=0.95)
    ps.save(fig, FIG / "env_days.png")


def fig_controllers(cfg: RoomConfig) -> None:
    env = RoomThermalEnv(cfg)
    means = {s: env.reset(seed=s)[1]["t_out_mean"] for s in SEEDS}
    picks = [
        ("冷天", min(SEEDS, key=lambda s: abs(means[s] - 1.0))),
        ("热天", min(SEEDS, key=lambda s: abs(means[s] - 30.0))),
    ]
    thermostat = ThermostatController(prestart_hours=0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.7], "hspace": 0.12, "wspace": 0.14})
    print("\nCONTROLLER FIGURE")
    for col, (label, seed) in enumerate(picks):
        episodes = {
            "不开机": (run_episode(env, seed, policy=lambda _e, _o: 0.0, record=True), ps.OFF, (0, (4, 2.5)), 1.6),
            "温控器，8 点开机": (run_episode(env, seed, policy=lambda e, _o: thermostat.act(e), record=True), ps.THERMOSTAT, "-", 1.8),
            "理论最优": (run_episode(env, seed, oracle=True, record=True), ps.ORACLE, "-", 2.3),
        }
        top, bottom = axes[0, col], axes[1, col]
        oracle = episodes["理论最优"][0]
        hours = oracle["trajectory"]["hour"] + cfg.dt_hours
        top.fill_between([cfg.occupied_start_hour, cfg.occupied_end_hour], cfg.comfort_low, cfg.comfort_high,
                         color=ps.COMFORT_SOFT, lw=0, label="有人时的舒适区（8–18 点）")
        top.plot(hours, oracle["trajectory"]["t_out"], color=ps.OUTDOOR, lw=1.1, alpha=0.8, label="室外温度")
        for name, (ep, color, ls, lw) in episodes.items():
            tr = ep["trajectory"]
            top.plot(hours, tr["t_air"], color=color, ls=ls, lw=lw, label=f"空气温度：{name}")
            if name != "不开机":
                bottom.step(hours, tr["hvac_action"], color=color, lw=1.2 if name != "理论最优" else 2.0, where="pre",
                            alpha=0.85 if name != "理论最优" else 1.0, zorder=3 if name == "理论最优" else 2)
            print(f"  {label} (mean outdoor {ep['t_out_mean']:.1f} C) {name}: energy {ep['energy_kwh']:.2f}, discomfort {ep['discomfort_kh']:.2f}, air at 8:00 {tr['t_air'][31]:.1f} C")
        top.plot(hours, oracle["trajectory"]["t_mass"], color=ps.ORACLE, ls=(0, (1, 1.6)), lw=1.4, label="墙体温度：理论最优")
        otr = oracle["trajectory"]
        first_on = np.flatnonzero(np.abs(otr["hvac_action"]) > 0.05)[0] * cfg.dt_hours
        print(f"  {label} oracle first switches on at {first_on:.2f} h")
        for end_hour in (9.0, 17.75):
            k = int(round(end_hour / cfg.dt_hours)) - 1
            print(f"  {label} oracle, step ending {end_hour:5.2f} h: action {otr['hvac_action'][k]:+.2f}, air {otr['t_air'][k]:.2f} C, "
                  f"walls {otr['t_mass'][k]:.2f} C, outdoor {otr['t_out'][k]:.1f} C, sun {env.q_solar[k]:.0f} W")
        ps.panel_title(top, "ab"[col], f"{label}：室外日均 {oracle['t_out_mean']:.0f} °C")
        top.set_xlim(0, 24)
        top.grid(axis="x", visible=False)
        bottom.axhspan(0, 1.12, color=ps.HEAT_SOFT, alpha=0.55, lw=0, zorder=0)
        bottom.axhspan(-1.12, 0, color=ps.COOL_SOFT, alpha=0.7, lw=0, zorder=0)
        bottom.text(0.3, 0.92, "制热", color=ps.HEAT, fontsize=8.5, va="top")
        bottom.text(0.3, -0.92, "制冷", color=ps.COOL, fontsize=8.5, va="bottom")
        bottom.axhline(0, color=ps.MUTED, lw=0.6)
        bottom.set_ylim(-1.12, 1.12)
        bottom.set_yticks([-1, 0, 1])
        bottom.set_xlabel("一天中的时刻")
        bottom.set_xticks(range(0, 25, 4))
        bottom.grid(axis="x", visible=False)
        for ax in (top, bottom):
            for h in (cfg.occupied_start_hour, cfg.occupied_end_hour):
                ax.axvline(h, color=ps.FAINT, lw=0.8, zorder=0)
    axes[0, 0].set_ylabel("温度（°C）")
    axes[1, 0].set_ylabel("动作 a")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.84)
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.085, 0.875), ncol=3, fontsize=9.5,
               columnspacing=1.6, handlelength=2.4)
    ps.header(fig, "冷天和热天：不开机、温控器与理论最优", x=0.09, top=0.965)
    ps.save(fig, FIG / "env_controllers.png")


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
