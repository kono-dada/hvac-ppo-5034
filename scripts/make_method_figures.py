"""Figure and numbers for report section 2 (MDP, algorithm, training pipeline).

Usage:  PYTHONPATH=. .venv/bin/python scripts/make_method_figures.py
Output: docs/report/figures/method_loop.png, plus every number quoted in
        docs/report/zh/02_method.md printed to stdout.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from hvac import plotting as ps
from hvac.env import RoomConfig, RoomThermalEnv
from hvac.train import DEFAULT_PPO, N_ENVS

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures"
ps.use_style()

TOTAL_STEPS = 500_000
EVAL_EVERY = 20_480


def print_budget() -> None:
    p = DEFAULT_PPO
    per_update = N_ENVS * p["n_steps"]
    n_updates = int(np.ceil(TOTAL_STEPS / per_update))
    minibatches = per_update // p["batch_size"]
    print("== training budget ==")
    print(f"samples per update: {N_ENVS} x {p['n_steps']} = {per_update} ({per_update / 96:.1f} days)")
    print(f"updates for {TOTAL_STEPS}: {n_updates} -> {n_updates * per_update} env steps "
          f"({n_updates * per_update / 96:.0f} days)")
    print(f"minibatches per epoch: {minibatches}; gradient steps per update: {minibatches * p['n_epochs']}")
    print(f"each sample used {p['n_epochs']} times")
    print(f"evaluation every {EVAL_EVERY} steps = {EVAL_EVERY // per_update} updates")


def print_horizons() -> None:
    g, lam = DEFAULT_PPO["gamma"], DEFAULT_PPO["gae_lambda"]
    print("== how far ahead rewards reach ==")
    for name, w in (("gamma", g), ("gamma*lambda", g * lam)):
        half = np.log(0.5) / np.log(w)
        print(f"{name}={w:.4f}: half weight after {half:.1f} steps ({half / 4:.1f} h); "
              f"weight at 19 steps {w**19:.2f}; at 96 steps {w**96:.2f}")
    for lam_ in (0.8, 0.9, 0.95, 0.99):
        w = g * lam_
        print(f"  lambda={lam_}: gamma*lambda={w:.4f}, half after {np.log(0.5) / np.log(w):.1f} steps, "
              f"weight at 19 steps {w**19:.2f}")


def print_example_obs() -> None:
    """A concrete observation, so the reader sees actual numbers."""
    env = RoomThermalEnv()
    for seed in range(100_000, 100_200):
        _, info = env.reset(seed=seed)
        if info["t_out_mean"] < 3:
            break
    for _ in range(28):  # off until 7:00
        obs, *_ = env.step(np.array([0.0], dtype=np.float32))
    hour = env.hours[env.k]
    print(f"== example observation (seed {seed}, daily mean {info['t_out_mean']:.1f} C, heat pump off) ==")
    print(f"time {int(hour)}:{int(round((hour % 1) * 60)):02d}  T_air={env.x[0]:.2f}  T_mass={env.x[1]:.2f}  "
          f"T_out={env.t_out[env.k]:.2f}  Q_sun={env.q_solar[env.k]:.0f}")
    print("obs:", np.round(obs, 2))


def fig_loop() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.1, 4.9)
    ax.axis("off")
    accent = ps.HEAT

    def card(x, num, title, body, color, chip=None):
        y, w, h = 1.35, 3.3, 2.75
        ax.add_patch(FancyBboxPatch((x + 0.04, y - 0.06), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                    fc="#000000", ec="none", alpha=0.06, zorder=1))  # soft shadow
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc="white", ec=ps.FAINT,
                                    lw=1.0, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.62), w, 0.62, boxstyle="round,pad=0,rounding_size=0.18", fc=color,
                                    ec="none", zorder=3))
        ax.add_patch(plt.Rectangle((x, y + h - 0.62), w, 0.3, fc=color, ec="none", zorder=3))
        ax.add_patch(plt.Circle((x + 0.42, y + h - 0.31), 0.19, fc="white", ec="none", zorder=4))
        ax.text(x + 0.42, y + h - 0.31, num, ha="center", va="center", fontsize=11, weight="semibold", color=color, zorder=5)
        ax.text(x + 0.75, y + h - 0.31, title, ha="left", va="center", fontsize=12.5, weight="semibold", color="white", zorder=5)
        ax.text(x + w / 2, y + (h - 0.62) / 2 + (0.18 if chip else 0), body, ha="center", va="center", fontsize=10,
                linespacing=1.6, zorder=5)
        if chip:
            ax.text(x + w / 2, y + 0.3, chip, ha="center", va="center", fontsize=9.5, color=accent, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.5", fc=ps.HEAT_SOFT, ec="none"))

    def arrow(p, q, rad=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#57606a", ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=0, shrinkB=0))

    card(0.2, "1", "收集数据", "8 间房同时运行\n每间走 256 步（15 分钟/步）\n共 2048 步，约 21 天\n动作带随机噪声", ps.COOL)
    card(4.35, "2", "算优势", "价值网络给每个状态估值\n用 GAE 算每步的\n“比预期好多少”\n参数 γ=0.99, λ", "#c98a1b",
         chip="λ：GAE lambda")
    card(8.5, "3", "更新网络", "2048 步数据打乱\n分成 32 批，每批 64 步\n重复 n_epochs 遍\n裁剪幅度 ε", ps.COMFORT,
         chip="ε：clip range    n_epochs：遍数")
    arrow((3.58, 2.72), (4.27, 2.72))
    arrow((7.73, 2.72), (8.42, 2.72))
    arrow((10.15, 1.27), (1.85, 1.27), rad=-0.2)
    ax.text(6.0, 0.12, "丢掉旧数据，用新策略重新收集；重复约 245 轮（50 万步）", ha="center", fontsize=10.5, color="#57606a")
    ax.text(0.2, 4.62, "PPO 的一轮训练", ha="left", va="center", fontsize=13.5, weight="semibold")
    ax.text(2.25, 4.62, "（红字三个参数是本文消融的对象）", ha="left", va="center", fontsize=10.5, color=accent)
    ps.save(fig, FIG / "method_loop.png")
    print("wrote", FIG / "method_loop.png")


if __name__ == "__main__":
    print_budget()
    print_horizons()
    print_example_obs()
    fig_loop()
