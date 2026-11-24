"""Figure and numbers for report section 2 (MDP, algorithm, training pipeline).

Usage:  PYTHONPATH=. .venv/bin/python scripts/make_method_figures.py
Output: docs/report/figures/method_loop.png, plus every number quoted in
        docs/report/zh/02_method.md printed to stdout.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from hvac.env import RoomConfig, RoomThermalEnv
from hvac.train import DEFAULT_PPO, N_ENVS

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures"
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    def box(x, y, w, h, title, body, face):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.15",
                                    fc=face, ec="#444444", lw=1.2))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=12.5, weight="bold")
        ax.text(x + w / 2, y + (h - 0.55) / 2, body, ha="center", va="center", fontsize=10, linespacing=1.5)

    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#333333",
                                     connectionstyle=f"arc3,rad={rad}"))

    box(0.2, 1.6, 3.3, 2.5, "① 收集数据",
        "8 间房同时运行\n每间走 256 步（15 分钟/步）\n共 2048 步，约 21 天\n动作带随机噪声", "#dbe9f6")
    box(4.35, 1.6, 3.3, 2.5, "② 算优势",
        "价值网络给每个状态估值\n用 GAE 算每步的\n“比预期好多少”\n参数 γ=0.99, λ", "#fde9cc")
    box(8.5, 1.6, 3.3, 2.5, "③ 更新网络",
        "2048 步数据打乱\n分成 32 批，每批 64 步\n重复 n_epochs 遍\n裁剪幅度 ε", "#d9efd9")
    arrow((3.55, 2.85), (4.3, 2.85))
    arrow((7.7, 2.85), (8.45, 2.85))
    arrow((10.15, 1.55), (1.85, 1.55), rad=-0.18)
    ax.text(6.0, 0.25, "丢掉旧数据，用新策略重新收集；重复约 245 轮（50 万步）", ha="center", fontsize=10.5)
    ax.text(6.0, 4.55, "PPO 的一轮训练（红字三个参数是本文消融的对象）", ha="center", fontsize=12)
    for x, y, s in ((6.0, 1.82, "λ：GAE lambda"), (10.15, 1.82, "ε：clip range    n_epochs：遍数")):
        ax.text(x, y, s, ha="center", fontsize=9.5, color="#c62828")
    fig.tight_layout()
    fig.savefig(FIG / "method_loop.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "method_loop.png")


if __name__ == "__main__":
    print_budget()
    print_horizons()
    print_example_obs()
    fig_loop()
