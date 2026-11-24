"""Tables, figures and prediction verdicts for the main ablation.

Reads runs/ablation/<config>/seed<k>/ (written by scripts/run_ablation.py) and
judges every prediction listed in docs/report/zh/03_ablation.md, section 3.4,
with the rules of section 3.1 (implemented in hvac/stats.py).

Usage:  python scripts/analyze_ablation.py
Writes: docs/report/data/ablation_summary.csv     one row per configuration
        docs/report/data/ablation_summary.md      the main results table of section 3.6
        docs/report/data/ablation_verdicts.csv    one row per prediction
        docs/report/data/ablation_verdicts.md     the same, as a markdown table
        docs/report/figures/ablation_*.png
Prints: every number quoted in section 3.6 (the time-of-day split for lambda
        comes from scripts/lambda_time_of_day.py)
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

from hvac.controllers import ThermostatController, tuned_thermostat
from hvac.env import RoomConfig
from hvac.evaluation import eval_seeds, evaluate_controller, mean_metrics
from hvac.stats import judge, mean_ci, sd_ratio_test, welch_ci

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "ablation"
DATA = ROOT / "docs" / "report" / "data"
FIG = ROOT / "docs" / "report" / "figures"

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

WEIGHT = RoomConfig().comfort_weight
TOTAL_STEPS = 204_800
SECOND_HALF = TOTAL_STEPS // 2  # rollback is measured on evaluation points after this many steps

GROUPS = {
    "epochs": ("n_epochs", ["epochs_1", "epochs_3", "baseline", "epochs_30"], [1, 3, 10, 30]),
    "clip": ("clip range", ["clip_0.05", "clip_0.1", "baseline", "clip_0.3", "clip_0.5"], [0.05, 0.1, 0.2, 0.3, 0.5]),
    "lambda": ("GAE λ", ["lambda_0.8", "lambda_0.9", "baseline", "lambda_0.99", "lambda_1"], [0.8, 0.9, 0.95, 0.99, 1.0]),
}
CONFIGS = ["baseline"] + [c for _, names, _ in GROUPS.values() for c in names if c != "baseline"]

METRICS = {  # column -> (label, delta used by "same"/"not_higher"/"much_higher")
    "cost": ("成本", 0.1),
    "energy_kwh": ("耗电量", 0.1),
    "discomfort_kh": ("不舒适度", 0.05),
    "rollback": ("回退", 0.1),
    "seed_sd": ("种子间差异", None),
    "approx_kl": ("approx_kl", "25%"),
    "clip_fraction": ("clip_fraction", "25%"),
}
INF = float("inf")

# (config, metric, kind, bounds, wording in 03_ablation.md). Mirrors the table at the end of section 3.4.
PREDICTIONS = [
    ("epochs_1", "cost", "range", (6.5, 8.5), "6.5 到 8.5"),
    ("epochs_1", "energy_kwh", "much_higher", None, "明显更高"),
    ("epochs_1", "discomfort_kh", "range", (-INF, 0.05), "接近 0（低于 0.05）"),
    ("epochs_1", "rollback", "not_higher", None, "曲线平滑（不高于基线）"),
    ("epochs_1", "seed_sd", "sd_larger", None, "更大"),
    ("epochs_3", "cost", "range", (5.4, 6.4), "5.4 到 6.4"),
    ("epochs_3", "energy_kwh", "higher", None, "更高"),
    ("epochs_3", "discomfort_kh", "lower", None, "更低"),
    ("epochs_3", "rollback", "not_higher", None, "曲线平滑（不高于基线）"),
    ("epochs_3", "seed_sd", "sd_larger", None, "更大"),
    ("epochs_30", "cost", "range", (4.6, 5.0), "4.6 到 5.0"),
    ("epochs_30", "energy_kwh", "lower", None, "略低"),
    ("epochs_30", "discomfort_kh", "higher", None, "更高"),
    ("epochs_30", "approx_kl", "much_higher", None, "明显更高"),
    ("epochs_30", "clip_fraction", "much_higher", None, "明显更高（正文）"),
    ("epochs_30", "rollback", "higher", None, "更大"),
    ("clip_0.05", "cost", "range", (5.2, 6.0), "5.2 到 6.0"),
    ("clip_0.05", "energy_kwh", "higher", None, "更高"),
    ("clip_0.05", "discomfort_kh", "not_higher", None, "不高于基线"),
    ("clip_0.05", "approx_kl", "range", (-INF, 0.003), "低于 0.003"),
    ("clip_0.05", "clip_fraction", "range", (0.30, INF), "高于 30%（正文）"),
    ("clip_0.05", "rollback", "not_higher", None, "曲线平滑（不高于基线）"),
    ("clip_0.1", "cost", "range", (4.8, 5.1), "4.8 到 5.1"),
    ("clip_0.1", "energy_kwh", "higher", None, "略高"),
    ("clip_0.1", "discomfort_kh", "same", None, "相当"),
    ("clip_0.1", "rollback", "same", None, "相当"),
    ("clip_0.3", "cost", "range", (4.6, 4.9), "4.6 到 4.9"),
    ("clip_0.3", "energy_kwh", "same", None, "相当"),
    ("clip_0.3", "discomfort_kh", "same", None, "相当"),
    ("clip_0.3", "approx_kl", "higher", None, "略高"),
    ("clip_0.3", "approx_kl", "range", "2x", "不到基线 2 倍（正文）"),
    ("clip_0.5", "cost", "range", (4.6, 4.9), "4.6 到 4.9"),
    ("clip_0.5", "energy_kwh", "same", None, "相当"),
    ("clip_0.5", "discomfort_kh", "same", None, "相当"),
    ("clip_0.5", "approx_kl", "higher", None, "更高（正文）"),
    ("clip_0.5", "approx_kl", "range", "2x", "不到基线 2 倍"),
    ("clip_0.5", "rollback", "higher", None, "更大（把握不大）"),
    ("clip_0.5", "seed_sd", "sd_larger", None, "更大（正文，把握不大）"),
    ("lambda_0.8", "cost", "range", (4.5, 5.0), "4.5 到 5.0"),
    ("lambda_0.8", "energy_kwh", "same", None, "相当"),
    ("lambda_0.8", "discomfort_kh", "range", (-INF, 0.3), "低于 0.3"),
    ("lambda_0.8", "rollback", "same", None, "相当"),
    ("lambda_0.9", "cost", "range", (4.6, 4.8), "4.6 到 4.8"),
    ("lambda_0.9", "energy_kwh", "same", None, "相当"),
    ("lambda_0.9", "discomfort_kh", "same", None, "相当"),
    ("lambda_0.9", "rollback", "same", None, "相当"),
    ("lambda_0.99", "cost", "range", (4.7, 5.1), "4.7 到 5.1"),
    ("lambda_0.99", "energy_kwh", "higher", None, "略高"),
    ("lambda_0.99", "rollback", "higher", None, "更大"),
    ("lambda_0.99", "seed_sd", "sd_larger", None, "更大（正文）"),
    ("lambda_1", "cost", "range", (4.8, 5.3), "4.8 到 5.3"),
    ("lambda_1", "energy_kwh", "higher", None, "更高"),
    ("lambda_1", "rollback", "higher", None, "更大"),
    ("lambda_1", "seed_sd", "sd_larger", None, "更大"),
]


# ---------------------------------------------------------------- loading
def rollback(curve: pd.DataFrame) -> float:
    """Largest rise of the evaluation cost above its running best, over the second half of training."""
    c = curve["cost"].to_numpy()
    t = curve["timesteps"].to_numpy()
    rises = [c[k] - c[:k].min() for k in range(1, len(c)) if t[k] > SECOND_HALF]
    return max(0.0, max(rises)) if rises else float("nan")


def load(config: str) -> list[dict]:
    runs = []
    for run in sorted((RUNS / config).glob("seed*"), key=lambda p: int(p.name[4:])):
        if not (run / "final_eval.json").exists():
            continue
        final = json.loads((run / "final_eval.json").read_text())["mean"]
        curve = pd.read_csv(run / "eval_curve.csv")
        curve["cost"] = curve["energy_kwh"] + WEIGHT * curve["discomfort_kh"]
        progress = pd.read_csv(run / "progress.csv")
        runs.append(
            {
                "config": config,
                "seed": int(run.name[4:]),
                "cost": final["energy_kwh"] + WEIGHT * final["discomfort_kh"],
                "energy_kwh": final["energy_kwh"],
                "discomfort_kh": final["discomfort_kh"],
                "violation_pct": final["violation_pct"],
                "rollback": rollback(curve),
                "approx_kl": float(progress["train/approx_kl"].mean()),
                "clip_fraction": float(progress["train/clip_fraction"].mean()),
                "train_seconds": final["train_seconds"],
                "curve": curve,
                "progress": progress,
            }
        )
    return runs


def references() -> dict[str, dict]:
    cfg, seeds = RoomConfig(), eval_seeds(100)
    out = {}
    for name, ctrl in [("温控器（调过参数）", tuned_thermostat()), ("未调参的温控器", ThermostatController(prestart_hours=5)),
                       ("理论最优", "oracle")]:
        m = mean_metrics(evaluate_controller(ctrl, cfg, seeds))
        m["cost"] = m["energy_kwh"] + WEIGHT * m["discomfort_kh"]
        out[name] = m
    return out


# ---------------------------------------------------------------- tables
def summary_table(runs: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        rs = runs.get(config, [])
        if not rs:
            continue
        row = {"config": config, "seeds": len(rs)}
        for key in ("cost", "energy_kwh", "discomfort_kh", "violation_pct", "rollback", "approx_kl", "clip_fraction"):
            v = np.array([r[key] for r in rs])
            m, lo, hi = mean_ci(v)
            row.update({key: m, f"{key}_sd": v.std(ddof=1) if len(v) > 1 else np.nan, f"{key}_ci_lo": lo, f"{key}_ci_hi": hi})
        row["train_seconds"] = float(np.mean([r["train_seconds"] for r in rs]))
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(x: float, metric: str) -> str:
    if metric == "clip_fraction":
        return f"{100 * x:.1f}%"
    if metric == "approx_kl":
        return f"{x:.4f}"
    return f"{x:.2f}" if abs(x) >= 0.995 or metric == "cost" else f"{x:.3f}"


def verdict_table(runs: dict[str, list[dict]]) -> pd.DataFrame:
    base = runs["baseline"]
    rows = []
    for config, metric, kind, bounds, wording in PREDICTIONS:
        rs = runs.get(config, [])
        if len(rs) < 2:
            continue
        sample_key = "cost" if metric == "seed_sd" else metric
        x = np.array([r[sample_key] for r in rs])
        b = np.array([r[sample_key] for r in base])
        delta = METRICS[metric][1]
        if delta == "25%":
            delta = 0.25 * b.mean()
        if bounds == "2x":
            bounds = (-INF, 2.0 * b.mean())
        res = judge(kind, x, b, bounds=bounds, delta=delta)
        if kind == "range":
            observed = f"{_fmt(res['value'], metric)} [{_fmt(res['ci95'][0], metric)}, {_fmt(res['ci95'][1], metric)}]"
        elif kind == "sd_larger":
            p = "p < 0.001" if res["p"] < 0.001 else f"p = {res['p']:.3f}"
            observed = f"标准差之比 {res['value']:.2f}，{p}"
        else:
            lo, hi = res["ci90"] if kind in ("same", "not_higher", "not_lower") else res["ci95"]
            tag = "90%" if kind in ("same", "not_higher", "not_lower") else "95%"
            observed = f"差 {res['value']:+.3g}，{tag} 区间 [{lo:+.3g}, {hi:+.3g}]"
        rows.append(
            {
                "config": config,
                "metric": METRICS[metric][0],
                "prediction": wording,
                "rule": kind,
                "observed": observed,
                "verdict": res["verdict"],
                "seeds": len(rs),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def _curve_stats(rs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = rs[0]["curve"]["timesteps"].to_numpy()
    y = np.stack([r["curve"][key].to_numpy() for r in rs])
    return t, y.mean(0), y.std(0, ddof=1) if len(rs) > 1 else np.zeros_like(t, dtype=float)


def fig_learning_curves(runs: dict[str, list[dict]], refs: dict[str, dict]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    panels = [("cost", "成本", (3.0, 10.0)), ("energy_kwh", "耗电量（kWh/天）", (3.0, 10.0)), ("discomfort_kh", "不舒适度（K·h/天）", (0.0, 1.0))]
    for row, (group, (label, names, values)) in enumerate(GROUPS.items()):
        colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(names)))
        for col, (key, title, ylim) in enumerate(panels):
            ax = axes[row, col]
            for name, value, color in zip(names, values, colors):
                rs = runs.get(name, [])
                if not rs:
                    continue
                t, m, s = _curve_stats(rs, key)
                ax.plot(t / 1e3, m, color="k" if name == "baseline" else color, lw=2.2 if name == "baseline" else 1.6,
                        label=f"{label} = {value:g}" + ("（基线）" if name == "baseline" else ""))
                ax.fill_between(t / 1e3, m - s, m + s, color="k" if name == "baseline" else color, alpha=0.12, lw=0)
            for ref, style in (("温控器（调过参数）", "--"), ("理论最优", ":")):
                ax.axhline(refs[ref][key], color="grey", ls=style, lw=1.2, label=ref if row == 0 and col == 0 else None)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(title)
            if row == 2:
                ax.set_xlabel("训练步数（千步）")
        axes[row, 0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("训练过程中在 100 个测试日上的评估（线为各种子的均值，阴影为 ±1 个标准差）")
    fig.tight_layout()
    fig.savefig(FIG / "ablation_learning_curves.png", dpi=150)
    plt.close(fig)


def fig_final(runs: dict[str, list[dict]], refs: dict[str, dict]) -> None:
    pred = {(c, m): b for c, m, k, b, _ in PREDICTIONS if k == "range" and m == "cost"}
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    panels = [("cost", "成本"), ("energy_kwh", "耗电量（kWh/天）"), ("discomfort_kh", "不舒适度（K·h/天）")]
    for row, (group, (label, names, values)) in enumerate(GROUPS.items()):
        for col, (key, title) in enumerate(panels):
            ax = axes[row, col]
            xs = np.arange(len(names))
            for i, name in enumerate(names):
                rs = runs.get(name, [])
                if key == "cost" and (name, "cost") in pred:
                    lo, hi = pred[(name, "cost")]
                    ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, color="C1", alpha=0.25, lw=0,
                                               label="预测区间" if i == 0 or (name == names[1] and (names[0], "cost") not in pred) else None))
                if not rs:
                    continue
                v = np.array([r[key] for r in rs])
                ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=10, color="C0", alpha=0.6)
                m, lo, hi = mean_ci(v)
                ax.errorbar(i, m, yerr=[[m - lo], [hi - m]] if len(v) > 1 else None, fmt="o", color="k", capsize=4, ms=6)
            ax.axhline(refs["温控器（调过参数）"][key], color="grey", ls="--", lw=1.2, label="温控器（调过参数）")
            ax.set_xticks(xs, [f"{v:g}" for v in values])
            ax.set_xlabel(label)
            ax.grid(alpha=0.3, axis="y")
            if row == 0:
                ax.set_title(title)
        handles, labels = axes[row, 0].get_legend_handles_labels()
        axes[row, 0].legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=8.5)
    fig.suptitle("训练结束时的结果：黑点为均值和 95% 置信区间，蓝点为各个种子，橙色为预测的成本区间")
    fig.tight_layout()
    fig.savefig(FIG / "ablation_final.png", dpi=150)
    plt.close(fig)


def fig_update_size(runs: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, group in enumerate(("clip", "epochs")):
        label, names, values = GROUPS[group]
        colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(names)))
        for row, key in enumerate(("train/approx_kl", "train/clip_fraction")):
            ax = axes[row, col]
            for name, value, color in zip(names, values, colors):
                rs = runs.get(name, [])
                if not rs:
                    continue
                y = np.stack([r["progress"][key].to_numpy() for r in rs]).mean(0)
                it = rs[0]["progress"]["time/iterations"].to_numpy()
                ax.plot(it, y, color="k" if name == "baseline" else color, lw=2.0 if name == "baseline" else 1.4,
                        label=f"{label} = {value:g}" + ("（基线）" if name == "baseline" else ""))
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
            ax.set_ylabel(key.split("/")[1])
            if row == 1:
                ax.set_xlabel("第几轮更新")
            if row == 0:
                ax.legend(fontsize=8.5)
    fig.suptitle("每轮更新的幅度（各种子的均值）。epochs = 1、3 的 approx_kl 偏小，见 3.1 节")
    fig.tight_layout()
    fig.savefig(FIG / "ablation_update_size.png", dpi=150)
    plt.close(fig)


def summary_markdown(summary: pd.DataFrame, runs: dict[str, list[dict]]) -> str:
    base = np.array([r["cost"] for r in runs["baseline"]])
    md = ["| 配置 | 成本（均值 ± 标准差） | 与基线之差（95% 区间） | 耗电 kWh | 不舒适度 K·h | 违规比例 | 回退 | `approx_kl` | `clip_fraction` | 单次训练（秒） |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in summary.itertuples():
        if r.config == "baseline":
            diff = "—"
        else:
            d, lo, hi = welch_ci([x["cost"] for x in runs[r.config]], base)
            diff = f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        kl = f"{r.approx_kl:.4f}" if r.config not in ("epochs_1", "epochs_3") else f"（{r.approx_kl:.4f}）"
        md.append(f"| {LABELS[r.config]} | {r.cost:.2f} ± {r.cost_sd:.2f} | {diff} | {r.energy_kwh:.2f} | {r.discomfort_kh:.3f} | "
                  f"{r.violation_pct:.1f}% | {r.rollback:.2f} | {kl} | {100 * r.clip_fraction:.1f}% | {r.train_seconds:.0f} |")
    return "\n".join(md) + "\n"


def mechanism_numbers(runs: dict[str, list[dict]]) -> None:
    """Numbers quoted in the analysis of section 3.6."""
    print("\nUPDATE SIZE VS CLIP RANGE (typical |ratio - 1| ~ sqrt(2 approx_kl))")
    base_kl = np.mean([r["approx_kl"] for r in runs["baseline"]])
    for name, eps in (("clip_0.05", 0.05), ("clip_0.1", 0.1), ("baseline", 0.2), ("clip_0.3", 0.3), ("clip_0.5", 0.5)):
        kl = np.mean([r["approx_kl"] for r in runs[name]])
        dev = np.sqrt(2 * kl)
        print(f"  eps {eps:4.2f}: approx_kl {kl:.4f} ({kl / base_kl:.1f} x baseline), typical deviation {dev:.3f} = {dev / eps:.2f} x eps")
    kl30 = np.mean([r["approx_kl"] for r in runs["epochs_30"]])
    print(f"  epochs 30: approx_kl {kl30:.4f} ({kl30 / base_kl:.1f} x baseline)")
    print("\nLARGEST SINGLE-ROUND approx_kl")
    for name in CONFIGS:
        mx = [r["progress"]["train/approx_kl"].max() for r in runs[name]]
        big = sum(int((r["progress"]["train/approx_kl"] > 0.1).sum()) for r in runs[name])
        print(f"  {name:12s} largest {max(mx):.3f}; rounds above 0.1 over all seeds: {big}")
    print("\nACTION NOISE sigma: end of training, and first update round below 0.3 / 0.1 (mean over seeds)")
    for name in CONFIGS:
        s = np.stack([r["progress"]["train/std"].to_numpy() for r in runs[name]]).mean(0)
        it = runs[name][0]["progress"]["time/iterations"].to_numpy()
        first = [int(it[np.argmax(s < v)]) if (s < v).any() else None for v in (0.3, 0.1)]
        print(f"  {name:12s} sigma at end {s[-1]:.3f}; below 0.3 at round {first[0]}, below 0.1 at round {first[1]}")
    print("\nBASELINE AT THE SAME sigma: mean baseline evaluation points whose sigma is within 10% of each config's final sigma")
    pts = np.concatenate([np.c_[r["curve"]["policy_std"], r["curve"]["cost"]] for r in runs["baseline"]])
    for name in CONFIGS:
        s = np.mean([r["curve"]["policy_std"].iloc[-1] for r in runs[name]])
        m = (pts[:, 0] > s / 1.1) & (pts[:, 0] < s * 1.1)
        same = f"{pts[m, 1].mean():.2f} (n={m.sum()})" if m.sum() >= 3 else "no baseline point"
        print(f"  {name:12s} final sigma {s:.3f}, final cost {np.mean([r['cost'] for r in runs[name]]):.2f}; baseline at that sigma: {same}")
    print("\nMEAN EVALUATION COST DURING TRAINING")
    for name in CONFIGS:
        t = list(runs[name][0]["curve"]["timesteps"])
        y = np.stack([r["curve"]["cost"].to_numpy() for r in runs[name]]).mean(0)
        print(f"  {name:12s} " + "  ".join(f"{k // 1000}k {y[t.index(k)]:.2f}" for k in (49_152, 98_304, 147_456, 204_800)))
    print("\nSEEDS BELOW THE TUNED THERMOSTAT (4.48)")
    for name in CONFIGS:
        c = np.array([r["cost"] for r in runs[name]])
        print(f"  {name:12s} {int((c < 4.48).sum())} of {len(c)}; best seed {c.min():.2f}")

    base = runs["baseline"]
    bc = np.array([r["cost"] for r in base])
    first5 = np.array([r["cost"] for r in base if r["seed"] < 5])
    print("\nBASELINE: seeds 0-4 (used for the predictions) vs all 10")
    print(f"  seeds 0-4 {first5.mean():.2f} +- {first5.std(ddof=1):.3f}; all 10 {bc.mean():.2f} +- {bc.std(ddof=1):.3f}; "
          f"per seed {np.round(bc, 2).tolist()}")
    worst = int(np.argmax(bc))
    print(f"  without seed {base[worst]['seed']}: sd {np.delete(bc, worst).std(ddof=1):.3f}")

    t = base[0]["curve"]["timesteps"].to_numpy()
    yb = np.stack([r["curve"]["cost"].to_numpy() for r in base]).mean(0)
    print("\nWHERE THE BASELINE WAS WHEN IT FIRST REACHED EACH CONFIG'S FINAL COST (mean curves, linear interpolation)")
    for name in CONFIGS:
        f = np.mean([r["cost"] for r in runs[name]])
        if not (yb <= f).any():
            print(f"  {name:12s} final {f:.2f}: the baseline never got this low")
            continue
        k = int(np.argmax(yb <= f))
        x = t[k] if k == 0 else t[k - 1] + (yb[k - 1] - f) / (yb[k - 1] - yb[k]) * (t[k] - t[k - 1])
        print(f"  {name:12s} final {f:.2f}: baseline at {x / 1e3:.0f}k steps ({x / TOTAL_STEPS:.0%} of training)")

    print("\nGRADIENT UPDATES: lr 3e-4, 32 minibatches per epoch. ln(sigma) can fall by at most about lr per update")
    lr, per_epoch, rounds = 3e-4, 2048 // 64, TOTAL_STEPS // 2048
    for name in ("baseline", "epochs_1", "epochs_3", "epochs_30"):
        n_ep = json.loads((RUNS / name / "seed0" / "spec.json").read_text())["ppo"]["n_epochs"]
        updates = rounds * n_ep * per_epoch
        sig = np.mean([r["curve"]["policy_std"].iloc[-1] for r in runs[name]])
        same = TOTAL_STEPS * updates / (rounds * 10 * per_epoch)
        then = f"cost {np.interp(same, t, yb):.2f}" if same <= TOTAL_STEPS else "beyond the run"
        print(f"  {name:10s} {updates:6d} updates; final sigma {sig:.3f}; ln-sigma drop = {np.log(1 / sig) / (lr * updates):.2f} of the bound; "
              f"baseline after the same number of updates ({same / 1e3:.0f}k steps): {then}")

    print("\nEVALUATION-TO-EVALUATION CHANGE OF COST, second half of training (mean |change| over 8,192 steps)")
    for name in CONFIGS:
        v = []
        for r in runs[name]:
            c, tt = r["curve"]["cost"].to_numpy(), r["curve"]["timesteps"].to_numpy()
            v.append(np.mean(np.abs(np.diff(c)[tt[1:] > SECOND_HALF])))
        print(f"  {name:12s} {np.mean(v):.3f}   (rollback {np.mean([r['rollback'] for r in runs[name]]):.3f})")

    print("\nDIFFERENCES FROM THE BASELINE NOT COVERED BY A PREDICTION (95% Welch intervals)")
    for name, key in (("lambda_0.8", "energy_kwh"), ("lambda_0.8", "discomfort_kh"), ("lambda_0.9", "energy_kwh"),
                      ("lambda_0.99", "discomfort_kh"), ("lambda_1", "discomfort_kh"), ("clip_0.3", "rollback"),
                      ("epochs_30", "discomfort_kh"), ("clip_0.5", "cost")):
        d, lo, hi = welch_ci([r[key] for r in runs[name]], [r[key] for r in base])
        print(f"  {name:12s} {key:14s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nHALF-WIDTH OF THE 90% INTERVAL OF THE DIFFERENCE FROM THE BASELINE (what 'same' has to fit inside delta)")
    for key, delta in (("cost", 0.1), ("energy_kwh", 0.1), ("discomfort_kh", 0.05)):
        hw = [(lambda d, lo, hi: (hi - lo) / 2)(*welch_ci([r[key] for r in runs[n]], [r[key] for r in base], 0.90))
              for n in CONFIGS if n not in ("baseline", "epochs_1")]
        print(f"  {key:14s} median {np.median(hw):.3f} (range {min(hw):.3f} to {max(hw):.3f}); delta {delta}")

    print("\nCLIP 0.5 PER SEED")
    for r in runs["clip_0.5"]:
        c = r["curve"]["cost"].to_numpy()
        print(f"  seed {r['seed']}: final {r['cost']:.2f}, best evaluation {c.min():.2f}, rollback {r['rollback']:.2f}, "
              f"largest approx_kl {r['progress']['train/approx_kl'].max():.2f}")
    for name in ("baseline", "clip_0.5"):
        p = [r["progress"] for r in runs[name]]
        early = np.mean([q.loc[q["time/iterations"] <= 30, "train/approx_kl"].mean() for q in p])
        late = np.mean([q.loc[q["time/iterations"] > 70, "train/approx_kl"].mean() for q in p])
        print(f"  {name:10s} approx_kl, rounds 2-30: {early:.4f}; rounds 71-100: {late:.4f}")

    print("\nBASELINE sigma after 10 and 30 update rounds:",
          np.round(np.mean([[r["progress"].loc[r["progress"]["time/iterations"] == k, "train/std"].item() for k in (11, 31)]
                            for r in base], 0), 3).tolist())

    print("\nSEED SPREAD OF FINAL COST vs BASELINE, every configuration (two-sided F test)")
    for name in CONFIGS[1:]:
        ratio, p = sd_ratio_test([r["cost"] for r in runs[name]], bc)
        print(f"  {name:12s} sd ratio {ratio:.2f}, p {p:.3f}{'  *' if p < 0.05 else ''}")


LABELS = {"baseline": "基线", "epochs_1": "epochs = 1", "epochs_3": "epochs = 3", "epochs_30": "epochs = 30",
          "clip_0.05": "clip = 0.05", "clip_0.1": "clip = 0.1", "clip_0.3": "clip = 0.3", "clip_0.5": "clip = 0.5",
          "lambda_0.8": "lambda = 0.8", "lambda_0.9": "lambda = 0.9", "lambda_0.99": "lambda = 0.99", "lambda_1": "lambda = 1.0"}


def fig_tradeoff(runs: dict[str, list[dict]], refs: dict[str, dict]) -> None:
    """Final energy vs discomfort of every configuration, on top of the baseline's own training path."""
    base = runs["baseline"]
    t = base[0]["curve"]["timesteps"].to_numpy()
    e = np.stack([r["curve"]["energy_kwh"].to_numpy() for r in base]).mean(0)
    d = np.stack([r["curve"]["discomfort_kh"].to_numpy() for r in base]).mean(0)
    keep = t >= 16_384
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(e[keep], d[keep], color="k", lw=1.5, marker=".", ms=5, label="基线的训练过程（每 8,192 步一个点）")
    for k in (24_576, 49_152, 98_304, 147_456):
        i = list(t).index(k)
        ax.annotate(f"{k // 1000}k 步", (e[i], d[i]), textcoords="offset points", xytext=(4, 6), fontsize=8.5, color="k")
    markers = {"epochs": "s", "clip": "^", "lambda": "o"}
    for group, (label, names, values) in GROUPS.items():
        colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(names)))
        for name, value, color in zip(names, values, colors):
            if name == "baseline" or not runs.get(name):
                continue
            ee = np.array([r["energy_kwh"] for r in runs[name]])
            dd = np.array([r["discomfort_kh"] for r in runs[name]])
            me, le, he = mean_ci(ee)
            md, ld, hd = mean_ci(dd)
            ax.errorbar(me, md, xerr=[[me - le], [he - me]], yerr=[[md - ld], [hd - md]], fmt=markers[group], color=color,
                        ms=8, capsize=3, mec="k", mew=0.6, label=f"{label} = {value:g}")
    be = np.array([r["energy_kwh"] for r in base]); bd = np.array([r["discomfort_kh"] for r in base])
    ax.plot(be.mean(), bd.mean(), "*", color="red", ms=16, mec="k", label="基线，训练结束")
    for name, style in (("温控器（调过参数）", "P"), ("理论最优", "X")):
        ax.plot(refs[name]["energy_kwh"], refs[name]["discomfort_kh"], style, color="grey", ms=11, mec="k", label=name)
    for c in (4.48, 4.73, 5.5):
        xs = np.linspace(3.3, 8.5, 50)
        ax.plot(xs, (c - xs) / WEIGHT, color="grey", lw=0.6, ls=":")
        ax.text(c - 2 * 0.0 - 0.02, 0.005, f"成本 {c}", fontsize=7.5, color="grey", ha="right", rotation=-60)
    ax.set_xlim(3.3, 8.5)
    ax.set_ylim(-0.01, 0.45)
    ax.set_xlabel("耗电量（kWh/天）")
    ax.set_ylabel("不舒适度（K·h/天）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.set_title("训练结束时各配置的位置（均值与 95% 区间）；虚线为等成本线")
    fig.tight_layout()
    fig.savefig(FIG / "ablation_tradeoff.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = {c: load(c) for c in CONFIGS}
    print("runs found: " + ", ".join(f"{c} {len(r)}" for c, r in runs.items()))
    refs = references()
    for name, m in refs.items():
        print(f"reference {name}: cost {m['cost']:.2f}, energy {m['energy_kwh']:.2f}, discomfort {m['discomfort_kh']:.3f}")

    summary = summary_table(runs)
    summary.to_csv(DATA / "ablation_summary.csv", index=False)
    if all(runs[c] for c in CONFIGS):
        (DATA / "ablation_summary.md").write_text(summary_markdown(summary, runs))
        print("\n" + summary_markdown(summary, runs))
        mechanism_numbers(runs)
    cols = ["config", "seeds", "cost", "cost_sd", "energy_kwh", "discomfort_kh", "violation_pct", "rollback", "approx_kl",
            "clip_fraction", "train_seconds"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[cols].round(4).to_string(index=False))

    verdicts = verdict_table(runs)
    if len(verdicts):
        verdicts.to_csv(DATA / "ablation_verdicts.csv", index=False)
        md = ["| 配置 | 指标 | 预测 | 实际 | 判定 |", "| --- | --- | --- | --- | --- |"]
        md += [f"| {LABELS[r.config]} | {r.metric} | {r.prediction} | {r.observed} | {r.verdict} |" for r in verdicts.itertuples()]
        (DATA / "ablation_verdicts.md").write_text("\n".join(md) + "\n")
        print("\n" + "\n".join(md))
        print("\nverdict counts:", verdicts["verdict"].value_counts().to_dict())
        print("undecided by rule:", verdicts.loc[verdicts["verdict"] == "无法判断", "rule"].value_counts().to_dict())
    print(f"\npredictions defined: {len(PREDICTIONS)}, judged: {len(verdicts)}")

    fig_learning_curves(runs, refs)
    fig_final(runs, refs)
    fig_update_size(runs)
    if all(runs[c] for c in CONFIGS):
        fig_tradeoff(runs, refs)
