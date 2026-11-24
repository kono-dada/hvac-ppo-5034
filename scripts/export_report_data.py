"""Export every number the Typst report prints.

The report (docs/report/typst/report.typ) reads docs/report/typst/data/report.json
instead of hand-copied numbers. This script only reads existing results and replays
saved policies on the fixed test days; it trains nothing and changes no result file.

Sources
    docs/report/data/ablation_summary.csv    per-configuration means (scripts/analyze_ablation.py)
    docs/report/data/ablation_verdicts.csv   the 54 pre-registered predictions and their verdicts
    runs/ablation/<config>/seed<k>/          per-seed results, learning curves, SB3 logs, saved policies
    hvac.controllers                         reference controllers, re-evaluated on the 100 test days

The definitions below repeat those of scripts/analyze_ablation.py (rollback, the
Adam bound on ln sigma, "where the baseline was") and scripts/lambda_time_of_day.py
(the split by time of day), so the numbers agree with docs/report/zh/03_ablation.md.

Usage:  PYTHONPATH=. .venv/bin/python scripts/export_report_data.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hvac.controllers import OffController, ThermostatController, tuned_thermostat  # noqa: E402
from hvac.env import RoomConfig, RoomThermalEnv  # noqa: E402
from hvac.evaluation import eval_seeds, evaluate_controller, mean_metrics, run_episode  # noqa: E402
from hvac.stats import HIT, MISS, PARTIAL, UNDECIDED, sd_ratio_test, welch_ci  # noqa: E402

DATA = ROOT / "docs" / "report" / "data"
RUNS = ROOT / "runs" / "ablation"
OUT = ROOT / "docs" / "report" / "typst" / "data" / "report.json"

# Table order: the baseline, then one block per hyperparameter.
GROUPS = [
    ("epochs", "n_epochs", 10, [("epochs_1", 1), ("epochs_3", 3), ("epochs_30", 30)]),
    ("clip", "clip_range", 0.2, [("clip_0.05", 0.05), ("clip_0.1", 0.1), ("clip_0.3", 0.3), ("clip_0.5", 0.5)]),
    ("lambda", "gae_lambda", 0.95, [("lambda_0.8", 0.8), ("lambda_0.9", 0.9), ("lambda_0.99", 0.99), ("lambda_1", 1.0)]),
]
CONFIGS = ["baseline"] + [name for _, _, _, members in GROUPS for name, _ in members]
# Predicted final cost (docs/report/zh/03_ablation.md, section 3.4).
PREDICTED_COST = {
    "epochs_1": (6.5, 8.5), "epochs_3": (5.4, 6.4), "epochs_30": (4.6, 5.0),
    "clip_0.05": (5.2, 6.0), "clip_0.1": (4.8, 5.1), "clip_0.3": (4.6, 4.9), "clip_0.5": (4.6, 4.9),
    "lambda_0.8": (4.5, 5.0), "lambda_0.9": (4.6, 4.8), "lambda_0.99": (4.7, 5.1), "lambda_1": (4.8, 5.3),
}  # fmt: skip
VERDICT_KEYS = {HIT: "hit", PARTIAL: "partial", MISS: "miss", UNDECIDED: "undecided"}
# Metric names as written in ablation_verdicts.csv -> keys used in the report.
HEADLINE_METRICS = {"成本": "cost", "耗电量": "energy_kwh", "不舒适度": "discomfort_kh"}
# How a pre-registered prediction is worded in the report's table.
RULE_LABELS = {"much_higher": "much higher", "higher": "higher", "lower": "lower", "same": "same", "not_higher": "not higher"}
RANGE_LABELS = {("epochs_1", "discomfort_kh"): "< 0.05", ("lambda_0.8", "discomfort_kh"): "< 0.3"}
WEIGHT = RoomConfig().comfort_weight
DIFF_KEYS = ("cost", "energy_kwh", "discomfort_kh", "rollback")


# ---------------------------------------------------------------- loading
def rollback(t: np.ndarray, cost: np.ndarray, second_half: int) -> float:
    """Largest rise of the evaluation cost above its running best, over the second half of training."""
    rises = [cost[k] - cost[:k].min() for k in range(1, len(cost)) if t[k] > second_half]
    return max(0.0, max(rises))


def load(config: str, total_steps: int) -> list[dict]:
    runs = []
    for run in sorted((RUNS / config).glob("seed*"), key=lambda p: int(p.name[4:])):
        final = json.loads((run / "final_eval.json").read_text())["mean"]
        curve = pd.read_csv(run / "eval_curve.csv")
        cost_curve = (curve["energy_kwh"] + WEIGHT * curve["discomfort_kh"]).to_numpy()
        progress = pd.read_csv(run / "progress.csv")
        runs.append({
            "seed": int(run.name[4:]),
            "dir": run,
            "cost": final["energy_kwh"] + WEIGHT * final["discomfort_kh"],
            "energy_kwh": final["energy_kwh"],
            "discomfort_kh": final["discomfort_kh"],
            "violation_pct": final["violation_pct"],
            "rollback": rollback(curve["timesteps"].to_numpy(), cost_curve, total_steps // 2),
            "t": curve["timesteps"].to_numpy(),
            "cost_curve": cost_curve,
            "sigma_end": float(curve["policy_std"].iloc[-1]),
            "approx_kl": progress["train/approx_kl"].to_numpy(),
            "iterations": progress["time/iterations"].to_numpy(),
        })  # fmt: skip
    return runs


def col(runs: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in runs])


def diff(x: np.ndarray, base: np.ndarray) -> dict:
    d, lo, hi = welch_ci(x, base, 0.95)
    return {"d": d, "lo": lo, "hi": hi, "significant": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- the ablation
def ablation(runs: dict[str, list[dict]], spec: dict) -> dict:
    summary = {row["config"]: row for row in csv.DictReader(open(DATA / "ablation_summary.csv"))}
    verdict_rows = list(csv.DictReader(open(DATA / "ablation_verdicts.csv")))
    base = runs["baseline"]
    total_steps = spec["total_timesteps"]
    per_round = spec["n_envs"] * spec["ppo"]["n_steps"]
    rounds = total_steps // per_round
    per_epoch = per_round // spec["ppo"]["batch_size"]
    base_curve = np.stack([r["cost_curve"] for r in base]).mean(0)
    t = base[0]["t"]

    def baseline_steps_to_reach(cost: float) -> float | None:
        """Where the baseline's mean learning curve first got as low as `cost` (linear interpolation)."""
        if not (base_curve <= cost).any():
            return None
        k = int(np.argmax(base_curve <= cost))
        if k == 0:
            return float(t[0])
        return float(t[k - 1] + (base_curve[k - 1] - cost) / (base_curve[k - 1] - base_curve[k]) * (t[k] - t[k - 1]))

    def entry(name: str, value: float) -> dict:
        s, mine = summary[name], runs[name]
        # The summary file and the run directories must describe the same runs.
        assert len(mine) == int(s["seeds"]), name
        for key in ("cost", "energy_kwh", "discomfort_kh", "violation_pct", "rollback"):
            assert abs(col(mine, key).mean() - float(s[key])) < 1e-9, (name, key)
        n_epochs = json.loads((mine[0]["dir"] / "spec.json").read_text())["ppo"]["n_epochs"]
        gradient_steps = rounds * n_epochs * per_epoch
        sigma_end = float(col(mine, "sigma_end").mean())
        mean_curve = np.stack([r["cost_curve"] for r in mine]).mean(0)
        out = {
            "name": name,
            "value": value,
            "seeds": int(s["seeds"]),
            **{key: float(s[key]) for key in (
                "cost", "cost_sd", "energy_kwh", "discomfort_kh", "violation_pct",
                "rollback", "approx_kl", "clip_fraction", "train_seconds",
            )},
            "seed_costs": [round(float(c), 4) for c in col(mine, "cost")],
            "sigma_end": sigma_end,
            "gradient_steps": gradient_steps,
            # Adam moves ln(sigma) by at most about one learning rate per gradient step.
            "sigma_drop_vs_bound": float(np.log(1 / sigma_end) / (spec["ppo"]["learning_rate"] * gradient_steps)),
            "baseline_steps_to_reach": baseline_steps_to_reach(float(s["cost"])),
            "cost_at": {str(k): float(mean_curve[list(mine[0]["t"]).index(k)]) for k in (49_152, 98_304, 147_456)},
            # SB3 logs no approx_kl for the first round (NaN), hence nanmax.
            "approx_kl_max": float(max(np.nanmax(r["approx_kl"]) for r in mine)),
            "rounds_kl_above_0.1": int(sum((r["approx_kl"] > 0.1).sum() for r in mine)),
        }  # fmt: skip
        if name != "baseline":
            out["diff"] = {key: diff(col(mine, key), col(base, key)) for key in DIFF_KEYS}
            ratio, p = sd_ratio_test(col(mine, "cost"), col(base, "cost"))
            out["sd_ratio"] = {"ratio": ratio, "p": p, "significant": bool(p < 0.05)}
            out["predicted_cost"] = list(PREDICTED_COST[name])
            counts = dict.fromkeys(VERDICT_KEYS.values(), 0)
            predictions: dict[str, dict] = {}  # the one prediction made about each headline metric
            for row in verdict_rows:
                if row["config"] != name:
                    continue
                counts[VERDICT_KEYS[row["verdict"]]] += 1
                if row["metric"] in HEADLINE_METRICS:
                    key = HEADLINE_METRICS[row["metric"]]
                    assert key not in predictions, (name, key)
                    if key == "cost":
                        lo, hi = PREDICTED_COST[name]
                        label = f"{lo:.1f}\u2013{hi:.1f}"
                    else:
                        label = RANGE_LABELS[name, key] if row["rule"] == "range" else RULE_LABELS[row["rule"]]
                    predictions[key] = {"label": label, "verdict": VERDICT_KEYS[row["verdict"]]}
            out["verdicts"] = counts
            out["predictions"] = predictions
        return out

    totals = dict.fromkeys(VERDICT_KEYS.values(), 0)
    for row in verdict_rows:
        totals[VERDICT_KEYS[row["verdict"]]] += 1
    totals["total"] = len(verdict_rows)

    base_costs = col(base, "cost")
    first5 = np.array([r["cost"] for r in base if r["seed"] < 5])
    worst = int(np.argmax(base_costs))
    thermostat_cost = mean_metrics(evaluate_controller(tuned_thermostat(), RoomConfig(), eval_seeds(100)))
    thermostat_cost = thermostat_cost["energy_kwh"] + WEIGHT * thermostat_cost["discomfort_kh"]
    return {
        "baseline": entry("baseline", float("nan")),
        "groups": [
            {"key": key, "param": param, "default": default, "rows": [entry(name, value) for name, value in members]}
            for key, param, default, members in GROUPS
        ],
        "verdict_totals": totals,
        "baseline_seeds": {
            "first5_mean": float(first5.mean()), "first5_sd": float(first5.std(ddof=1)),
            "worst_seed": base[worst]["seed"], "worst_cost": float(base_costs[worst]),
            "sd_without_worst": float(np.delete(base_costs, worst).std(ddof=1)),
        },  # fmt: skip
        "runs_below_thermostat": {
            name: int((col(runs[name], "cost") < thermostat_cost).sum()) for name in CONFIGS
        },
        "n_runs": sum(len(runs[name]) for name in CONFIGS),
        "rounds": rounds,
    }


# ---------------------------------------------------------------- reference controllers
def controllers() -> list[dict]:
    """Reference controllers on the same 100 test days as the PPO evaluation."""
    cfg, seeds = RoomConfig(), eval_seeds(100)
    table = [
        ("off", "HVAC off", OffController()),
        ("thermostat_8", "Thermostat, on at 8:00", ThermostatController(prestart_hours=0)),
        ("thermostat_3", "Thermostat, on at 3:00", ThermostatController(prestart_hours=5)),
        ("thermostat_tuned", "Thermostat, tuned", tuned_thermostat()),
        ("oracle", "Oracle (knows the weather)", "oracle"),
    ]
    out = []
    for key, label, controller in table:
        m = mean_metrics(evaluate_controller(controller, cfg, seeds))
        out.append({
            "key": key, "label": label, "energy_kwh": m["energy_kwh"], "discomfort_kh": m["discomfort_kh"],
            "violation_pct": m["violation_pct"], "cost": m["energy_kwh"] + WEIGHT * m["discomfort_kh"],
        })  # fmt: skip
    return out


# ---------------------------------------------------------------- GAE lambda by time of day
def lambda_by_time_of_day(runs: dict[str, list[dict]]) -> list[dict]:
    """Energy and discomfort split by time of day (same split as scripts/lambda_time_of_day.py)."""
    from lambda_time_of_day import replay  # scripts/lambda_time_of_day.py
    from stable_baselines3 import PPO

    cfg, days = RoomConfig(), eval_seeds(100)
    hour_end = (np.arange(cfg.steps_per_episode) + 1) * cfg.dt_hours  # time at the end of each step
    occupied = (hour_end > cfg.occupied_start_hour) & (hour_end <= cfg.occupied_end_hour)
    before = hour_end <= cfg.occupied_start_hour
    after = hour_end > cfg.occupied_end_hour
    first2h = occupied & (hour_end <= cfg.occupied_start_hour + 2)

    def split(energy: np.ndarray, discomfort: np.ndarray, t_out_mean: np.ndarray) -> list[float]:
        return [
            energy[:, before].sum(1).mean(), energy[:, occupied].sum(1).mean(), energy[:, after].sum(1).mean(),
            discomfort[:, first2h].sum(1).mean(), discomfort[:, occupied & ~first2h].sum(1).mean(),
            discomfort[t_out_mean < 10.0].sum(1).mean(), discomfort[t_out_mean >= 22.0].sum(1).mean(),
        ]  # fmt: skip

    keys = ("energy_before", "energy_occupied", "energy_after", "discomfort_first2h", "discomfort_rest",
            "discomfort_cold_days", "discomfort_hot_days")  # fmt: skip
    rows = []
    for label, controller in (("Oracle", "oracle"), ("Thermostat, tuned", tuned_thermostat())):
        env = RoomThermalEnv(cfg)
        if controller == "oracle":
            eps = [run_episode(env, s, oracle=True, record=True) for s in days]
        else:
            eps = [run_episode(env, s, policy=lambda e, _o: controller.act(e), record=True) for s in days]
        values = split(
            np.stack([e["trajectory"]["energy_kwh"] for e in eps]),
            np.stack([e["trajectory"]["discomfort_kh"] for e in eps]),
            np.array([e["t_out_mean"] for e in eps]),
        )
        rows.append({"label": label, "kind": "reference", **dict(zip(keys, map(float, values)))})
    members = dict(GROUPS[2][3])
    for name, lam in (("lambda_0.8", 0.8), ("lambda_0.9", 0.9), ("baseline", GROUPS[2][2]),
                      ("lambda_0.99", 0.99), ("lambda_1", 1.0)):  # fmt: skip
        assert name == "baseline" or members[name] == lam
        per_seed = []
        for r in runs[name]:
            out = replay(PPO.load(r["dir"] / "model", device="cpu"), cfg, days)
            per_seed.append(split(out["energy"], out["discomfort"], out["t_out_mean"]))
            # The replay must reproduce the stored final evaluation of this run.
            assert abs(out["energy"].sum(1).mean() - r["energy_kwh"]) < 1e-6, (name, r["seed"])
        rows.append({"label": f"{lam:g}", "kind": "baseline" if name == "baseline" else "ppo", "value": lam,
                     **dict(zip(keys, map(float, np.mean(per_seed, 0))))})  # fmt: skip
    return rows


def clean(obj):
    """Round floats and replace NaN (not valid JSON) by None."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, 6)
    return obj


if __name__ == "__main__":
    spec = json.loads((RUNS / "baseline" / "seed0" / "spec.json").read_text())
    runs = {name: load(name, spec["total_timesteps"]) for name in CONFIGS}
    data = {
        "total_steps": spec["total_timesteps"],
        "comfort_weight": WEIGHT,
        "ablation": ablation(runs, spec),
        "controllers": controllers(),
        "lambda_by_time": lambda_by_time_of_day(runs),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean(data), indent=1, ensure_ascii=False) + "\n")

    # Everything below is printed so the numbers quoted in the prose can be checked by eye.
    print(f"wrote {OUT.relative_to(ROOT)}")
    for c in data["controllers"]:
        print(f"  {c['label']:28s} energy {c['energy_kwh']:.2f}  discomfort {c['discomfort_kh']:.2f}  "
              f"violations {c['violation_pct']:.1f}%  cost {c['cost']:.2f}")
    ab = data["ablation"]
    b = ab["baseline"]
    print(f"  {'baseline':12s} cost {b['cost']:.2f} ± {b['cost_sd']:.2f}  sigma_end {b['sigma_end']:.3f} "
          f"({b['sigma_drop_vs_bound']:.2f} of bound)  kl max {b['approx_kl_max']:.2f}")
    for group in ab["groups"]:
        for row in group["rows"]:
            d, rb, reach = row["diff"]["cost"], row["diff"]["rollback"], row["baseline_steps_to_reach"]
            print(f"  {row['name']:12s} cost {row['cost']:.2f} ± {row['cost_sd']:.2f}  "
                  f"diff {d['d']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]  "
                  f"rollback diff {rb['d']:+.2f} [{rb['lo']:+.2f}, {rb['hi']:+.2f}]  "
                  f"sd ratio {row['sd_ratio']['ratio']:.2f} (p {row['sd_ratio']['p']:.3f})  "
                  f"sigma_end {row['sigma_end']:.3f} ({row['sigma_drop_vs_bound']:.2f} of bound)  "
                  f"baseline there at {'never' if reach is None else f'{reach / 1e3:.0f}k'}  "
                  f"kl max {row['approx_kl_max']:.2f}, rounds > 0.1: {row['rounds_kl_above_0.1']}  "
                  f"cost at 49k/98k/147k {' / '.join(f'{v:.2f}' for v in row['cost_at'].values())}")
    print(f"  baseline cost at 49k/98k/147k {' / '.join(f'{v:.2f}' for v in b['cost_at'].values())}")
    print("  verdict totals", ab["verdict_totals"])
    print("  baseline seeds", ab["baseline_seeds"])
    print("  runs below the tuned thermostat", {k: v for k, v in ab["runs_below_thermostat"].items() if v},
          "of", ab["n_runs"])
    print("  lambda by time of day: energy before 8 / 8-18 / after 18 | discomfort 8-10 / 10-18 | cold / hot days")
    for row in data["lambda_by_time"]:
        print(f"  {row['label']:>18s} {row['energy_before']:.2f} {row['energy_occupied']:.2f} {row['energy_after']:.2f} | "
              f"{row['discomfort_first2h']:.3f} {row['discomfort_rest']:.3f} | "
              f"{row['discomfort_cold_days']:.3f} {row['discomfort_hot_days']:.3f}")
