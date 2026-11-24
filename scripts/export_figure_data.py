"""Export the data behind every report figure as JSON, for the Typst figures.

The Typst figures (docs/report/typst/figures/*.typ) only draw; every number
they show comes from the files written here. The computations reuse the
functions of the existing figure scripts, so the Typst figures and the
matplotlib PNGs show the same data.

Usage:  PYTHONPATH=. .venv/bin/python scripts/export_figure_data.py
Writes: docs/report/typst/figures/data/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_ablation as ab  # noqa: E402
import make_method_figures as mf  # noqa: E402
from hvac.controllers import ThermostatController  # noqa: E402
from hvac.env import RoomConfig, RoomThermalEnv, discretize  # noqa: E402
from hvac.evaluation import eval_seeds, run_episode  # noqa: E402
from hvac.stats import mean_ci  # noqa: E402
from hvac.train import DEFAULT_PPO, N_ENVS  # noqa: E402

OUT = ROOT / "docs" / "report" / "typst" / "figures" / "data"
SEEDS = eval_seeds(100)


def r(x, nd: int = 4):
    """Round floats (also inside lists) so the JSON stays small."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, nd) for v in x]
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def write(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {name}.json")


# ---------------------------------------------------------------- environment
def time_constants_h(cfg: RoomConfig) -> list[float]:
    """Time constants of the room (hours, fast first), from the eigenvalues of the exact one-step matrix."""
    a_d, _ = discretize(cfg)
    mu = np.linalg.eigvals(a_d).real
    return sorted(float(-cfg.dt_hours / np.log(m)) for m in mu)


def config_numbers(cfg: RoomConfig) -> dict:
    return {
        "ua_air_out": cfg.ua_air_out, "ua_air_mass": cfg.ua_air_mass, "ua_mass_out": cfg.ua_mass_out,
        "c_air": cfg.c_air, "c_mass": cfg.c_mass, "cop": cfg.cop,
        "q_hvac_max": cfg.q_hvac_max, "solar_to_air": cfg.solar_to_air, "q_solar_peak": cfg.q_solar_peak,
        "q_internal_occupied": cfg.q_internal_occupied, "q_internal_idle": cfg.q_internal_idle,
        "comfort_low": cfg.comfort_low,
        "comfort_high": cfg.comfort_high, "occupied_start": cfg.occupied_start_hour,
        "occupied_end": cfg.occupied_end_hour, "dt_hours": cfg.dt_hours, "comfort_weight": cfg.comfort_weight,
        "t_out_mean_range": list(cfg.t_out_mean_range), "time_constants_h": r(time_constants_h(cfg), 3),
    }


def step_response(cfg: RoomConfig) -> dict:
    """Same simulation as make_environment_figures.fig_step_response."""
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
    i_off = int(on_to / cfg.dt_hours)
    return {"t_out": t_out, "on_from": on_from, "on_to": on_to, "total": total, "hours": r(hours),
            "air": r(air), "mass": r(mass), "i_off": i_off, "i_later": i_off + 4}


def days(cfg: RoomConfig) -> dict:
    env = RoomThermalEnv(cfg)
    out = []
    for seed in range(12):
        _, info = env.reset(seed=seed)
        out.append({"seed": seed, "t_out_mean": r(info["t_out_mean"], 3), "t_out": r(env.t_out, 3),
                    "q_solar": r(env.q_solar, 1)})
    return {"hours": r(env.hours, 3), "days": out}


def controllers(cfg: RoomConfig) -> dict:
    """Same days and controllers as make_environment_figures.fig_controllers."""
    env = RoomThermalEnv(cfg)
    means = {s: env.reset(seed=s)[1]["t_out_mean"] for s in SEEDS}
    picks = [("cold", min(SEEDS, key=lambda s: abs(means[s] - 1.0))),
             ("hot", min(SEEDS, key=lambda s: abs(means[s] - 30.0)))]
    thermostat = ThermostatController(prestart_hours=0)
    out = []
    for label, seed in picks:
        eps = {
            "off": run_episode(env, seed, policy=lambda _e, _o: 0.0, record=True),
            "thermostat": run_episode(env, seed, policy=lambda e, _o: thermostat.act(e), record=True),
            "oracle": run_episode(env, seed, oracle=True, record=True),
        }
        tr = eps["oracle"]["trajectory"]
        out.append({
            "label": label, "seed": seed, "t_out_mean": r(eps["oracle"]["t_out_mean"], 3),
            "hours": r(tr["hour"] + cfg.dt_hours, 3), "t_out": r(tr["t_out"], 3), "t_mass_oracle": r(tr["t_mass"], 3),
            **{f"t_air_{k}": r(ep["trajectory"]["t_air"], 3) for k, ep in eps.items()},
            **{f"action_{k}": r(ep["trajectory"]["hvac_action"], 3) for k, ep in eps.items() if k != "off"},
            "metrics": {k: {"energy": r(ep["energy_kwh"], 3), "discomfort": r(ep["discomfort_kh"], 3)}
                        for k, ep in eps.items()},
        })
    return {"days": out}


def method() -> dict:
    p = DEFAULT_PPO
    per_update = N_ENVS * p["n_steps"]
    return {"n_envs": N_ENVS, "n_steps": p["n_steps"], "per_update": per_update, "days": round(per_update / 96),
            "batch_size": p["batch_size"], "minibatches": per_update // p["batch_size"], "gamma": p["gamma"],
            "n_epochs": p["n_epochs"], "rounds": int(np.ceil(mf.TOTAL_STEPS / per_update)),
            "total_steps": mf.TOTAL_STEPS}


# ---------------------------------------------------------------- ablation
def group_meta() -> dict:
    out = {}
    for g, (label, names, values) in ab.GROUPS.items():
        base = values[names.index("baseline")]
        below = sorted((v for v in values if v < base), reverse=True)
        above = sorted(v for v in values if v > base)
        out[g] = {"label": label, "names": names, "values": values,
                  # side: -1 below the baseline value, +1 above; rank: 0 = nearest the baseline
                  "side": [0 if v == base else (-1 if v < base else 1) for v in values],
                  "rank": [0 if v == base else (below.index(v) if v < base else above.index(v)) for v in values]}
    return out


def ablation(runs: dict, refs: dict) -> dict:
    keys = ("cost", "energy_kwh", "discomfort_kh")
    configs = {}
    for name in ab.CONFIGS:
        rs = runs[name]
        t, *_ = ab._curve_stats(rs, "cost")
        c = {"seeds": [x["seed"] for x in rs], "t": r(t), "curves": {}, "final": {}}
        for k in keys:
            _, m, s = ab._curve_stats(rs, k)
            c["curves"][k] = {"mean": r(m), "sd": r(s)}
            v = np.array([x[k] for x in rs])
            mm, lo, hi = mean_ci(v)
            c["final"][k] = {"values": r(v), "mean": r(mm), "lo": r(lo), "hi": r(hi)}
        prog = [x["progress"] for x in rs]
        c["iterations"] = r(prog[0]["time/iterations"].to_numpy())
        for k in ("approx_kl", "clip_fraction"):
            c[k] = r(np.stack([p[f"train/{k}"].to_numpy() for p in prog]).mean(0), 6)
        configs[name] = c
    predictions = {c: list(b) for c, m, k, b, _ in ab.PREDICTIONS if k == "range" and m == "cost"}
    # verdicts of those cost predictions, judged by the same rule as the tables of section 3.6
    verdict_key = {"命中": "hit", "部分命中": "partial", "不命中": "miss", "无法判断": "undecided"}
    vt = ab.verdict_table(runs)
    vt = vt[(vt["rule"] == "range") & (vt["metric"] == ab.METRICS["cost"][0])]
    verdicts = {row.config: verdict_key[row.verdict] for row in vt.itertuples()}
    # per-config means over the whole training (what the summary table reports), with 95% CIs over seeds
    st = ab.summary_table(runs).set_index("config")
    summary = {c: {k: {"mean": r(st.at[c, k], 6), "lo": r(st.at[c, f"{k}_ci_lo"], 6), "hi": r(st.at[c, f"{k}_ci_hi"], 6)}
                   for k in ("approx_kl", "clip_fraction", "rollback")} for c in st.index}
    names = {"温控器（调过参数）": "thermostat", "未调参的温控器": "thermostat_untuned", "理论最优": "oracle"}
    return {
        "total_steps": ab.TOTAL_STEPS, "weight": ab.WEIGHT, "groups": group_meta(), "configs": configs,
        "predicted_cost": predictions, "verdicts": verdicts, "summary": summary,
        "refs": {names[k]: {kk: r(m[kk]) for kk in keys} for k, m in refs.items()},
    }


if __name__ == "__main__":
    cfg = RoomConfig()
    write("env", {"config": config_numbers(cfg), "step": step_response(cfg), "days": days(cfg),
                  "controllers": controllers(cfg)})
    write("method", method())
    runs = {c: ab.load(c) for c in ab.CONFIGS}
    missing = [c for c, rs in runs.items() if len(rs) < 2]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    write("ablation", ablation(runs, ab.references()))
