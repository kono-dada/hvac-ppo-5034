"""Run read-only clip 0.2/0.5 gradient probes on the original clip=0.5 trajectories."""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from hvac.gradient_probe import ClipGradientProbePPO
from hvac.train import train_ppo
from scripts.run_ablation import EVAL_DAYS, EVAL_EVERY, STEPS, configs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "gradient_probe" / "clip_0.5"


def run_seed(spec):
    seed, force = spec
    import hvac.diagnostics as diagnostics
    run = OUT / f"seed{seed}"
    if not force and all((run / f).exists() for f in ("gradient_epochs.csv", "adam_probes.csv", "model.zip", "final_eval.json")):
        return seed
    # Isolate the opt-in subclass within this worker; the standard training
    # entry point and its defaults remain unchanged.
    original = diagnostics.DiagnosticPPO
    diagnostics.DiagnosticPPO = ClipGradientProbePPO
    try:
        result = train_ppo(run, ppo_overrides=configs()["clip_0.5"], total_timesteps=STEPS,
                           seed=seed, eval_every=EVAL_EVERY, n_curve_days=EVAL_DAYS,
                           n_final_days=EVAL_DAYS, diagnostics=True)
    finally:
        diagnostics.DiagnosticPPO = original
    print(f"gradient probe seed{seed}: cost {-result['return']:.6f}, seconds {result['train_seconds']:.1f}", flush=True)
    return seed


def verify(seeds):
    from scripts.analyze_clip_diagnostics import _same_model
    rows = []
    for seed in seeds:
        run = OUT / f"seed{seed}"
        orig = ROOT / "runs" / "ablation" / "clip_0.5" / run.name
        a, b = (pd.read_csv(p / "progress.csv", dtype=str, keep_default_na=False) for p in (orig, run))
        cols = sorted(set(a.columns) - {"time/fps", "time/time_elapsed"})
        fa, fb = (json.loads((p / "final_eval.json").read_text()) for p in (orig, run))
        check = {
            "seed": seed,
            "progress": a[cols].equals(b[cols]),
            "curve": pd.read_csv(orig / "eval_curve.csv", dtype=str).equals(pd.read_csv(run / "eval_curve.csv", dtype=str)),
            "final_per_day": fa["per_day"] == fb["per_day"],
            "final_mean": {k: v for k, v in fa["mean"].items() if k != "train_seconds"}
                          == {k: v for k, v in fb["mean"].items() if k != "train_seconds"},
            "policy_and_adam": _same_model(orig, run),
            "epochs": len(pd.read_csv(run / "gradient_epochs.csv")) == 1000,
            "adam_probes": len(pd.read_csv(run / "adam_probes.csv")) == 600,
        }
        check["passed"] = all(v for k, v in check.items() if k != "seed")
        print(check, flush=True)
        rows.append(check)
    evidence = {"all_passed": all(r["passed"] for r in rows), "runs": rows}
    (OUT.parent / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    if not evidence["all_passed"]:
        raise SystemExit("Gradient-probe reproduction check failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    seeds = list(range(args.seeds))
    with get_context("spawn").Pool(args.workers) as pool:
        list(pool.imap_unordered(run_seed, [(seed, args.force) for seed in seeds]))
    verify(seeds)
