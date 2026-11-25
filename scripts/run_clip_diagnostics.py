"""Post-hoc diagnostics of clipping: rerun some ablation configurations with extra logging.

Designed after the ablation results were known (task-20261125-03), to check the
explanations in section 3.6 of docs/report/zh/03_ablation.md. The training
itself is identical to scripts/run_ablation.py (same settings, same seeds); the
only difference is hvac.diagnostics.DiagnosticPPO, which records what clipping
does inside each update. scripts/analyze_clip_diagnostics.py checks that every
rerun reproduces the original run exactly.

Usage:  python -m scripts.run_clip_diagnostics [--workers 10] [--seeds 3] [--list]
Output: runs/diagnostics/<config>/seed<k>/   (same files as the ablation, plus
        diag_epochs.csv and diag_rounds.csv)
Then:   python -m scripts.analyze_clip_diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hvac.train import run_many
from scripts.run_ablation import EVAL_DAYS, EVAL_EVERY, STEPS, configs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "diagnostics"
# All clip ranges; epochs 1 and 3 for their true end-of-round KL.
CONFIGS = ["clip_0.05", "clip_0.1", "baseline", "clip_0.3", "clip_0.5", "epochs_1", "epochs_3"]

def build_jobs(names=None, seeds=range(3), force=False) -> list[dict]:
    overrides = configs()
    jobs = []
    for name in (names or CONFIGS):
        for seed in seeds:
            run = OUT / name / f"seed{seed}"
            # A final evaluation alone is not a completion marker: the interrupted
            # session may have stopped before save_diagnostics wrote both CSVs.
            artifacts = ("spec.json", "progress.csv", "eval_curve.csv", "final_eval.json",
                         "model.zip", "diag_epochs.csv", "diag_rounds.csv")
            if not force and all((run / f).is_file() for f in artifacts):
                continue
            jobs.append({
                "run_dir": run, "ppo_overrides": overrides[name],
                "total_timesteps": STEPS, "seed": seed, "eval_every": EVAL_EVERY,
                "n_curve_days": EVAL_DAYS, "n_final_days": EVAL_DAYS, "diagnostics": True,
            })
    return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--only", nargs="+", choices=CONFIGS)
    parser.add_argument("--force", action="store_true", help="rerun even completed diagnostics")
    parser.add_argument("--list", action="store_true", help="show pending runs without training")
    args = parser.parse_args()
    if args.workers < 1 or args.seeds < 1:
        parser.error("--workers and --seeds must be positive")
    jobs = build_jobs(args.only, range(args.seeds), force=args.force)
    print(f"{len(jobs)} pending diagnostic runs", flush=True)
    if args.list:
        for job in jobs:
            print(job["run_dir"])
    else:
        # Completion was checked above; allow repair of runs with final_eval only.
        run_many(jobs, workers=args.workers, skip_done=False)
