"""Analyse the post-hoc clipping diagnostics (task-20261125-03).

Reads runs/diagnostics/<config>/seed<k>/ (from scripts/run_clip_diagnostics.py),
checks that every rerun reproduces runs/ablation/<config>/seed<k>/ exactly, and
prints every number quoted in the post-hoc section of docs/report/zh/03_ablation.md.

Writes: docs/report/data/clip_diagnostics.md   the four tables of section 3.7
        docs/report/data/clip_diagnostics_summary.csv
        docs/report/data/clip_diagnostics_verification.json
        docs/report/figures/diag_clip.png

Usage:  python -m scripts.analyze_clip_diagnostics
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3
import torch as th

from hvac.diagnostics import SB3_VERSION
from scripts.run_clip_diagnostics import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "runs" / "diagnostics"
ABL = ROOT / "runs" / "ablation"
DATA = ROOT / "docs" / "report" / "data"
FIG = ROOT / "docs" / "report" / "figures"

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLIPS = [("clip_0.05", 0.05), ("clip_0.1", 0.1), ("baseline", 0.2), ("clip_0.3", 0.3), ("clip_0.5", 0.5)]
EPOCHS = [("epochs_1", 1), ("epochs_3", 3), ("baseline", 10)]
EARLY = (2, 30)  # update rounds
LATE = (71, 100)
NOISE_LEVEL = 1 / 32  # consistency of 32 independent zero-mean minibatch gradients


def runs_of(config: str) -> list[Path]:
    return sorted(DIAG.glob(f"{config}/seed*"))


def _same_nested(a, b) -> bool:
    """Compare saved model/Adam tensors exactly, including step counters."""
    if isinstance(a, th.Tensor) and isinstance(b, th.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and th.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_nested(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same_nested(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _same_model(orig: Path, run: Path) -> bool:
    with zipfile.ZipFile(orig / "model.zip") as a, zipfile.ZipFile(run / "model.zip") as b:
        return all(_same_nested(
            th.load(io.BytesIO(a.read(name)), map_location="cpu", weights_only=True),
            th.load(io.BytesIO(b.read(name)), map_location="cpu", weights_only=True),
        ) for name in ("policy.pth", "policy.optimizer.pth"))


def _diagnostic_integrity(ep: pd.DataFrame, rd: pd.DataFrame, spec: dict) -> bool:
    n_rounds = spec["total_timesteps"] // (spec["n_envs"] * spec["ppo"]["n_steps"])
    n_epochs = spec["ppo"]["n_epochs"]
    keys = pd.MultiIndex.from_product([range(1, n_rounds + 1), range(1, n_epochs + 1)])
    return bool(
        rd["round"].tolist() == list(range(1, n_rounds + 1))
        and pd.MultiIndex.from_frame(ep[["round", "epoch"]]).equals(keys)
        and ep["identity_residual_max"].max() < 1e-4
        and np.allclose(ep["zeroed_frac"] + ep["against_frac"]
                        + ep.get("boundary_roundoff_frac", 0) + ep.get("surrogate_tie_frac", 0),
                        ep["clip_fraction"], rtol=0, atol=1e-6)
        and np.allclose(ep[["u_lt_m1", "u_m1_0", "u_0_05", "u_05_08", "u_08_1", "u_gt_1"]].sum(axis=1), 1)
        and np.isfinite(ep["kl_exact"]).all() and (ep["kl_exact"] >= -1e-7).all()
        and ep[["consistency_c", "consistency_u"]].ge(0).all().all()
        and ep[["consistency_c", "consistency_u"]].le(1 + 1e-6).all().all()
    )


def reproduction_check(diag: Path = DIAG, abl: Path = ABL, evidence_path: Path | None = None) -> bool:
    """Require all 7 x 3 runs; exclude wall-clock timing, not training quantities."""
    if stable_baselines3.__version__ != SB3_VERSION:
        raise RuntimeError(f"Expected SB3 {SB3_VERSION}, found {stable_baselines3.__version__}")
    ok_all = True
    evidence = []
    print("REPRODUCTION: rerun with diagnostics vs the original ablation run")
    for name in CONFIGS:
        for seed in range(3):
            run = diag / name / f"seed{seed}"
            orig = abl / name / run.name
            check = {"config": name, "seed": seed}
            required = ("spec.json", "progress.csv", "eval_curve.csv", "final_eval.json", "model.zip")
            complete = all((orig / f).is_file() and (run / f).is_file() for f in required)
            complete &= all((run / f).is_file() for f in ("diag_epochs.csv", "diag_rounds.csv"))
            check["complete"] = complete
            if complete:
                # Compare original CSV tokens, not rounded floating-point parses.
                pa, pb = (pd.read_csv(p / "progress.csv", dtype=str, keep_default_na=False) for p in (orig, run))
                timing = {"time/fps", "time/time_elapsed"}
                progress_cols = sorted(set(pa.columns) - timing)
                check["progress_except_wall_clock"] = (
                    set(pa.columns) == set(pb.columns) and pa[progress_cols].equals(pb[progress_cols]))
                ca, cb = (pd.read_csv(p / "eval_curve.csv", dtype=str, keep_default_na=False) for p in (orig, run))
                check["eval_curve"] = ca.equals(cb)
                fa, fb = (json.loads((p / "final_eval.json").read_text()) for p in (orig, run))
                check["final_per_day"] = fa["per_day"] == fb["per_day"]
                ma, mb = ({k: v for k, v in f["mean"].items() if k != "train_seconds"} for f in (fa, fb))
                check["final_mean_except_wall_clock"] = ma == mb
                sa, sb = (json.loads((p / "spec.json").read_text()) for p in (orig, run))
                check["spec"] = sa == sb
                check["policy_and_adam"] = _same_model(orig, run)
                ep, rd = (pd.read_csv(run / f) for f in ("diag_epochs.csv", "diag_rounds.csv"))
                check["diagnostic_integrity"] = _diagnostic_integrity(ep, rd, sb)
                logged = pb.loc[pb["time/iterations"] != "1", "train/approx_kl"].to_numpy(dtype=float)
                check["recomputed_approx_kl"] = len(rd) == len(logged) + 1 and np.allclose(
                    rd["sb3_approx_kl"].iloc[:-1].to_numpy(), logged, rtol=1e-6, atol=0)
                check["max_gradient_identity_residual"] = float(ep["identity_residual_max"].max())
                # Legacy records are accepted only when all their identities pass;
                # the three failing runs were rerun with boundary-aware masks.
                check["recording_schema"] = "boundary-aware" if "boundary_roundoff_frac" in ep else "legacy-validated"
            check["passed"] = all(v for k, v in check.items() if k not in {"config", "seed", "max_gradient_identity_residual", "recording_schema"})
            ok_all &= check["passed"]
            evidence.append(check)
            print(f"  {name:10s} {run.name}: " + ", ".join(f"{k} {v}" for k, v in check.items()
                  if k not in {"config", "seed", "max_gradient_identity_residual"}))
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        sources = [ROOT / "hvac/diagnostics.py", ROOT / "hvac/train.py", Path(sys.modules["stable_baselines3.ppo.ppo"].__file__)]
        evidence_path.write_text(json.dumps({
            "sb3_version": stable_baselines3.__version__, "torch_version": th.__version__,
            "source_hash_scope": "current verification sources; inherited runs may use the validated legacy schema",
            "gradient_identity_relative_tolerance": 1e-4,
            "expected_runs": 21, "all_passed": ok_all,
            "excluded_wall_clock_columns": ["time/fps", "time/time_elapsed", "mean.train_seconds"],
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "runs": evidence,
        }, indent=2) + "\n")
    print(f"  ALL IDENTICAL: {ok_all}")
    return ok_all


def load(config: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ep, rd = [], []
    for run in runs_of(config):
        seed = int(run.name.removeprefix("seed"))
        ep.append(pd.read_csv(run / "diag_epochs.csv").assign(seed=seed))
        rd.append(pd.read_csv(run / "diag_rounds.csv").assign(seed=seed))
    return pd.concat(ep, ignore_index=True), pd.concat(rd, ignore_index=True)


def window(df: pd.DataFrame, lo_hi: tuple[int, int]) -> pd.DataFrame:
    return df[(df["round"] >= lo_hi[0]) & (df["round"] <= lo_hi[1])]


def per_seed_mean(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """Mean over seeds of the per-seed mean, and the spread (min-max half range) across seeds."""
    s = df.groupby("seed")[col].mean()
    return float(s.mean()), float((s.max() - s.min()) / 2)


def end_of_round(ep: pd.DataFrame) -> pd.DataFrame:
    last = ep.groupby(["seed", "round"])["epoch"].transform("max")
    return ep[ep["epoch"] == last]


def kl_parts(rounds: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Split Gaussian KL(old || new) into policy-mean and standard-deviation terms."""
    q = rounds["sigma_start"] / rounds["sigma_end"]
    mean_part = 0.5 * rounds["mean_shift_rms_end"] ** 2 * q ** 2
    std_part = -np.log(q) + 0.5 * q ** 2 - 0.5
    return mean_part, std_part


def main_table(data: dict) -> pd.DataFrame:
    rows = []
    for name, eps in CLIPS:
        ep, rd = data[name]
        for label, win in (("前期（第 2–30 轮）", EARLY), ("后期（第 71–100 轮）", LATE)):
            e, r = window(ep, win), window(rd, win)
            end = end_of_round(e)
            first = e[e["epoch"] == 1]
            row = {"clip": eps, "阶段": label, "sigma": r["sigma_start"].mean()}
            for col in ("clip_fraction", "zeroed_frac", "against_frac", "z_norm_share", "z_proj_share", "cos_c_u",
                        "consistency_c", "consistency_u", "grad_clipped", "grad_clipped_unclipped_surrogate",
                        "grad_scale", "grad_norm_actor", "grad_norm_critic", "cos_to_mean_c", "cos_to_mean_u",
                        "logstd_g_c", "logstd_g_u", "logstd_g_z"):
                row[col] = e[col].mean()
            row["near_boundary"] = end["u_08_1"].mean()
            row["past_boundary"] = end["u_gt_1"].mean()
            row["kl_epoch1"] = first["kl_exact"].mean()
            row["kl_end"] = r["kl_exact_end"].mean()
            km, ks = kl_parts(r)
            row["kl_mean_component"], row["kl_std_component"] = km.mean(), ks.mean()
            row["kl_mean_share"] = km.mean() / row["kl_end"]
            row["mu_shift_abs"] = r["mu_shift_abs"].mean()
            row["mean_shift_rms_end"] = r["mean_shift_rms_end"].mean()
            row["path_efficiency"] = r["path_efficiency"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def markdown(data: dict) -> str:
    """The four tables of section 3.7 (late rounds unless explicitly labelled)."""
    a = ["| clip range | σ | 越界比例（SB3 的 `clip_fraction`） | 其中梯度被置零 | 其中仍在往回拉 | "
         "被置零数据的梯度长度 / 无裁剪梯度长度 | 被置零数据沿无裁剪梯度方向的份额 | 实际梯度与无裁剪梯度的夹角 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    b = ["| clip range | 小批梯度一致性：实际使用 | 不带裁剪 | 第 1 遍后已走完全轮 KL 的 | 一轮内参数走过的路径长度 | 净位移 | 净位移 / 路径 |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    c = ["| clip range | 梯度范数裁剪的触发比例（全程） | 策略网络梯度范数 | 价值网络梯度范数 | 平均缩放系数 | σ 在 0.1–0.2 之间的轮次里的缩放系数 |",
         "| --- | --- | --- | --- | --- | --- |"]
    for name, eps in CLIPS:
        ep, rd = data[name]
        e, r = window(ep, LATE), window(rd, LATE)
        k = e.groupby("epoch")["kl_exact"].mean()
        angle = np.degrees(np.arccos(e["cos_c_u"].mean()))
        a.append(f"| {eps:g} | {r['sigma_start'].mean():.3f} | {100 * e['clip_fraction'].mean():.1f}% | "
                 f"{100 * e['zeroed_frac'].mean():.1f}% | {100 * e['against_frac'].mean():.1f}% | "
                 f"{100 * e['z_norm_share'].mean():.0f}% | {100 * e['z_proj_share'].mean():.0f}% | 约 {angle:.0f}° |")
        b.append(f"| {eps:g} | {e['consistency_c'].mean():.3f} | {e['consistency_u'].mean():.3f} | "
                 f"{k.iloc[0] / k.iloc[-1]:.0%} | {r['actor_path_length'].mean():.2f} | "
                 f"{r['actor_net_displacement'].mean():.3f} | {r['path_efficiency'].mean():.2f} |")
        same = rd[(rd["sigma_start"] >= 0.10) & (rd["sigma_start"] <= 0.20)]
        same_e = ep.merge(same[["seed", "round"]], on=["seed", "round"])
        c.append(f"| {eps:g} | {100 * ep['grad_clipped'].mean():.1f}% | {e['grad_norm_actor'].mean():.2f} | "
                 f"{e['grad_norm_critic'].mean():.2f} | {e['grad_scale'].mean():.3f} | {same_e['grad_scale'].mean():.3f} |")
    d = ["| epochs | SB3 记录的 `approx_kl` | 一轮结束时的真实 KL | 记录值 / 真实值 |", "| --- | --- | --- | --- |"]
    for name, n_ep in EPOCHS:
        r = data[name][1]
        d.append(f"| {n_ep} | {r['sb3_approx_kl'].mean():.4f} | {r['kl_exact_end'].mean():.4f} | "
                 f"{r['sb3_approx_kl'].mean() / r['kl_exact_end'].mean():.2f} |")
    return "\n\n".join("\n".join(t) for t in (a, b, c, d)) + "\n"


def figure(data: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(CLIPS)))

    # (a) KL after each epoch of a round, late rounds
    ax = axes[0, 0]
    for (name, eps), c in zip(CLIPS, colors):
        e = window(data[name][0], LATE)
        k = e.groupby("epoch")["kl_exact"].mean()
        ax.plot(k.index, k.values, "o-", color=c, lw=2 if name == "baseline" else 1.3, label=f"clip = {eps:g}")
    ax.set_yscale("log")
    ax.set_xlabel("一轮之内的第几遍")
    ax.set_ylabel("与收集数据时策略的 KL（整批 2,048 条）")
    ax.set_title("(a) 每一遍结束时走了多远（第 71–100 轮平均）")
    ax.set_xticks(range(1, 11))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)

    # (b) share of the unclipped gradient that comes from zeroed samples
    ax = axes[0, 1]
    x = np.arange(len(CLIPS))
    w = 0.27
    for i, (col, label) in enumerate((("zeroed_frac", "梯度被置零的数据占比"),
                                      ("z_proj_share", "它们沿无裁剪梯度方向的份额"),
                                      ("z_norm_share", "它们的梯度长度 / 无裁剪梯度长度"))):
        vals = [window(data[n][0], LATE)[col].mean() for n, _ in CLIPS]
        ax.bar(x + (i - 1) * w, vals, w, label=label)
    ax.set_xticks(x, [f"{e:g}" for _, e in CLIPS])
    ax.set_xlabel("clip range")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("(b) 被裁掉的数据少，但它们本来贡献的梯度多（第 71–100 轮）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8.5)

    # (c) minibatch gradient consistency
    ax = axes[1, 0]
    for i, (col, label) in enumerate((("consistency_c", "带裁剪（实际使用）"), ("consistency_u", "不带裁剪（反事实）"))):
        vals = [window(data[n][0], LATE)[col].mean() for n, _ in CLIPS]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=label)
    ax.axhline(NOISE_LEVEL, color="k", ls="--", lw=1, label="32 个互不相关的小批梯度（1/32）")
    ax.set_xticks(x, [f"{e:g}" for _, e in CLIPS])
    ax.set_xlabel("clip range")
    ax.set_ylabel("‖小批梯度的平均‖² / 小批梯度‖²的平均")
    ax.set_title("(c) 一遍之内 32 个小批梯度的一致性（第 71–100 轮）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8.5)

    # (d) gradient-norm clipping: scale factor over training
    ax = axes[1, 1]
    for (name, eps), c in zip(CLIPS, colors):
        e = data[name][0]
        s = e.groupby("round")["grad_scale"].mean()
        ax.plot(s.index, s.values, color=c, lw=2 if name == "baseline" else 1.3, label=f"clip = {eps:g}")
    ax.set_yscale("log")
    ax.set_xlabel("第几轮更新")
    ax.set_ylabel("max_grad_norm / 梯度总范数（1 = 不裁剪）")
    ax.set_title("(d) 梯度范数裁剪的缩放系数（越小裁得越多）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)

    fig.suptitle("事后分析：clip range 在每轮更新里起了什么作用（3 个种子，均值）")
    fig.tight_layout()
    fig.savefig(FIG / "diag_clip.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    reproduced = reproduction_check(evidence_path=DATA / "clip_diagnostics_verification.json")
    if not reproduced:
        raise SystemExit("Incomplete or non-identical diagnostics: refusing to publish analysis.")
    data = {name: load(name) for name in CONFIGS}
    print("\nseeds per config:", {k: int(v[1]["seed"].nunique()) for k, v in data.items()})

    t = main_table(data)
    DATA.mkdir(parents=True, exist_ok=True)
    t.to_csv(DATA / "clip_diagnostics_summary.csv", index=False)
    (DATA / "clip_diagnostics.md").write_text(markdown(data))
    print("\n" + markdown(data))
    pd.set_option("display.width", 250)
    print("\nMAIN TABLE (mean over seeds, minibatches and epochs in the window)")
    print(t.round(4).to_string(index=False))

    print("\nSPREAD ACROSS SEEDS, late rounds (mean, +- half range over the 3 seeds)")
    for name, eps in CLIPS:
        e = window(data[name][0], LATE)
        r = window(data[name][1], LATE)
        parts = [f"{col} {m:.3f}+-{h:.3f}" for col in ("zeroed_frac", "z_proj_share", "z_norm_share", "consistency_c",
                                                      "consistency_u", "grad_scale")
                 for m, h in [per_seed_mean(e, col)]]
        m, h = per_seed_mean(r, "kl_exact_end")
        print(f"  clip {eps:<4g} " + ", ".join(parts) + f", kl_end {m:.4f}+-{h:.4f}")

    print("\nKL AFTER EACH EPOCH, late rounds (exact Gaussian KL on all 2048 samples)")
    for name, eps in CLIPS:
        k = window(data[name][0], LATE).groupby("epoch")["kl_exact"].mean()
        print(f"  clip {eps:<4g} " + " ".join(f"{v:.4f}" for v in k.values) + f"   epoch 10 / epoch 1 = {k.iloc[-1] / k.iloc[0]:.2f}")
    print("  same, early rounds")
    for name, eps in CLIPS:
        k = window(data[name][0], EARLY).groupby("epoch")["kl_exact"].mean()
        print(f"  clip {eps:<4g} " + " ".join(f"{v:.4f}" for v in k.values) + f"   epoch 10 / epoch 1 = {k.iloc[-1] / k.iloc[0]:.2f}")

    print("\nDISTRIBUTION OF u = sign(A)(r-1)/eps AT THE END OF A ROUND, late rounds")
    for name, eps in CLIPS:
        end = end_of_round(window(data[name][0], LATE))
        cols = ["u_lt_m1", "u_m1_0", "u_0_05", "u_05_08", "u_08_1", "u_gt_1"]
        print(f"  clip {eps:<4g} " + "  ".join(f"{c} {100 * end[c].mean():5.1f}%" for c in cols))

    print("\nGRADIENT-NORM CLIPPING (max_grad_norm 0.5, actor and critic together)")
    for name, eps in CLIPS:
        e = data[name][0]
        print(f"  clip {eps:<4g} fired on {100 * e['grad_clipped'].mean():.1f}% of updates "
              f"(lowest round {100 * e.groupby('round')['grad_clipped'].mean().min():.0f}%); "
              f"with the unclipped surrogate it would fire on {100 * e['grad_clipped_unclipped_surrogate'].mean():.1f}%; "
              f"actor norm early {window(e, EARLY)['grad_norm_actor'].mean():.2f} late {window(e, LATE)['grad_norm_actor'].mean():.2f}; "
              f"critic norm early {window(e, EARLY)['grad_norm_critic'].mean():.3f} late {window(e, LATE)['grad_norm_critic'].mean():.3f}; "
              f"scale late {window(e, LATE)['grad_scale'].mean():.3f}")

    print("\nlog_std GRADIENT (late rounds, mean per minibatch; positive = gradient descent lowers sigma)")
    for name, eps in CLIPS:
        e = window(data[name][0], LATE)
        print(f"  clip {eps:<4g} used {e['logstd_g_c'].mean():+.4f}, unclipped {e['logstd_g_u'].mean():+.4f}, "
              f"removed {e['logstd_g_z'].mean():+.4f}, removed/unclipped "
              f"{100 * e['logstd_g_z'].mean() / e['logstd_g_u'].mean():.1f}%")
    print("\nMEAN MINIBATCH DIRECTION COSINE TO EPOCH MEAN, late rounds")
    for name, eps in CLIPS:
        e = window(data[name][0], LATE)
        print(f"  clip {eps:<4g} actual {e['cos_to_mean_c'].mean():.3f}, unclipped {e['cos_to_mean_u'].mean():.3f}")

    print("\nSAME-SIGMA COMPARISON: rounds whose starting sigma is between 0.10 and 0.20")
    for name, eps in CLIPS:
        e, r = data[name]
        rr = r[(r["sigma_start"] >= 0.10) & (r["sigma_start"] <= 0.20)]
        ee = e.merge(rr[["seed", "round"]], on=["seed", "round"])
        print(f"  clip {eps:<4g} {len(rr):3d} rounds: zeroed {100 * ee['zeroed_frac'].mean():.1f}%, "
              f"z_proj_share {100 * ee['z_proj_share'].mean():.0f}%, consistency c {ee['consistency_c'].mean():.3f} "
              f"u {ee['consistency_u'].mean():.3f}, kl_end {rr['kl_exact_end'].mean():.4f}, "
              f"path efficiency {rr['path_efficiency'].mean():.2f}, grad scale {ee['grad_scale'].mean():.3f}")

    print("\nTRUE END-OF-ROUND KL vs WHAT SB3 LOGS (all rounds; SB3 averages the last epoch, measured before each step)")
    for name, n_ep in EPOCHS:
        r = data[name][1]
        print(f"  epochs {n_ep:2d}: exact KL at end of round {r['kl_exact_end'].mean():.4f}, "
              f"SB3 approx_kl {r['sb3_approx_kl'].mean():.4f}, ratio {r['kl_exact_end'].mean() / r['sb3_approx_kl'].mean():.2f}")
    for name, n_ep in EPOCHS:
        e = data[name][0]
        k = e.groupby("epoch")["kl_exact"].mean()
        print(f"  epochs {n_ep:2d}: KL after each epoch " + " ".join(f"{v:.4f}" for v in k.values))

    print("\nTRUE END-OF-ROUND KL vs WHAT SB3 LOGS, clip ranges (late rounds)")
    for name, eps in CLIPS:
        r = window(data[name][1], LATE)
        print(f"  clip {eps:<4g} exact {r['kl_exact_end'].mean():.4f}, SB3 approx_kl {r['sb3_approx_kl'].mean():.4f}, "
              f"ratio {r['kl_exact_end'].mean() / r['sb3_approx_kl'].mean():.2f}")

    print("\nLATE-ROUND KL DECOMPOSITION: action-mean movement and sigma movement")
    for name, eps in CLIPS:
        r = window(data[name][1], LATE)
        km, ks = kl_parts(r)
        assert np.allclose(km + ks, r["kl_exact_end"], atol=5e-7, rtol=2e-5)
        print(f"  clip {eps:<4g} mean KL {km.mean():.6f}, sigma KL {ks.mean():.6f}, "
              f"mean share {100 * km.mean() / r['kl_exact_end'].mean():.1f}%, "
              f"action-mean shift {r['mu_shift_abs'].mean():.6f}, "
              f"shift in old sigma units {r['mean_shift_rms_end'].mean():.6f}")
    print("\nCLIP 0.5 PERFORMANCE: original 10-seed final evaluation")
    performance = {}
    clip_final = []
    for config in ("baseline", "clip_0.5"):
        means = [json.loads((ABL / config / f"seed{s}" / "final_eval.json").read_text())["mean"] for s in range(10)]
        performance[config] = pd.DataFrame(means).mean(numeric_only=True)
        if config == "clip_0.5":
            clip_final = means
        m = performance[config]
        print(f"  {config}: cost {-m['return']:.6f}, energy {m['energy_kwh']:.6f}, discomfort {m['discomfort_kh']:.6f}")
    delta = performance["clip_0.5"] - performance["baseline"]
    print(f"  discomfort contribution to cost increase: {2 * delta['discomfort_kh']:.6f}, "
          f"share {100 * 2 * delta['discomfort_kh'] / -delta['return']:.1f}%")
    worst = int(np.argmin([m["return"] for m in clip_final]))
    curve = pd.read_csv(ABL / "clip_0.5" / f"seed{worst}" / "eval_curve.csv").set_index("timesteps")
    for step in (155_648, 204_800):
        m = curve.loc[step]
        print(f"  worst seed {worst}, step {step}: cost {-m['return']:.6f}, discomfort {m['discomfort_kh']:.6f}, energy {m['energy_kwh']:.6f}")

    print("\nCONSISTENCY BY EPOCH, late rounds (used gradient c / unclipped u); 1/32 = 0.031")
    for name, eps in CLIPS:
        g = window(data[name][0], LATE).groupby("epoch")[["consistency_c", "consistency_u"]].mean()
        print(f"  clip {eps:<4g} c " + " ".join(f"{v:.3f}" for v in g["consistency_c"]) +
              "  |  u " + " ".join(f"{v:.3f}" for v in g["consistency_u"]))

    print("\nACROSS ROUNDS: does a round undo the previous ones? (early / late window; 1 = all rounds push the same way)")
    for name, eps in CLIPS:
        r = data[name][1]
        parts = []
        for label, win in (("early", (11, 30)), ("late", LATE)):
            w = window(r, win)
            parts.append(f"{label}: cos(prev round) {w['cos_prev_round_displacement'].mean():+.3f}, "
                         f"params 10-round net/sum {w['param_efficiency_10'].mean():.3f}, "
                         f"action mean 10-round net/sum {w['mu_efficiency_10'].mean():.3f}, "
                         f"per-round action-mean shift {w['mu_shift_abs'].mean():.4f}")
        print(f"  clip {eps:<4g} " + " | ".join(parts))
    for name, n_ep in EPOCHS:
        w = window(data[name][1], LATE)
        print(f"  epochs {n_ep:2d} late: cos(prev round) {w['cos_prev_round_displacement'].mean():+.3f}, "
              f"params net/sum {w['param_efficiency_10'].mean():.3f}, action mean net/sum {w['mu_efficiency_10'].mean():.3f}")
    print("  spread across seeds, late, action mean net/sum:")
    for name, eps in CLIPS:
        m, h = per_seed_mean(window(data[name][1], LATE), "mu_efficiency_10")
        print(f"    clip {eps:<4g} {m:.3f} +- {h:.3f}")

    print("\nPATH EFFICIENCY (net displacement / path length of the actor parameters within a round)")
    for name, eps in CLIPS:
        r = data[name][1]
        e = data[name][0]
        print(f"  clip {eps:<4g} early {window(r, EARLY)['path_efficiency'].mean():.3f}, late {window(r, LATE)['path_efficiency'].mean():.3f}; "
              f"path length per round early {window(r, EARLY)['actor_path_length'].mean():.4f} late {window(r, LATE)['actor_path_length'].mean():.4f}; "
              f"net displacement early {window(r, EARLY)['actor_net_displacement'].mean():.4f} late {window(r, LATE)['actor_net_displacement'].mean():.4f}; "
              f"cos between consecutive steps late {window(e, LATE)['actor_step_cos_prev'].mean():+.3f}")

    print("\nSHARE OF THE END-OF-ROUND KL ALREADY REACHED AFTER THE FIRST EPOCH")
    for name, eps in CLIPS:
        e = data[name][0]
        parts = []
        for label, win in (("early", EARLY), ("late", LATE)):
            k = window(e, win).groupby("epoch")["kl_exact"].mean()
            parts.append(f"{label} {k.iloc[0] / k.iloc[-1]:.0%}")
        print(f"  clip {eps:<4g} " + ", ".join(parts))

    print("\nDENSITY OF u JUST INSIDE THE BOUNDARY (fraction per unit of u), end of round, late rounds")
    for name, eps in CLIPS:
        end = end_of_round(window(data[name][0], LATE))
        print(f"  clip {eps:<4g} (0, 0.5] {end['u_0_05'].mean() / 0.5:.3f}   (0.5, 0.8] {end['u_05_08'].mean() / 0.3:.3f}   "
              f"(0.8, 1] {end['u_08_1'].mean() / 0.2:.3f}   beyond 1: {100 * end['u_gt_1'].mean():.1f}% of samples")

    figure(data)
    print("\nreproduced exactly:", reproduced)
