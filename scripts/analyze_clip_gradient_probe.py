"""Summarize same-state clip-gradient and shadow-Adam probes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "gradient_probe"
DATA = ROOT / "docs" / "report" / "data"


def summarize(e, a):
    mean2, mean5 = e["mean_norm_h2"].mean(), e["mean_norm_h5"].mean()
    mu2, mu5 = np.sqrt((a["mu_step_rms_02"] ** 2).mean()), np.sqrt((a["mu_step_rms_05"] ** 2).mean())
    wmu, wp = a["mu_step_rms_05"] ** 2, a["param_step_05"] ** 2
    return {
        "extra_samples_frac": e["extra_samples_frac"].mean(),
        "gradient_norm_02": e["norm_g2"].mean(), "gradient_norm_05": e["norm_g5"].mean(),
        "extra_gradient_norm_share": e["added_norm_share"].mean(),
        "extra_gradient_projection": e["added_proj_g5"].mean(),
        "gradient_angle_deg": np.degrees(np.arccos(e["gradient_cos_g2_g5"].mean())),
        "consistency_raw_02": e["consistency_g2"].mean(), "consistency_raw_05": e["consistency_g5"].mean(),
        "consistency_clipped_02": e["consistency_h2"].mean(), "consistency_clipped_05": e["consistency_h5"].mean(),
        "mean_clipped_gradient_norm_02": mean2, "mean_clipped_gradient_norm_05": mean5,
        "mean_clipped_gradient_norm_ratio": mean5 / mean2,
        "extra_mean_projection_raw": e["added_mean_proj_raw"].mean(),
        "extra_mean_projection_clipped": e["added_mean_proj_norm_clipped"].mean(),
        "extra_action_mean_gradient_projection_clipped": e["added_mu_mean_proj_norm_clipped"].mean(),
        "extra_descent_alignment": e["added_cos_to_actual_descent"].mean(),
        "adam_param_step_02": a["param_step_02"].mean(), "adam_param_step_05": a["param_step_05"].mean(),
        "adam_param_step_cos": a["param_step_cos"].mean(),
        "adam_extra_param_projection": np.average(a["param_extra_proj"], weights=wp),
        "adam_mu_step_rms_02": mu2, "adam_mu_step_rms_05": mu5,
        "adam_mu_step_ratio": mu5 / mu2,
        "adam_mu_step_cos": a["mu_step_cos"].mean(),
        "adam_extra_mu_projection": np.average(a["mu_extra_proj"], weights=wmu),
        "restore_cos_02": e["restore_cos_g02"].mean(), "restore_cos_05": e["restore_cos_g05"].mean(),
        "restore_dot_clipped_02": e["restore_dot_h02"].mean(),
        "restore_dot_clipped_05": e["restore_dot_h05"].mean(),
        "momentum_outward_cos": e["momentum_outward_cos"].mean(),
        "actual_step_outward_cos": e["step_outward_cos"].mean(),
        "distance_delta_02": a["distance_delta_02"].mean(),
        "distance_delta_05": a["distance_delta_05"].mean(),
    }


if __name__ == "__main__":
    evidence = json.loads((OUT / "verification.json").read_text())
    assert evidence["all_passed"]
    es, ads, rows = [], [], []
    for seed in range(3):
        run = OUT / "clip_0.5" / f"seed{seed}"
        e = pd.read_csv(run / "gradient_epochs.csv")
        e = e[e["round"].between(71, 100)]
        a = pd.read_csv(run / "adam_probes.csv")
        assert len(e) == 300 and len(a) == 600
        es.append(e.assign(seed=seed)); ads.append(a.assign(seed=seed))
        rows.append({"seed": str(seed), **summarize(e, a)})
    all_e, all_a = pd.concat(es), pd.concat(ads)
    rows.append({"seed": "pooled", **summarize(all_e, all_a)})
    table = pd.DataFrame(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    table.to_csv(DATA / "clip_gradient_probe.csv", index=False)
    row = rows[-1]
    md = ["# clip = 0.5 同状态梯度诊断", "",
          "三个种子，第 71–100 轮。每小批在相同参数和数据上计算两个目标的梯度；Adam 对照从实际当前状态出发，每遍在第 16、32 个小批各做一次独立单步，共 1,800 对。", "",
          "| 量 | clip = 0.2 | clip = 0.5 |", "| --- | --- | --- |",
          f"| 梯度与回拉方向的余弦 | {row['restore_cos_02']:.3f} | {row['restore_cos_05']:.3f} |",
          f"| 范数裁剪后的回拉强度 | {row['restore_dot_clipped_02']:.5f} | {row['restore_dot_clipped_05']:.5f} |",
          f"| 裁剪后的梯度一致性 | {row['consistency_clipped_02']:.4f} | {row['consistency_clipped_05']:.4f} |",
          f"| 每遍平均梯度范数（范数裁剪后） | {row['mean_clipped_gradient_norm_02']:.5f} | {row['mean_clipped_gradient_norm_05']:.5f} |",
          f"| 同状态 Adam 单步参数位移 | {row['adam_param_step_02']:.6f} | {row['adam_param_step_05']:.6f} |",
          f"| 同状态 Adam 单步动作均值 RMS 改动 | {row['adam_mu_step_rms_02']:.6f} | {row['adam_mu_step_rms_05']:.6f} |",
          f"| 同状态 Adam 单步漂移 D 的变化 | {row['distance_delta_02']:+.3e} | {row['distance_delta_05']:+.3e} |", "",
          "回拉梯度定义为 D = 1/2 × mean((当前动作均值 − 本轮开始动作均值)²) 的参数梯度；沿其负方向更新会降低漂移。回拉强度为范数裁剪后的策略梯度与该梯度的点积。", "",
          f"放宽阈值使回拉强度下降 {100 * (1 - row['restore_dot_clipped_05'] / row['restore_dot_clipped_02']):.1f}%。", "",
          f"放宽阈值释放的数据：{100 * row['extra_samples_frac']:.1f}%；释放梯度长度 / 实际梯度长度：{100 * row['extra_gradient_norm_share']:.1f}%；梯度转向角：{row['gradient_angle_deg']:.1f}°。", "",
          f"释放部分在平均梯度方向上的投影份额（范数裁剪后）：{100 * row['extra_mean_projection_clipped']:.1f}%；平均梯度范数比：{row['mean_clipped_gradient_norm_ratio']:.2f}。", "",
          f"同状态单步动作均值 RMS 比：{row['adam_mu_step_ratio']:.2f}；释放梯度带来的差异沿实际动作均值步方向的份额：{100 * row['adam_extra_mu_projection']:.1f}%。", ""]
    (DATA / "clip_gradient_probe.md").write_text("\n".join(md))
    print(table.to_string(index=False))
    print("\n" + "\n".join(md))
