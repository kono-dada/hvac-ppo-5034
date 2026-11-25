# 任务报告

## 交付

- 第 3 节新增 3.3 "Design choices, tested beforehand"（`<sec-pre>`）和表 2：默认设置、午夜截断、不观测墙温、舒适权重 0.5 / 8、奖励 ×0.1、奖励归一化、1.5M 步，各 5 个种子，全部按 J = E + 2D 打分。
  - 正文分四段：午夜为终止状态、观测墙温、舒适权重取 2、不缩放奖励。
  - 3.1 和预算说明里原先手写的 4.54 / 5.19 / 4.23 改为从数据读取。预算说明补了一句：1.5M 步时已优于调参后的恒温器（4.23 对 4.48）。
- 5.3 新增通栏图 4 [ablation-clip-diagnostics.typ](../../../docs/report/typst/figures/ablation-clip-diagnostics.typ)，含四个面板：
  - (a)、(b)：10 个种子下 `approx_kl` 和 `clip_fraction` 随训练的变化，取自原 update-size 图中 clip 的部分；
  - (c)：被置零样本的占比及其梯度长度占比；
  - (d)：一轮更新内每一遍的 KL。
  - 正文删去被图取代的数字：ε=0.05/0.5 的占比和角度、余弦 0.53→0.22、"65% 长度"、μ 位移 2.4 倍。
- [export_report_data.py](../../../scripts/export_report_data.py) 新增 `pre`（预实验汇总）和 `diagnostics.late.*.kl_by_epoch`（第 71–100 轮每一遍的 KL）。
- `ksteps()` 遇到 ≥1M 的值输出 "1.5M"。
- [build_typst_figures.sh](../../../scripts/build_typst_figures.sh) 注册了新图。

## 验证

- 完整导出后，report.json 原有字段与改动前逐项相同，只新增 `pre` 和 `kl_by_epoch`。
- `scripts/build_report.sh` 编译通过，PDF 共 10 页（含参考文献，正文约 9.5 页）。逐页检查了新表、新图和交叉引用。
- 修复了 Typst 在 `#num(...)` 后吞掉分号的问题，改为断句。
- `pytest`：39 passed。

## 注意

- 预实验的预算（500k 步、5 个种子）与消融实验不同，已在表注和正文注明。
- 奖励缩放为什么变差没有调查，正文如实写明。
- 表号顺延：原表 2 / 表 3 现在是表 3 / 表 4，原图 4 现在是图 5，都通过引用自动更新。
