# 任务报告 task-20261125：铺开消融实验

完成时间：2026-11-25 01:40（Asia/Shanghai）

## 实际交付

按时间顺序，分三次提交，保证“规则先于结果、结果先于分析”在 git 记录里可查：

| commit | 内容 | 当时是否已有非基线结果 |
| --- | --- | --- |
| `ab7f73b` | 预注册修订：10 个种子、判定规则、调参温控器、去掉“追上温控器的步数” | 否 |
| `42917a8` | 原始结果：12 种配置 × 10 个种子 | 有结果，未分析 |
| 本次最后一次提交 | 分析脚本输出、图、3.6 节结果与分析 | 是 |

### 1. 调参温控器（用户决定 4）

- [scripts/tune_thermostat.py](../../../scripts/tune_thermostat.py)：在 200 个独立日子（种子 300000–300199，既非训练日也非测试日）上网格搜索 378 种组合。最优：提前 2.5 小时开机，余量 1.0 K，增益 0.75；最优点不在网格边缘。
- [hvac/controllers.py](../../../hvac/controllers.py)：新增 `TUNED_THERMOSTAT` 和 `tuned_thermostat()`。类的默认参数没有改，旧的预实验脚本结果不受影响。
- 100 个测试日上：耗电 4.24 kWh、不舒适度 0.12 K·h、违规 3.6%、成本 4.48（未调参版本 5.37）。
- [01_environment.md](../../../docs/report/zh/01_environment.md) 1.5 节对照表加了这一行，并说明“下文的温控器指调过参数的版本”。

### 2. 预注册修订（用户决定 1、2、5）

[03_ablation.md](../../../docs/report/zh/03_ablation.md)：

- 文首加修订说明，列出 4 项改动和原文 commit `8a7a3a4`。
- 3.1 节：种子数改为 10；指标表去掉“追上温控器的步数”，回退的算法写精确；补充判定规则（δ 表、各种说法的判定表、不做多重比较校正的说明）。
- 3.4 节末：54 条预测与判定规则的逐条对应表。
- “最好的温控器（5.37）”改称“未调参的温控器”。预测的数值区间都没有动。

统计代码在 [hvac/stats.py](../../../hvac/stats.py)，测试在 [tests/test_stats.py](../../../tests/test_stats.py)（Welch 区间与 scipy 对照，各种判定的正反例）。

### 3. 实验

- [scripts/run_ablation.py](../../../scripts/run_ablation.py)：`SEEDS = range(10)`。
- 新跑 115 次（基线种子 5–9 + 11 种配置 × 10），10 个进程并行，约 7 分钟。单次训练 11 秒（epochs = 1）到 83 秒（epochs = 30）。

### 4. 分析与结果

- [scripts/analyze_ablation.py](../../../scripts/analyze_ablation.py)：读取全部结果，按预注册规则自动判定，生成
  - [ablation_summary.md](../../../docs/report/data/ablation_summary.md) / `.csv`（总表）
  - [ablation_verdicts.md](../../../docs/report/data/ablation_verdicts.md) / `.csv`（54 条逐条判定）
  - 图：[ablation_final.png](../../../docs/report/figures/ablation_final.png)、[ablation_learning_curves.png](../../../docs/report/figures/ablation_learning_curves.png)、[ablation_tradeoff.png](../../../docs/report/figures/ablation_tradeoff.png)、[ablation_update_size.png](../../../docs/report/figures/ablation_update_size.png)
  - 并打印 3.6 节引用的所有其他数字。
- [scripts/lambda_time_of_day.py](../../../scripts/lambda_time_of_day.py)：把 lambda 各配置的耗电和不舒适度按时段拆开（含理论最优和温控器作参照）。
- 03_ablation.md 3.6 节：总表、逐条判定、按参数的分析（与预测不同之处及原因）、训练稳定性、总结。

**判定结果**：54 条中命中 31、部分命中 3、不命中 5、无法判断 15。

**主要发现**：

1. epochs：方向全对。3 遍比预测好（5.38，预测 5.4–6.4），原因是用基线早期进度换算低估了少遍数的效率。
2. clip range：错得最多。“clip_fraction 低 = 裁剪不起作用”是误读：`approx_kl` 随 clip range 近似成比例变化，每轮更新大致停在边界附近。0.05 慢得比预测少；0.5 明显不稳定（成本 +0.37，种子间标准差 3.9 倍，回退 6 倍），正是 3.5 节列出的风险。
3. GAE lambda：成本 5 条全中，但 0.8/0.9 的耗电与不舒适度并不“相当”。lambda 控制的是预热多少：小了预热不足、早上越界；大了预热过多、多耗电。
4. 基线 10 个种子是 4.73 ± 0.12，比预测时 5 个种子的 4.70 ± 0.05 更分散（种子 5 是 5.04）。
5. 20 万步时没有一种配置平均胜过调参温控器（4.48），只有 epochs = 30 的 2 个种子做到。

## 验证

- `pytest`：21 项全部通过（原 14 项 + 新增统计测试 7 项）。
- 分析脚本在只有基线结果时先干跑过一遍，确认能正常读取和输出；干跑产物在预注册提交前已删除。全部结果到齐后，54 条预测全部得到判定（脚本输出 `predictions defined: 54, judged: 54`）。
- 回退的定义先在基线 5 个种子上复现了预测时报告的 0.03–0.09。
- 3.6 节的数字都来自上述两个脚本的输出，两张大表从生成的文件直接插入。写完后又逐条核对了正文里的数字，改正了 3 处与脚本输出不一致或没有算过的说法。

## 待解决

1. **第 2 节和消融设置不一致（同事负责的文件，未改）。** [02_method.md](../../../docs/report/zh/02_method.md) 2.3 节仍写 50 万步、每 20,480 步在 30 个测试日上评估、约 85 秒、245 轮；[make_method_figures.py](../../../scripts/make_method_figures.py) 的 `TOTAL_STEPS = 500_000` 和 `method_loop.png` 同样如此。消融实际是 204,800 步（100 轮）、每 8,192 步在 100 个测试日上评估、基线单次约 31 秒。需要和同事商量以哪个为准。
2. 第 1 节的 [make_environment_figures.py](../../../scripts/make_environment_figures.py) 已加入调参温控器的数值输出，但 `env_controllers.png` 这张图里没有画调参温控器，是否要加由第 1 节负责人决定。
3. 3.6 节里有三处解释明确标为“未验证”：clip_fraction 低的机制、clip = 0.05 进度快于每轮改动比例的原因、大 lambda 预热过多的原因；另外 epochs = 1 后半程评估成本来回晃的原因没有查清。如果报告需要，可以针对性地补实验。
4. “回退”这个指标在曲线走平的配置上会偏大（epochs = 1），在曲线下降快的配置上会偏小，3.6.6 节已经说明；如果要换一个更好的稳定性指标，需要另行约定，且只能作为事后分析。
