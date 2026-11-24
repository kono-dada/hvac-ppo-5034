# 任务报告 task-20261125-05：重新设计 Typst 图表的样式与可视化方式

完成时间：2026-11-25 06:05（Asia/Shanghai）。用户要求提前收工，`method-loop` 没有重画（见“待解决”）。

## 实际交付

### 共享样式 [style.typ](../../../docs/report/typst/figures/style.typ)（重写）

- 中性色为主：墨色 `#1d232b` 用于正文、基线和理论最优；网格只保留浅色横线；去掉 y 轴脊线和 y 轴刻度线（通过透明的 y 轴 stroke 实现，lilaq 的 tick 设置是全局的，无法按轴分别关掉）。
- 语义颜色：温控器橙色 `#e8832a`；制热红、制冷蓝、舒适区绿；空气用墨色，墙体和地板用棕色 `#9a6a3f`，室外用灰蓝色，太阳用琥珀色；预测区间用紫色 `#7d68c6`。判定颜色和报告模板一致：命中绿、部分命中琥珀、不命中红。
- 消融配色：小于基线的取值用三级蓝，大于基线的用三级橙红，基线为墨色。
- 公用小部件：`panel`（面板字母和单位）、`tag`、`swatch`、`legend-row`、`verdict-mark`（实心圆 / 半实心圆 / 叉，不只靠颜色区分）、`dodge`（末端标签防重叠）、`heading-block`。

### 9 个图模块

| 模块 | 改动 |
| --- | --- |
| [env-step-response](../../../docs/report/typst/figures/env-step-response.typ) | 空气黑线、墙体棕线，加热时段用红色底带标出；22.4、16.3、17.7 °C 直接标在曲线上，并标出“关机 7 小时后两者都在 16.5 °C”；右上角注明模型的两个时间尺度 0.6 h 和 39 h |
| [env-days](../../../docs/report/typst/figures/env-days.typ) | 用蓝–灰–红发散色表示每天的平均气温，顶部色条上用刻度标出这 12 天；舒适区只画在工作时间内；右侧面板是日照 |
| [env-controllers](../../../docs/report/typst/figures/env-controllers.typ) | 温度单位直接写在刻度上；温控器的动作画成半透明面积，理论最优画成黑线；每个面板下面加一张当天的耗电和不舒适度小表 |
| [env-schematic](../../../docs/report/typst/figures/env-schematic.typ) | 按新配色重画（空气中性色、墙体棕色、室外浅灰蓝）；补上热容 1 MJ/K 和 10 MJ/K、两个时间尺度、COP 和耗电 |
| [ablation-final](../../../docs/report/typst/figures/ablation-final.typ) | 改成森林图：每个取值一行，按超参数分组；三列分别是成本、耗电、不舒适度，共用横轴；每行画 10 个种子的点、均值和 95% CI，再叠加紫色的预测区间；成本列右边是判定标记；基线的 CI 画成贯穿各行的竖带，温控器画成参考线；epochs = 1 超出范围，用数字和箭头表示 |
| [ablation-learning-curves](../../../docs/report/typst/figures/ablation-learning-curves.typ) | 3×3 小多图，每组超参数的图例放在该行的行首；去掉第 0 步的评估点（未训练的策略，数值远超坐标范围） |
| [ablation-tradeoff](../../../docs/report/typst/figures/ablation-tradeoff.typ) | 基线的训练路径保持黑色；标签改成短的直接标注；四条等成本线（温控器 4.48、基线 4.73、5.0、5.5）；基线终点改成黑色大圆点（原来是红色方块）；温控器和理论最优改用菱形 |
| [ablation-update-size](../../../docs/report/typst/figures/ablation-update-size.typ) | 线末直接标注取值，不再用图例；做 5 轮滑动平均；clip_fraction 为 0 时用下限代替；epochs = 1、3 的 approx_kl 仍用虚线并加文字说明；标出 ε 从 0.05 到 0.5 时训练期均值相差约 50 倍 |
| [method-loop](../../../docs/report/typst/figures/method-loop.typ) | 未改（见“待解决”） |

### 数据导出 [export_figure_data.py](../../../scripts/export_figure_data.py)

- `env.json` 的 config 新增：`c_air`、`c_mass`、`cop`、`q_solar_peak`、`q_internal_occupied`、`q_internal_idle`，以及 `time_constants_h`，即离散系统矩阵特征值换算出的时间常数，为 [0.594, 38.793] h。
- `ablation.json` 新增两项：
  - `verdicts`：每个配置的成本预测判定，来自 `analyze_ablation.verdict_table`；
  - `summary`：approx_kl、clip_fraction 和回退的均值与 95% CI，来自 `summary_table`。
- 原有字段和数值都没有改。

## 验证

- `SKIP_EXPORT=1 scripts/build_typst_figures.sh` 一次构建了 9 张图，没有报错，也没有警告。
  - `ablation-update-size` 曾出现 lilaq 布局不收敛的警告，原因是基线的末端标签超出了数据区；改短标签后警告消失。
- 逐张查看了 `docs/report/typst/build/*.png`，确认没有标签重叠、没有元素被裁切，颜色语义与上表一致。
- `figures-preview.typ` 和正式报告 `report.typ` 都能编译通过（报告输出到 `/tmp`，没有覆盖 `docs/report/report.pdf`），没有错误或警告。

## 颜色和语义变化（正式报告改图注时注意）

- 空气温度曲线由橙色改为黑色，墙体由深色改为棕色虚点线。如果图注或正文写了“橙线是空气”，需要更新。
- `ablation-tradeoff` 的基线终点由红色方块改为黑色大圆点；正文说“黑线是基线的训练路径”，这一点仍然成立。
- 温控器仍是橙色，理论最优和基线仍是黑色或墨色。
- `ablation-learning-curves` 和 `ablation-tradeoff` 的图注不需要改。

## 待解决

- `method-loop` 还是旧样式（三张彩色卡片），和其他图的配色不完全统一，内容没有问题。
- 正式报告的 `docs/report/report.pdf` 由 task-04/06 负责，没有重新生成，里面仍是旧图。
- 图在单栏（约 8.3 cm）宽度下的效果没有单独检查。报告目前用 `shrink: 0.8` 把图放在通栏。
