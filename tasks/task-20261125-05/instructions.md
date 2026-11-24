# 任务记录 task-20261125-05：重新设计 Typst 图表的样式与可视化方式

记录时间：2026-11-25 04:58（Asia/Shanghai）

## 用户要求

用户原话：“看看 docs/report/typst/figures，这些图片还是感觉不好看啊，无论是样式还是配色。我感觉你可以做的更好！不要被已有的样式束缚，用最佳的方式可视化这些数据。”

即：task-20261125-02 用 Typst 重画的 9 张图，样式和配色都不满意。这次不受原样式约束，重新设计每张图，用最合适的图形表达数据。

## 范围

[`docs/report/typst/figures/`](../../docs/report/typst/figures/) 下的共享样式 `style.typ` 和 9 个图模块：

| 模块 | 内容 |
| --- | --- |
| `env-schematic` | 房间的两个热容节点与热流 |
| `env-step-response` | 满功率加热 2 小时的阶跃响应 |
| `env-days` | 随机抽到的 12 天天气 |
| `env-controllers` | 冷天、热天上的三种控制方式 |
| `method-loop` | PPO 的一轮训练 |
| `ablation-learning-curves` | 消融实验的学习曲线 |
| `ablation-final` | 训练结束时的结果与预测区间 |
| `ablation-tradeoff` | 耗电与不舒适度的权衡 |
| `ablation-update-size` | 每轮更新的大小（approx_kl、clip_fraction） |

可以改图的类型、布局、配色、字体用法和标注方式。需要新数据时扩展 `scripts/export_figure_data.py`。

## 约束

- 不改数据和结论。图里的数字全部来自 `figures/data/*.json`（由 `scripts/export_figure_data.py` 从实验结果导出），不手抄。
- 图中文字用英文（服务于英文正式报告）。
- 保持接口：文件名不变，每个模块导出 `render(width: text-width, title: true)`；正式报告（task-20261125-04）和 `figures-preview.typ` 用 `render(title: false)` 导入。
- 保留数据的关键语义：基线突出显示、温控器与理论最优两条参照、epochs = 1/3 的 approx_kl 偏低需标注、3.4 节的成本预测区间、10 个种子的 95% 置信区间、工作时间 8–18 点、舒适区 21–24 °C、制热与制冷的区分。
- 只用仓库内的工具链：`tools/typst/`（Typst 0.15.1，lilaq 0.6.0，CeTZ 0.5.3）和 `tools/fonts/`（Fira Sans / Fira Math），不往系统里装东西。
- 不碰其他任务的文件：task-20261125-03 的诊断文件（`hvac/diagnostics.py`、`runs/diagnostics/`、`scripts/*clip_diagnostics.py`、`docs/report/data/clip_diagnostics*`、`docs/report/figures/diag_clip.png`）；task-20261125-04 的报告文件（`docs/report/typst/template.typ`、报告正文、`refs.bib`、`data/`）；`docs/report/figures/` 下 matplotlib 生成的 PNG 和中文稿。
- 不提交 git，除非用户要求。

## 补充要求（05:58）

用户原话：“快点开始修改”，之后：“总之先收工吧，没时间了。先把figure相关的提交”。

即：停止继续打磨，把和图相关的改动提交到 git。提交范围只含图：Typst 图模块与其数据、构建脚本、字体，以及 task-20261125-02 的 matplotlib 图美化（同属图的工作）。不提交 task-03 的诊断文件、task-04/06 的报告正文与版式（`lib.typ`、`report.typ`、`template.typ`、`report.pdf`）和 `03_ablation.md`。

## 交付物

1. 重写后的 `docs/report/typst/figures/style.typ` 与 9 个图模块。
2. 用 `scripts/build_typst_figures.sh` 重新生成 `docs/report/typst/build/<name>.{pdf,svg,png}`，并逐张检查。
3. `docs/report/typst/build/figures-preview.pdf`：图嵌入 A4 页面（`title: false`）的效果。
4. 本目录下的 `report/report.md`：实际交付、验证结果、待解决事项。颜色或图形语义如有变化，写明，方便正式报告据此改图注。
