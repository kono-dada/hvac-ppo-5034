# 任务记录 task-20261125-02：报告图表美化

记录时间：2026-11-25（Asia/Shanghai）

## 用户要求

用户原话：“OK，看看我们的仓库——实验基本完毕，我们该着手最终的report了。首先是把图都变得更美观一点，新开一个任务，把关键的图片都用更美观的方式呈现。”

## 范围

报告正文（[01_environment.md](../../docs/report/zh/01_environment.md)、[02_method.md](../../docs/report/zh/02_method.md)、[03_ablation.md](../../docs/report/zh/03_ablation.md)）引用的全部图，以及 [ablation_update_size.png](../../docs/report/figures/ablation_update_size.png)（这张图由分析脚本生成，但正文目前没有引用）：

| 图 | 生成脚本 |
| --- | --- |
| env_schematic / env_step_response / env_days / env_controllers | [make_environment_figures.py](../../scripts/make_environment_figures.py) |
| method_loop | [make_method_figures.py](../../scripts/make_method_figures.py) |
| ablation_final / ablation_learning_curves / ablation_tradeoff / ablation_update_size | [analyze_ablation.py](../../scripts/analyze_ablation.py) |

`tasks/task-20261124/report/figures/` 里的预实验图不在报告里，不处理。

## 约束

- 只改呈现方式，不改数据和结论。三个脚本打印的数字（正文引用的数字）必须和改之前逐字相同。
- 文件名不变，正文的图片链接不用改。
- 图上的文字保持中文，和目前的中文稿一致。
- 02_method.md 属于另一位同事，只重绘它引用的 method_loop.png（内容不变），不改正文。
- 统一风格的代码放进一个共享模块，三个脚本共用。

## 交付物

1. 共享绘图风格模块，三个脚本改为使用它。
2. 重新生成的 9 张图。
3. 本目录下的 `report/report.md`：改了什么、怎么验证的、还有哪些待定。

## 补充要求（第 2 轮，2026-11-25 04:00）

用户原话：“感觉还是不够美观。这样，我的最终报告打算使用typst，你看看能不能用typst做出这些图来。图片内的字符都换成英语。”

追加：“你去找找看看能不能直接下载二进制和相关文档到当前目录里”（用户拒绝用 Homebrew 往系统里装 typst）。

### 做法

- typst 二进制（0.15.1）放在仓库内 `tools/typst/`，包缓存放在 `tools/typst/packages/`（设 `TYPST_PACKAGE_CACHE_PATH`），文档放在 `tools/docs/`。不往系统里装东西。
- Python 只负责算数据：三个脚本额外导出画图用的数据（JSON/CSV），不改已有打印与已有表格。
- 每张图一个 `.typ` 文件，共用一个样式文件，用 lilaq（数据图）和 CeTZ（示意图）画。每个文件既能单独编译成 PDF/SVG/PNG 预览，也能被最终的 typst 报告直接 `#include`/导入。
- 图内文字全部换成英文。

### 约束（第 2 轮）

- 原来的 matplotlib PNG 仍被中文稿引用，保留不动；typst 版本输出到新目录。
- 03_ablation.md 正被 task-20261125-03 修改，本任务不改正文；不碰 task-03 的 `diag_clip.png`。
- 数据与结论不变：数字直接来自脚本导出的数据，不在 typst 里手抄。

### 交付物（第 2 轮）

4. 仓库内可用的 typst 工具链与文档（`tools/`）。
5. 9 张图的 typst 版本（英文），以及一键构建脚本。
6. 更新 `report/report.md`。
