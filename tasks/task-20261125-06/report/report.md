# 任务报告 task-20261125-06：统一图中字体，图 1 改为单栏

完成时间：2026-11-25 06:30（Asia/Shanghai）。任务要求见 [instructions.md](../instructions.md)。

## 结论

- 图中文字改用和正文一样的 New Computer Modern（数学用 New Computer Modern Math），不再用 Fira Sans。
- 所有图都按原尺寸放进报告，不再用 `shrink: 0.8` 缩放，所以图中文字在页面上统一是 8 pt，和表格一样大（图注 9 pt，正文 10 pt）。
- 图 1（房间示意图）重新排版，放进单栏。
- 报告仍是 8 页。PDF 内嵌的字体只剩 New CM、New CM Math，以及代码名用的 DejaVu Sans Mono。

## 改动

### 共享样式 [style.typ](../../../docs/report/typst/figures/style.typ)

- 字体：`font = "New Computer Modern"`，`math-font = "New Computer Modern Math"`；行高的计算方式和报告里的图保持一致（cap-height 到 baseline）。
- 字号：统一用 `base-size = 8pt`。原来各模块里在 0.9–1.12 em 之间的零散字号（刻度、坐标轴标签、图例、标题、注释）全部去掉，统一成 1 em。只保留两处特殊字号：判定标记的符号大小，以及独立构建时的大标题。
- 字重：这套字体只有常规和粗体两种，原来的 semibold 一律改成 bold，medium 改成 regular。

### 图 1 [env-schematic.typ](../../../docs/report/typst/figures/env-schematic.typ)（重画）

- 版面按单栏 8.25 cm 设计：空气和墙体左右并排；空气通过窗户向上与室外换热，墙体通过外墙向下与室外换热，热泵放在空气下方，太阳光从上方射入（30% 进空气，70% 进墙体）。
- 数值和含义都没变：两个热容、三个热导、两个时间常数、热泵功率、COP、太阳光峰值、内部热源。
- 原来的 `≈`（`sym.tilde.op`）新字体里没有，会回退到 Libertinus，所以改用数学字体来排。
- 在报告中改用新的 `column-figure`（[lib.typ](../../../docs/report/typst/lib.typ)），放在 2.1 节右栏，紧挨引用它的文字。

### 不再缩放，高度相应调低（保持页数不变）

| 图 | 绘图区高度 |
| --- | --- |
| env-controllers | 4.9 → 4.0 cm（温度），2.0 → 1.65 cm（动作） |
| ablation-learning-curves | 3.1 → 2.5 cm |
| ablation-tradeoff | 6.4 → 5.2 cm |

报告里 3 处 `shrink: 0.8` 已删除（[02-environment.typ](../../../docs/report/typst/sections/02-environment.typ)、[05-results.typ](../../../docs/report/typst/sections/05-results.typ)）。

### 换字体后的小修

- [ablation-tradeoff](../../../docs/report/typst/figures/ablation-tradeoff.typ)：等成本线标签、“zoomed in b”、“baseline path”加了白底，避免与虚线重叠。
- [env-step-response](../../../docs/report/typst/figures/env-step-response.typ)：“Air 22.4 °C when the heat pump stops”折成两行。新字体比原来的宽，这行会撞上右上角的说明框。
- [template.typ](../../../docs/report/typst/template.typ) 里说明图用 Fira Sans 的注释已更新；[figures-preview.typ](../../../docs/report/typst/figures-preview.typ) 的正文字体也改成了 NCM。

## 验证

- `SKIP_EXPORT=1 scripts/build_typst_figures.sh`：9 张图全部构建成功，没有警告。
- `SKIP_EXPORT=1 scripts/build_report.sh`：报告 8 页，没有警告。逐页看过 `build/report-pages/`，另外逐张看过独立构建的 9 张 PNG：没有文字重叠或被裁切。
- PDF 内嵌字体（`strings report.pdf | grep BaseFont`）：只有 NewCM、NewCMMath 和 DejaVuSansMono，没有 Fira，也没有回退到 Libertinus。
- `figures-preview.typ` 编译通过。
- 没改数据，没改正文文字（只动了图的摆放代码）。

## 待解决

- 报告还是 8 页，比约定的 6 页多两页。这在本任务之前就是这样，原因是版式改成了 LaTeX article 风格，本任务没有处理。
- 工作区里有其他任务没提交的改动（`template.typ`、`report.typ`、`05-results.typ` 的表格列宽、task-03 的诊断文件等）。本任务没有提交任何东西，也没有回退这些改动。
- `method-loop` 只跟着换了字体，内容仍是预实验的 245 轮 / 50 万步（报告没有用这张图）。

## 提交（2026-11-25 11:25，用户：“OK，提交”）

- 提交范围：本任务的图字体与图 1 改动，以及工作区里此前未提交的正文版式改动。后者包括 `template.typ`（LaTeX article 风格）、`report.typ`（封面字段）、`lib.typ`（图内行距），以及 `05-results.typ`（表 2 列宽自动）。这些版式改动没有任务记录，但图字体统一正是以它为前提，拆开提交会让正文和图的字体又对不上，所以一并提交。另外提交了重新生成的 `report.pdf` 和本任务记录。
- 未提交：task-20261125-03 的诊断文件（`hvac/diagnostics.py`、`hvac/gradient_probe.py`、`hvac/train.py`、`docs/report/zh/03_ablation.md`、`docs/report/data/clip_*`、`runs/diagnostics/` 等）和 task-20261125-07 的审阅记录。
- 提交前重新运行了 `scripts/build_report.sh` 和 `scripts/build_typst_figures.sh`，均无报错；PDF 与源文件一致。
