# 任务报告 task-20261125-08：重排报告图 4

## 实际交付

改动文件：

- [`ablation-tradeoff.typ`](../../../docs/report/typst/figures/ablation-tradeoff.typ)：图 4 重排。
- [`05-results.typ`](../../../docs/report/typst/sections/05-results.typ)：图注改成与新图一致。
- 重新生成的 [`report.pdf`](../../../docs/report/report.pdf) 与 `docs/report/typst/build/ablation-tradeoff.{pdf,svg,png}`。
- 对比图：[改前](fig4-before.png)、[改后](fig4-after.png)（按报告通栏宽度 17.2 cm 单独渲染）。

具体改动：

1. **baseline 训练路径**
   - 起点（16k 步，更早的点远超坐标范围）画成空心圆，并标注“start (16k steps)”。
   - 终点（205k 步）画成大实心点，在 b 中标注“baseline end (205k steps)”。
   - 路径上加箭头表示训练方向：a 中 2 个，b 中 2 个。
   - 删去原来右下角那段“baseline path, training runs ←”文字，改由箭头表达方向。
2. **a 图去掉重复内容**：其他设置的终点已在 b 中放大显示，a 中不再重复，只保留 b 范围之外的“1 epoch”和 oracle，以及表示 b 范围的灰框（标注改为“panel b”）。原来挤在一起的十个标记都去掉了。
3. **b 图标签重新布置**
   - 每个标签放在点的斜上方或斜下方，不压点，也不压自身置信区间的横竖线。
   - “tuned thermostat”拆成两行，放在点的左下方。
   - 标签之间互不重叠。
   - 置信区间线变细（1.2 → 0.9 pt）。
   - x 轴刻度从每 0.1 改为每 0.2。
   - 等成本线标注改为“cost 4.73 (baseline)”这种形式。
4. **图例**
   - 原来的“■▲● vary …”拆成三项：n_epochs、clip range ε、GAE λ。
   - baseline 一项的图示改成“空心圆—箭头—实心点”，与图中一致。
   - 全部图例放在一行。
5. 图高从 5.2 cm 增至 6.0 cm，两栏宽度比从 1 : 1.6 改为 1 : 1.75。报告仍为 8 页，后续排版没有被打乱。
6. 图注补充了以下说明：
   - 起点、终点的画法；
   - 箭头表示训练方向；
   - a 图只画 b 范围之外的设置。

## 验证

- `SKIP_EXPORT=1 scripts/build_report.sh` 与 `SKIP_EXPORT=1 scripts/build_typst_figures.sh ablation-tradeoff` 均成功。
- 逐一查看了第 6–8 页的页面图（`build/report-pages/`）：
  - 图 4 在第 7 页顶部，文字无重叠；
  - 第 8 页排版正常。
- 数据未改：图中所有数值仍读自 `figures/data/ablation.json`。

## 说明 / 待定

- 图注中的“16k / 205k steps”是写死的文字（与原图注里写死的“10 seeds”一样）。若训练步数变了，需要同步修改。
- 工作区里 `report.json`、`00-abstract.typ` 等文件的改动来自其他任务，本任务没有动这些文件，也没有提交 git。

## 提交（2026-11-25 11:44，用户要求“提交”）

- 只提交本任务的改动：`ablation-tradeoff.typ`、`05-results.typ` 中图 4 图注那一处、本任务目录。
- 同一文件中其他任务的改动（表格、5.2/5.3 节、诊断段落等）以及 `report.pdf`、`report.json`、摘要等未提交，仍留在工作区。`report.pdf` 是构建产物，里面混有其他任务未提交的内容，等那些任务提交后再重新构建、一并提交。
