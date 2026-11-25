# 任务记录 task-20261125-08：重排报告图 4（能耗–不适度权衡图）

记录时间：2026-11-25 11:30（Asia/Shanghai）

## 用户要求

用户原话：“这个报告的 figure4，感觉很多文字都挤在一起了，不好看，帮我做的更正常一点。另外 baseline during training 也可以强调一下起点位置，以及用箭头表示训练的方向”

## 范围

- 图 4 即 `docs/report/typst/figures/ablation-tradeoff.typ`（报告中 `<fig-tradeoff>`，位于 `sections/05-results.typ`）。
- 要做的：
  1. 减少图中文字重叠、拥挤：标签、等成本线标注、图例重新排布。
  2. baseline 训练路径：标出起点，用箭头表示训练方向。
- 交付：修改后的图模块，重新构建的 `docs/report/report.pdf` 与独立图。

## 约束

- 不改数据和结论。正文只在图注需要配合时调整。
- 工作区有其他任务未提交的改动，不回退、不提交 git。
