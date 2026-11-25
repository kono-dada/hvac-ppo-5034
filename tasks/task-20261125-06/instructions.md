# 任务记录 task-20261125-06：统一图中字体，图 1 改为单栏

记录时间：2026-11-25 06:05（Asia/Shanghai）

## 用户要求

用户原话：“看看现在的 report。figure里的字体也感觉很难看，帮我统一一下；figure1 可以放单栏里，这只是一个环境的图片。”

## 现状（开始前）

- 正文版式（`template.typ`，工作区未提交的改动）已改成 LaTeX article 风格：标题、正文、图注、表格全部用 New Computer Modern。
- 图（`docs/report/typst/figures/style.typ`）仍用 Fira Sans / Fira Math，且报告里以 `shrink: 0.8` 缩小放置，图中字号实际约 6.4 pt，与 9 pt 图注、10 pt 正文、8 pt 表格都对不上。
- 图 1（`env-schematic`，房间示意图）放在通栏，约占 0.76 栏宽。

## 要做的

1. 图的字体与正文统一：New Computer Modern（数学用 New Computer Modern Math），图中各处文字的字号在页面上统一（约 8 pt，与表格一致），不再靠缩放得到。
2. 图 1 改为单栏图，按单栏宽度（约 8.25 cm）重新排布示意图，保证文字可读、不重叠。
3. 重新构建报告和独立图，逐页检查。

## 约束

- 不改数据和结论，不改正文内容（图的摆放代码除外）。
- 工作区里有其他未提交的改动（`template.typ`、`report.typ`、`lib.typ`、`05-results.typ`、task-03 的诊断文件等），不回退、不提交。
- 不提交 git，除非用户要求。

## 交付物

- 修改后的 `figures/style.typ`、`figures/env-schematic.typ` 及需要调整的图模块、报告中图的摆放。
- 重新生成的 `docs/report/report.pdf`。
- 本目录 `report/report.md`。
