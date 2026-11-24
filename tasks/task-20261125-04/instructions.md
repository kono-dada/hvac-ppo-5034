# 任务记录 task-20261125-04：正式报告（英文，Typst）

记录时间：2026-11-25 04:13（Asia/Shanghai）

## 用户要求

用户原话：“OK，看看作业要求、仓库、以及我们的蓝图。有另外的 agent 在准备把现有的 figure 通过 typst 重新绘制；还有一个 agent 在诊断梯度，这些你都可以先忽略。现在开始制作我们的正式 report，使用英语和typst。”

对应 `AGENTS.md` 里的作业要求和最终报告蓝图的全部四项：

1. 通俗地讲清楚环境怎么定义、为什么这么定义、对 PPO 的主要挑战是什么。
2. 讲清楚 MDP 设置、所用算法、整个训练流程。
3. 先对实验结果做预测，再做实验，对比结果与预测。
4. （进阶）有区别时分析原因。

评分标准（报告部分）：Problem Definition & Literature Review 20%、Methodology 30%、Results & Analysis 30%、Clarity & Presentation 20%。

## 范围

把三份中文稿（[01_environment.md](../../docs/report/zh/01_environment.md)、[02_method.md](../../docs/report/zh/02_method.md)、[03_ablation.md](../../docs/report/zh/03_ablation.md)）写成一份完整的英文正式报告，用 Typst 排版，输出 PDF。
中文稿里没有、但评分标准要求的部分需要新写：摘要、引言（问题定义）、文献综述、讨论与局限、结论、参考文献。

## 约束

- 语言：英文。排版：Typst，使用仓库内的工具链（`tools/typst/`、`tools/fonts/`），不往系统里装东西。
- 读者：刚学会 PPO、对本项目一无所知的同学，以及评分老师。
- 不做新实验，不改数据和结论。报告里的数字必须和中文稿、`docs/report/data/` 以及脚本输出一致；表格尽量直接读数据文件，不手抄。
- 预注册的预测和判定规则（中文稿 3.1 到 3.5 节）只翻译，不改内容。
- 同时在进行的两个任务先忽略，不碰它们的文件：
  - task-20261125-02（图表改用 Typst 重画）：`docs/report/typst/figures/`、`scripts/build_typst_figures.sh`、`scripts/export_figure_data.py`、`hvac/plotting.py` 以及三个出图脚本。报告直接导入它的图模块；还没画好的图先留占位。
  - task-20261125-03（clip range 的梯度诊断）：`hvac/diagnostics.py`、`runs/diagnostics/`、`scripts/*clip_diagnostics.py`、`docs/report/data/clip_diagnostics.md`、`diag_clip.png`，以及它将要写进 03_ablation.md 的“事后分析”小节。报告先按已提交的 3.6 节写，给事后分析留好位置。
- 文献综述里的每条引用都要核对出处，不引用没核实过的结论。

## 补充（2026-11-25 04:20，用户对两个问题的答复）

1. 篇幅与格式：“没有模板，你做一个好看的；感觉4～6页左右就够了吧，只是一个作业”。
   据此：版式自行设计；全文（含参考文献）控制在 6 页以内。三份中文稿合计远超这个篇幅，所以正式报告是它们的精简版，不是全文翻译。54 条逐条判定、判定规则的完整表格等细节不进正文，只给汇总并指向仓库里的文件。
2. 封面信息：先留占位，之后再填。作者姓名和学号用醒目的占位符，课程代码按 `AGENTS.md` 写 AIAA 5024（仓库目录名是 hvac-5034，两者不一致，待用户确认）。

## 补充（2026-11-25 04:55 之后，用户在写作过程中的指示）

3. “不过我建议你不要一个文件写完所有正文，而是分开，容易维护”。
   据此：正文按节拆成 `sections/` 下的独立文件，版式（`template.typ`）、数据与工具函数（`lib.typ`）、封面信息与章节顺序（`report.typ`）各自分开。
4. “赶快开始写吧”“继续”。
5. “内容都写好了是吧，你先别管排版了”。
   据此：内容写完后不再调整版面。停手时报告因为图表改版而超出 6 页，这一项留作待办，不在本次处理。

## 交付物

1. 报告源文件：`docs/report/typst/`（入口 `report.typ`、版式 `template.typ`、数据与工具 `lib.typ`、各节 `sections/`、参考文献 `refs.bib`）。
2. 数据导出脚本 `scripts/export_report_data.py` 和构建脚本 `scripts/build_report.sh`：一条命令生成 PDF。
3. PDF：`docs/report/report.pdf`。（最初记的路径是 `docs/report/typst/build/report.pdf`；后来另一个任务把 `build/` 加进了 `.gitignore`，所以 PDF 改放到上一级。）
4. 本目录下的 `report/report.md`：实际交付、验证结果、待解决事项。
