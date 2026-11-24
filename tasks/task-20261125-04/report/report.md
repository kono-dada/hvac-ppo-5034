# 任务报告 task-20261125-04：正式报告（英文，Typst）

记录时间：2026-11-25 05:35（Asia/Shanghai）

任务要求见 [instructions.md](../instructions.md)。

## 结论

英文报告的内容已全部写完，能用一条命令从数据生成 PDF：[report.pdf](../../../docs/report/report.pdf)。
排版没有收尾：按用户指示停手时，PDF 是 7 页，超出约定的 6 页（原因见“待解决”第 1 条）。
本任务新建的 17 个文件已按用户要求单独提交（提交说明以“正式报告（英文，Typst）”开头）；这次提交单独检出还不能编译，原因见“待解决”第 8 条。

## 实际交付

### 报告源文件（[docs/report/typst/](../../../docs/report/typst/)）

| 文件 | 内容 |
| --- | --- |
| [report.typ](../../../docs/report/typst/report.typ) | 入口：封面信息（作者、课程代码、日期）和章节顺序 |
| [template.typ](../../../docs/report/typst/template.typ) | 版式：A4 双栏，正文 Libertinus Serif，标题、图注和表格用 Fira Sans（与图一致） |
| [lib.typ](../../../docs/report/typst/lib.typ) | 读入数据、数字格式、判定标记、图表摆放的工具函数 |
| [00-abstract.typ](../../../docs/report/typst/sections/00-abstract.typ) | 摘要 |
| [01-introduction.typ](../../../docs/report/typst/sections/01-introduction.typ) | 引言：问题、HVAC 强化学习文献、PPO 超参数文献 |
| [02-environment.typ](../../../docs/report/typst/sections/02-environment.typ) | 环境：房间模型、成本、对照控制器、对 PPO 的挑战 |
| [03-method.typ](../../../docs/report/typst/sections/03-method.typ) | 方法：MDP、PPO 与三个参数、实验设计和判定规则 |
| [04-predictions.typ](../../../docs/report/typst/sections/04-predictions.typ) | 预测：基线的三点观察和由此得出的预测 |
| [05-results.typ](../../../docs/report/typst/sections/05-results.typ) | 结果：总览、三个参数逐个分析、训练稳定性 |
| [06-conclusion.typ](../../../docs/report/typst/sections/06-conclusion.typ) | 讨论与结论：三个指标的小结、预测为什么错、建议、局限 |
| [refs.bib](../../../docs/report/typst/refs.bib) | 17 条参考文献 |

报告用了 4 张图（房间示意图、冷热两天的三种控制器、学习曲线、耗电–不舒适度平面）和 3 张表（对照控制器、预测与结果对照、lambda 分时段）。
图直接导入 task-20261125-02 的图模块（`figures/*.typ`），没有改动那些文件。

### 脚本

- [export_report_data.py](../../../scripts/export_report_data.py)：把报告里所有表格和大部分正文数字导出到 `docs/report/typst/data/report.json`。只读已有结果、重放已保存的策略，不训练，不改任何结果文件。
- [build_report.sh](../../../scripts/build_report.sh)：先导出数据，再编译 PDF，并把每页渲染成 PNG 放进 `docs/report/typst/build/report-pages/`（该目录已被 `.gitignore` 忽略）。

```bash
scripts/build_report.sh                 # 实测 11 秒
SKIP_EXPORT=1 scripts/build_report.sh   # 只重新编译
```

## 和中文稿相比，有意做的取舍

报告是三份中文稿的精简版，不是全文翻译（6 页的约定是用户定的）。

- **没有进报告的内容**：54 条预测的逐条判定表和判定规则的完整表格（正文给汇总，并指向 `docs/report/data/ablation_verdicts.md`）；单温度模型的对比表；基线训练记录表；观测向量的逐项表格；`approx_kl` 与 clip range 的对照表（改成两句话）。
- **没有用的图**：`env-step-response`、`env-days`、`method-loop`、`ablation-final`、`ablation-update-size`。其中 `method-loop` 写的仍是预实验的 245 轮、50 万步，与消融实验的 100 轮不一致，本来也不能直接用。
- **用词**：中文稿的“基线”译为 baseline，“回退”译为 setback，“理论最优”译为 oracle。
- **训练设置**：全文按消融实验的实际设置写（204,800 步、100 轮、每 8,192 步在 100 个测试日上评估），没有沿用 `02_method.md` 里预实验的 50 万步。此前记录的“第 2 节和消融设置不一致”在英文报告里不存在。
- **两处说法比中文稿 3.6.4 节收得更紧**，因为 task-20261125-03 的事后诊断（中文稿 3.7 节）给出了相反的实测：
  1. 中文稿写“每轮更新大致走到裁剪边界附近才停下来”（原文已标注未验证）。诊断显示样本没有堆在边界附近。英文报告只写消融实验本身能支持的事实：每轮改动的幅度随 clip range 成比例变化；`clip_fraction` 数的是越界样本有多少，不是它们有多重要。
  2. 中文稿写 epochs = 1 和 3 的 `approx_kl` 偏小。诊断显示 3 遍时记录值与真实值之比是 1.03。英文报告改为“与其他行不可比”，不说偏小。
- **新写的部分**：摘要、引言里的文献综述、讨论与结论。

## 验证

- **构建**：`scripts/build_report.sh` 完整运行通过，没有警告输出。05:23 那次构建是 6 页；05:28 和 05:33 两次都是 7 页，差别来自 `figures/env-controllers.typ` 在 05:26 被改版。
- **表格数字不手抄**：三张表全部由 `report.json` 生成。导出脚本内置了三项核对，不通过就报错退出：
  - 12 种配置的成本、耗电、不舒适度、违规比例、回退，由各种子的运行目录重新计算，与 `ablation_summary.csv` 相差小于 1e-9；
  - lambda 分时段表重放了 50 个已保存的策略，重放得到的日均耗电与各自 `final_eval.json` 里的值相差小于 1e-6；
  - 54 条预测的判定计数为命中 31、部分命中 3、不命中 5、无法判断 15，与中文稿一致。
- **正文里手写的数字**：用一段只读脚本独立重算了一遍，全部吻合：满功率一步升温 2.24 °C；时间常数 0.59 和 38.8 小时；冷天（25 天）理论最优平均 5:38、最早 3:15 开机，最早一天到 8 点正好 19 步；理论最优 80% 的上班时间贴在舒适区边缘 0.05 °C 以内；示例冷天里 8 点开机的温控器 10:30 进入舒适区，理论最优 5:15 开机；基线种子 0–4 在 24,576 步时耗电 7.76、不舒适度 0.00，前半程 `clip_fraction` 6.0%；3 遍的终点对应基线约 13,400 次梯度更新。
  这次核对改正了两处初稿的错误：“理论最优 5:15 起满功率”（实际 5:15 开机时是 83%，5:30 才满功率）和 GAE 权重减半步数的表述（加了“约”）。
- **文献**：
  - 作者、题目、刊物、卷期页码：6 篇对照 arXiv 页面，10 篇对照 Crossref 记录，SB3 一篇对照 ACM 页面上显示的论文首页。
  - 报告引用的具体结论，对照原文核对过的有：PPO 论文的四个得分（0.76、0.82、0.70、−0.39）和“7 个机器人任务”；GAE 论文的 0.9 到 0.99；Andrychowicz 等的 25 万个智能体、多遍、lambda = 0.9、0.25；Engstrom 等关于裁剪与信赖域的说法；Colas 等关于 Welch 检验和 bootstrap 的说法；Manjavacas 等的那句引文；Arroyo 等的 21–24 °C；Wei 等、Drgoňa 等、Biemann 等、Blum 等各自摘要里的说法。
  - 没有在本次核对原文的：见“待解决”第 6 条。
- **测试**：`pytest tests/test_env.py tests/test_stats.py`，21 项通过。`tests/test_diagnostics.py` 属于 task-20261125-03，没有运行。

## 待解决

1. **排版没有收尾，PDF 现在是 7 页。** 05:23 时已经排成正好 6 页；随后 `env-controllers` 图改版（加了“当天得分”小表，变高了），最后两条参考文献被挤到第 7 页。用户指示先不管排版，所以没有再调。图定稿以后需要重新排一次。
2. **图还在变，报告会跟着变。** 报告实时导入图模块，对方保存到一半时构建会失败（05:26 遇到过一次，脚本以非零状态退出，旧 PDF 没有被覆盖），重跑即可。
3. **封面信息是占位符。** 作者和学号待填（`report.typ` 顶部，改一处）。课程代码按 `AGENTS.md` 写的 AIAA 5024，但仓库目录名是 hvac-5034，需要用户确认。
4. **事后诊断（中文稿 3.7 节）没有写进报告**，这是按用户“先忽略”的指示。结论一节的局限里有一句“我们没有考察为什么这么少的被裁剪样本有这么大的影响”，诊断并入以后这句话就不成立了，要一起改。
5. **调参温控器的违规比例**，报告表 1 是 3.7%，中文稿 1.5 节是 3.6%。真值恰好是 3.65%（4000 个上班步里 146 个），两边四舍五入的方式不同，数据没有问题。
6. **两处文献细节凭常识填写，没有在本次查到出处页面**：GAE 论文发表在 ICLR 2016；SB3 论文在 JMLR 第 22 卷的编号是 268。另外 Bacher 和 Madsen“单个温度不足以描述建筑动态”的说法沿用中文稿 1.3 节，本次没有重读原文。
7. **引言没有用 IEA 的全球数字。** IEA 页面打不开（HTTP 403），只在一份转引它的 UNFCCC 文件里看到数字，所以没有引，改用 Wei 等摘要里的美国数字。
8. **已提交，但这次提交单独检出不能编译。** 提交的是本任务新建的 17 个文件：`docs/report/typst/` 下的 `report.typ`、`template.typ`、`lib.typ`、`refs.bib`、`sections/`（7 个）、`data/report.json`，`docs/report/report.pdf`，两个脚本，以及本任务目录的两份记录。没有修改任何已跟踪的文件，也没有提交其他任务的文件。
   报告源文件依赖 task-20261125-02 的三样东西，它们在本次提交时都还没有进 git：图模块 `docs/report/typst/figures/`（含 `figures/data/*.json`）、字体 `tools/fonts/`、安装 Typst 的 `scripts/setup_typst.sh`。等那个任务提交以后，仓库才能从干净的检出编译出报告。
9. **提交的 PDF 对应的是 05:33 时的图。** PDF 由提交的这份源文件生成（源文件没有比 PDF 新的）；但 PDF 生成之后，`figures/ablation-learning-curves.typ` 又被改过，所以 PDF 里的图 3 不是最新版。重新运行 `scripts/build_report.sh` 即可更新。
