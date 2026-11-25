# 任务报告 task-20261125-07：正式报告文本审阅

完成时间：2026-11-25（Asia/Shanghai）。只审阅，没有改报告源文件。任务要求见 [instructions.md](../instructions.md)。

审阅前提：读者只拿到 PDF 和代码包，看不到 git 历史、`docs/`、中文稿或 `tasks/`。

## A. 报告提到了读者看不到的东西（用户指出的问题及同类问题）

1. **commit 哈希**：[04-predictions.typ:5](../../../docs/report/typst/sections/04-predictions.typ)。读者无法核对。建议改成“预测和判定规则是在训练基线以外的任何配置之前写下并固定的”，并把预注册内容放进附录或代码包。
2. **逐条判定表的仓库路径**：[05-results.typ:137](../../../docs/report/typst/sections/05-results.typ) 写着“The full list is in the repository (`docs/report/data/ablation_verdicts.md`)”。这个文件是**中文的**，而且 `docs/` 不一定会提交。建议：在英文附录里放 54 条预测的判定表（附录不受 6 页限制的话），或者由代码脚本生成英文 CSV 并随代码包提交，正文写“see the supplementary code”。
3. **“pre-experiment(s)”出现三次却没有介绍**：[03-method.typ:23](../../../docs/report/typst/sections/03-method.typ)（4.54 → 5.19）和 [03-method.typ:61](../../../docs/report/typst/sections/03-method.typ)（500k 步 4.54，1.5M 步 4.23）。至少用一句话说明：预实验是什么配置、几个种子。
4. **没有说明代码怎么用**：报告没有提代码包，也没有写软件版本（SB3 2.9.2、Gymnasium 1.4.0）。建议在方法部分末尾加一句“Code and reproduction”。

## B. 和报告自身的规则或数据对不上（硬伤）

5. **λ 的预测数写错**：[05-results.typ:200](../../../docs/report/typst/sections/05-results.typ) 写“All five predictions about the cost held”，但 λ 只有 4 个非基线取值（0.8、0.9、0.99、1），判定表里也只有 4 条成本预测。应为 **four**。
6. **“部分命中”没有定义**：[03-method.typ:75-76](../../../docs/report/typst/sections/03-method.typ) 只定义了 held 和 failed，接着写“Everything else is undecided”。但表 2 里有 3 条 ◐ partly held。中文稿的规则是：均值在区间外，但 95% 区间与预测区间重叠，判为部分命中。英文稿漏掉了这一条，现在的文字等于说这 3 条应判为 undecided。
7. **λ 的耗电和不舒适度预测被说成全部失败**：[05-results.typ:201](../../../docs/report/typst/sections/05-results.typ) 写“The prediction that energy and discomfort would also stay the same did not [hold]”。判定表的实际结果是：0.8 的不舒适度（< 0.3）命中，0.9 的不舒适度不命中，两条耗电预测都无法判断。建议改为“only one of them failed outright, but the direction is consistent: …”。
8. **“equivalent”和自己的等价判定冲突**：[05-results.typ:190](../../../docs/report/typst/sections/05-results.typ) 写“0.1 to 0.3 are equivalent here”。按 3.3 节，“same”需要 90% 区间落在 ±0.1 内才能成立；0.3 成本差的 95% 区间是 [−0.07, 0.13]，0.1 是 [−0.07, 0.15]，都没有被判为相同。上一段的 “cannot be told apart” 才是准确的说法。
9. **事后诊断与正文冲突**（关系到提交哪些代码）：
   - [05-results.typ:181-182](../../../docs/report/typst/sections/05-results.typ) 写“The log of a single setting cannot show how strongly those few samples hold an update back. Only varying ε can.”
   - [06-conclusion.typ:29](../../../docs/report/typst/sections/06-conclusion.typ) 的局限部分写“we did not examine why so few clipped samples carry so much weight”。
   - 但 task-20261125-03 已经做了这件事：在单个设置上分解出了被裁剪样本的梯度（基线约 7% 的样本贡献了 59% 的梯度长度，使梯度转向约 40°）。
   - 如果 `hvac/diagnostics.py`、`gradient_probe.py` 会随代码提交，报告与代码就互相矛盾。二选一：在报告里加一小段标明“post-hoc”的诊断，并删掉上面两句；或者不提交诊断代码。
   - 同一小节 [05-results.typ:188](../../../docs/report/typst/sections/05-results.typ) 把 ε = 0.5 失稳的原因写成事实（“one large update is enough to push the room outside”）。这条解释没有验证，而且与诊断结论（回拉减弱、均值漂移累积）不完全一致。

## C. 提交要件和格式

10. **篇幅**：当前 PDF 有 8 页（7 页正文加 1 页参考文献），约定是 4–6 页。
11. **封面信息未填**：作者和学号还是占位符。课程代码写的是 AIAA 5024，仓库目录名是 hvac-5034，需要确认哪个正确。
12. **代码包没有 README**：仓库根目录没有 README。代码注释还在引用读者看不到的文件，例如 [hvac/stats.py:4](../../../hvac/stats.py)、[scripts/run_ablation.py:6](../../../scripts/run_ablation.py)、[scripts/run_clip_diagnostics.py:3](../../../scripts/run_clip_diagnostics.py)（`docs/report/zh/03_ablation.md`、`task-20261125-03`）。打包前需要清理，并决定提交范围（预计是 `hvac/`、`scripts/`、`tests/`、`pyproject.toml`，加 README；不包括 `tasks/`、`AGENTS.md`、中文稿和 `runs/` 里的大文件）。

## D. 小问题

13. 违规率的定义前后不一致：表 1 的说法是“more than 0.1 K outside the comfort band”，表 2 只说“outside the comfort band”。
14. 单位写法不一致：正文和表格写 K·h，图 2、3、4 写 “Kh”。
15. 表 1 的 “Thermostat, on at 3:00” 正文没有提到。表 3 只写 “Thermostat”，应写明是 tuned thermostat。
16. [03-method.typ:41](../../../docs/report/typst/sections/03-method.typ)：λ = 1 时，估计值是**折扣**回报减去 V(s_t)，而且每 256 步的采样段末尾会自举。说成 “the observed return” 不够准确。
17. 参考文献 [10]：Andrychowicz et al. 已发表于 ICLR 2021（bib 键名里也写的是 2021），目前引用的是 arXiv 2020 版本。

## 核对过、没有问题的地方（抽查）

54 条预测的总数和各类计数；1 个 epoch 多耗电 75%；approx_kl 相差 50 倍；sqrt(2 KL)/ε 在 0.6–1.1 之间；KL 上限 0.19；GAE 半衰期 3、6、11、35 步；γ^19 = 0.83；9,600 和 13,400 次梯度步；σ 下限 0.38；预热量约为 oracle 的 3 倍；SB3 默认超参数（网络结构、学习率、max_grad_norm、按小批归一化优势、approx_kl 只统计最后一遍）；PPO 论文里的 0.76、0.82、0.70、−0.39；Andrychowicz 等人的三条建议。

## 待解决

以上问题都没有修改，等用户决定。需要先定的是第 9 条（报告是否写入事后诊断）和第 2 条（判定表放附录还是代码包），因为这两条决定了篇幅和提交范围。

## 第二轮：按用户答复修改报告（2026-11-25 11:50）

用户答复：事后诊断可以写入；写作时只看定性上是否一致，只展开讲反直觉的结果。

### 改动

| 位置 | 改动 | 对应清单 |
| --- | --- | --- |
| [00-abstract.typ](../../../docs/report/typst/sections/00-abstract.typ) | 删去 31/3/5/15 的计数，改为“多数预测方向正确，两条在定性上错了”；加一句诊断结论 | 1、2 |
| [02-environment.typ](../../../docs/report/typst/sections/02-environment.typ) | 补一句说明 3:00 开机的温控器 | 15 |
| [03-method.typ](../../../docs/report/typst/sections/03-method.typ) | 说明预实验的设置（默认 PPO、5 个种子、50 万步）；修正 λ = 1 的说法；补上“部分命中”的定义；说明正文只深入讨论定性上不一致的结果；新增 Code 段（代码随报告提交，注明 SB3 和 Gymnasium 版本） | 3、4、6、16 |
| [04-predictions.typ](../../../docs/report/typst/sections/04-predictions.typ) | 删去 commit 哈希，改为“在训练基线以外的配置之前写下，之后没有改动” | 1 |
| [05-results.typ](../../../docs/report/typst/sections/05-results.typ) | 见下 | 2、5、7、8、9、13 |
| [06-conclusion.typ](../../../docs/report/typst/sections/06-conclusion.typ) | “预测为什么错”改写为两条定性错误（clip、λ）；局限部分改为“诊断是事后设计的，只用了 3 个种子” | 9 |
| [refs.bib](../../../docs/report/typst/refs.bib) | Andrychowicz et al. 改为 ICLR 2021 的正式标题与出处 | 17 |
| [export_report_data.py](../../../scripts/export_report_data.py) | 新增 `diagnostics()`，从 `docs/report/data/clip_diagnostics_summary.csv` 和 `clip_gradient_probe.csv` 导出诊断数字；正文不手抄 | 9 |

[05-results.typ](../../../docs/report/typst/sections/05-results.typ) 的改动：

- 表 2：删去统计全部判定的最后一列，判定符号保留；图注说明判定依据 3.3 节的规则；违规率定义改为“超出 0.1 K 以上”。
- 5.1 节：删去计数和仓库路径，改为定性小结。点名两条定性上错了的预测（clip 放宽不影响结果；λ 在 0.8–0.95 之间策略不变）以及 1 遍的回退。
- 5.2 节：方向一致的结果一句带过，删去关于 O2 的细节。
- 5.3 节：删去“只看单一设置的日志无法回答”和“策略贴近边界，一次大更新就越界”两处说法；新增标明 post-hoc 的诊断段落，包括：
  - 梯度被删的份额和转向角；
  - clip = 0.5 释放的梯度抵消回拉，回拉强度下降 55%；
  - 同一 Adam 状态下单步后漂移的变化；
  - KL 是基线的 12 倍，其中 99.7% 来自均值项；
  - 舒适度占成本增量的 62%；
  - 范数裁剪在每种 clip 下都几乎始终触发。

  原来的“0.1 to 0.3 are equivalent”改为“see no difference in cost”。
- 5.4 节：five 改为 four；把“耗电和不舒适度都没保持不变”改为实际结果：不舒适度显著上升，耗电略低（不显著）。
- 5.5 节：说明 1 遍的回退违背了“训练平稳”的预测。

### 验证

- 重新运行 `scripts/build_report.sh`（含数据导出），编译无警告。`report.json` 里原有字段与 HEAD 逐项相同，只新增 `diagnostics`。
- 诊断数字与中文稿 3.7 节、`clip_gradient_probe.md` 一致：7%、59%、40°；79%、49°；47%、35°；13%、64.6%；0.529 → 0.221；回拉强度降 54.6%；KL 12 倍；均值项 99.7%；每轮均值改动 2.4 倍；舒适度占成本增量 62%。
- 逐页查看了改动涉及的第 4–8 页，版面正常。正文已不再出现 commit、repository、`docs/` 路径或 54 条计数。

### 仍待解决

- **篇幅**：PDF 从 8 页增加到 9 页（加入诊断段落后），离 4–6 页更远。
- **代码包**：没有 README，注释里引用了中文稿和 task 编号（第 12 条），提交范围尚未确定。
- **封面**：作者和学号未填，课程代码待确认（第 11 条）。
- **单位**：图里的“Kh”未改。图 3 已显示为 K·h，未逐张核对其余各图。
- **git**：没有提交。

## 第三轮：封面与提交（2026-11-25 12:00）

- [report.typ](../../../docs/report/typst/report.typ)：课程代码改为 AIAA 5034；作者为 Fangda Chen (50040218)、Liang Zhang (50040696)，去掉占位符。[pyproject.toml](../../../pyproject.toml) 和 [AGENTS.md](../../../AGENTS.md) 的课程代码一并改正，仓库里已不再出现 5024（`runs/` 数据中的数字除外）。
- 重新构建报告，无警告；看过封面，显示正确。报告仍为 9 页。
- 已连同 task-20261125-09（第 4 节改写）一起提交到本地 `main`，未推送。
