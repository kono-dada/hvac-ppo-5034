# 任务报告 task-20261125-02：报告图表美化

完成时间：2026-11-25（Asia/Shanghai）

## 实际交付

### 1. 共享绘图风格模块

新增 [hvac/plotting.py](../../../hvac/plotting.py)，三个出图脚本都改用它：

- `use_style()`：统一字体（PingFang SC，公式用 stixsans）、去掉上框和右框、浅色网格、标题左对齐、导出 200 dpi、自动裁边。
- 统一配色：温控器橙色、理论最优和基线黑色（正文写的“橙色”“黑线”仍然对得上），制热红、制冷蓝、舒适区绿。
- `ablation_colors()`：消融图里小于基线的取值用蓝色系，大于基线的用橙红色系，基线是黑色，离基线越远颜色越深。参数往哪个方向变，从颜色就能看出来。
- `panel_title()` / `header()` / `save()`：分图标题统一写成“(a) …”，图的总标题和灰色说明行放在左上角。

### 2. 重绘的 9 张图（文件名不变，正文链接不用改）

| 图 | 主要改动 |
| --- | --- |
| [env_schematic.png](../../../docs/report/figures/env_schematic.png) | 重画示意图：圆角节点并标出“热容小·变化快”/“热容大·变化慢”；传热系数改成放在箭头上的标签；太阳画成图标；热泵改为制热（红）和制冷（蓝）两个箭头 |
| [env_step_response.png](../../../docs/report/figures/env_step_response.png) | 去掉图例，曲线旁直接写名字；正文引用的 22.4 / 17.7 / 16.3 °C 三个点标在图上 |
| [env_days.png](../../../docs/report/figures/env_days.png) | 颜色按当日室外均温取，加了色标；舒适区和“有人”时段直接写在图上 |
| [env_controllers.png](../../../docs/report/figures/env_controllers.png) | 动作子图的背景分为制热（红）和制冷（蓝）两区；理论最优画粗线、温控器画细线；8 点和 18 点加竖线 |
| [method_loop.png](../../../docs/report/figures/method_loop.png) | 改成卡片样式（带彩色标题栏和编号），消融的三个参数用红色标签标出；文字内容和原图完全相同 |
| [ablation_final.png](../../../docs/report/figures/ablation_final.png) | 配置按方向着色，单个种子画成淡色点，均值和 95% 区间画成粗点和粗线；预测区间改用淡紫色，不和橙色温控器混淆；基线一列加灰底，刻度加粗；图例统一放在顶部 |
| [ablation_learning_curves.png](../../../docs/report/figures/ablation_learning_curves.png) | 纵轴收紧（不舒适度 0–0.4，原为 0–1），训练后半段的差异能看清；每一行的图例移到右侧，不再遮挡曲线；各行左侧写明所改参数 |
| [ablation_tradeoff.png](../../../docs/report/figures/ablation_tradeoff.png) | 拆成“全貌”和“放大”两幅；放大图里每个点旁直接写配置名，置信区间改为淡色；等成本线的标签不再出界 |
| [ablation_update_size.png](../../../docs/report/figures/ablation_update_size.png) | epochs = 1、3 的 approx_kl 改用虚线并在图例里注明“（偏小）”，和 3.1 节的说明对应 |

## 验证

- **数字没变**：改动前先保存了三个脚本的标准输出，改完重跑后逐字 diff：`make_environment_figures.py`、`make_method_figures.py`、`analyze_ablation.py` 的输出完全一致（只少了一条 matplotlib 的字体警告）。`docs/report/data/` 下的表格没有任何变化。
- **只改了呈现方式**：脚本里只替换了绘图函数，统计和打印数字的代码一行没动。
- `pytest`：21 项全部通过。
- 9 张图逐张目视检查过，没有文字重叠或出界。

## 待定事项

1. **正文语言**：图上的文字目前是中文，和中文稿一致。如果最终报告用英文写，需要把标签换成英文，配色和版式可以直接复用。
2. **正文措辞**：01 和 03 节提到的“橙色”“黑色”“黑线”与新图一致，不用改。不过 3.6 节写的“下图的黑线”现在出现在两幅子图里，可以考虑改成“(b) 图的黑线”（这一节不属于本任务，没有改）。
3. **[ablation_update_size.png](../../../docs/report/figures/ablation_update_size.png)** 仍然没有被正文引用。
4. 本任务没有提交 commit。另外，工作区里出现了不属于本任务的未跟踪文件（`hvac/diagnostics.py`、`tasks/task-20261125-03/`），本任务没有碰它们。提交时请只加入本任务的文件。

---

# 第 2 轮：用 Typst 重做全部图（英文）

完成时间：2026-11-25（Asia/Shanghai）

## 实际交付

### 1. 仓库内的 Typst 工具链（不往系统里装）

- [scripts/setup_typst.sh](../../../scripts/setup_typst.sh)：把 Typst 0.15.1 二进制下载到 `tools/typst/`，把包缓存（lilaq 0.6.0、CeTZ 0.5.3、fletcher 0.6.0 及依赖）放到 `tools/typst/packages/`，并把离线文档下载到 `tools/docs/`：Typst 语言文档、lilaq 文档网站源码、CeTZ 手册 PDF、fletcher 文档。脚本可重复运行。
- 字体：[tools/fonts/](../../../tools/fonts/) 放了 Fira Sans 和 Fira Math（SIL OFL 许可，许可文件同目录）。这些字体纳入 git，所以图在任何机器上都长得一样；编译时用 `--ignore-system-fonts` 确保不会悄悄换成系统字体。
- 二进制、包缓存、文档共约 50 MB，可以重新下载，已写进 [.gitignore](../../../.gitignore)；输出目录 `docs/report/typst/build/` 也一样。说明见 [tools/README.md](../../../tools/README.md)。

### 2. 数据与绘图分离

- [scripts/export_figure_data.py](../../../scripts/export_figure_data.py)：把每张图要用的数据导出成 JSON，放在 `docs/report/typst/figures/data/`。计算直接调用现有脚本里的函数（`analyze_ablation.load/_curve_stats/references`，环境模拟与 `fig_*` 用同样的种子和步骤），所以 Typst 图和 matplotlib 图的数据是同一份。Typst 里不手抄任何数字，图上的 22.4 / 17.7 / 16.3 °C、30 / 400 / 50 W/K、245 轮等都从 JSON 读取。
- [scripts/build_typst_figures.sh](../../../scripts/build_typst_figures.sh)：先导出数据，再把每张图编译成 PDF、SVG、PNG（250 ppi）。可以只传部分图名；`SKIP_EXPORT=1` 时跳过数据导出。

### 3. 9 张图的 Typst 版本（`docs/report/typst/figures/`）

| Typst 文件 | 对应原图 | 画法 |
| --- | --- | --- |
| [style.typ](../../../docs/report/typst/figures/style.typ) | — | 共享主题：字体、配色（沿用第 1 轮的约定：温控器橙色，理论最优和基线黑色，制热红，制冷蓝，舒适区绿；消融参数小于基线用蓝色系，大于基线用橙红色系）、坐标轴、图例样式、小工具函数 |
| [env-schematic.typ](../../../docs/report/typst/figures/env-schematic.typ) | env_schematic | CeTZ |
| [env-step-response.typ](../../../docs/report/typst/figures/env-step-response.typ) | env_step_response | lilaq |
| [env-days.typ](../../../docs/report/typst/figures/env-days.typ) | env_days | lilaq，手画色标 |
| [env-controllers.typ](../../../docs/report/typst/figures/env-controllers.typ) | env_controllers | lilaq |
| [method-loop.typ](../../../docs/report/typst/figures/method-loop.typ) | method_loop | CeTZ，文字是原图的英文翻译 |
| [ablation-learning-curves.typ](../../../docs/report/typst/figures/ablation-learning-curves.typ) | ablation_learning_curves | lilaq |
| [ablation-final.typ](../../../docs/report/typst/figures/ablation-final.typ) | ablation_final | lilaq |
| [ablation-tradeoff.typ](../../../docs/report/typst/figures/ablation-tradeoff.typ) | ablation_tradeoff | lilaq，拥挤处用引线标注 |
| [ablation-update-size.typ](../../../docs/report/typst/figures/ablation-update-size.typ) | ablation_update_size | lilaq，对数纵轴 |

图内文字全部是英文。每个文件导出 `render(width: …, title: true)` 函数，有两种用法：

- 单独编译：用 [standalone.typ](../../../docs/report/typst/figures/standalone.typ) 编译，带图标题和副标题，输出到 `docs/report/typst/build/<名字>.{pdf,svg,png}`。
- 在 Typst 报告里直接嵌入（推荐）：`#import "figures/ablation-final.typ" as final`，再写 `#figure(final.render(title: false), caption: […])`。这样图是矢量，字体和正文一致，标题由报告的 caption 提供。[figures-preview.typ](../../../docs/report/typst/figures-preview.typ) 就是这种用法的样例，编译出一份 A4 预览 `build/figures-preview.pdf`。

## 验证

- 关键数字与第 1 轮保存的脚本输出（`/tmp/figbase/*.txt`）对照，一致：阶跃响应 22.4 / 16.3 / 17.7 °C；冷天和热天的选取（均温 0.9 / 30.1 °C）及三种控制方式的耗电和不舒适度；参考控制器成本 4.48 / 5.37 / 3.49；12 个配置训练结束时的成本均值与 `ablation_summary` 一致。
- 9 张图全部编译通过，PNG 宽度都是 16 cm（A4 加 2.5 cm 页边距时的正文宽度）。我逐张看过，没有文字重叠或出界；A4 预览里嵌入后的效果也检查过。
- 第 1 轮的 matplotlib PNG、三个出图脚本、`hvac/plotting.py`、`docs/report/data/` 这一轮都没有改动（PNG 的修改时间仍是第 1 轮的）。
- `pytest`：26 项全部通过（比第 1 轮多出的 5 项来自 task-03 新增的测试）。

## 待定事项

1. 中文稿（`docs/report/zh/`）仍然引用 matplotlib 的中文 PNG，没有改。等英文 Typst 报告开写时，直接导入 `.typ` 图即可。
2. 图标题和副标题是我拟的英文，正式报告里建议用 `title: false`，把说明写进 caption。
3. method-loop 的文字是 02_method.md 原图的直译，内容没有增减；它属于同事负责的章节，请对方确认译法。
4. 3.6 节“下图的黑线”的措辞问题，以及 ablation_update_size 仍未被引用，和第 1 轮一样，仍待处理。
5. 没有提交 commit。提交时请只加入本任务的文件：`tools/README.md`、`tools/fonts/`、`.gitignore`、`scripts/{setup_typst.sh,build_typst_figures.sh,export_figure_data.py}`、`docs/report/typst/`（`build/` 已忽略），以及第 1 轮的文件。
