# AIAA5024 期末项目

以下是本项目的作业原始要求。制定方案、修改代码或撰写报告时，以此核对项目范围。

## 作业题目与要求

**Building HVAC Control with PPO: Effect of Clip Range, GAE Lambda, and Epochs**

A controller sets heating/cooling to balance comfort and energy use in a simple thermal model of a room.

Train PPO and run an ablation over clip range, GAE lambda, and number of epochs. Report comfort violations, energy use, and training stability.

Custom thermal model (Gymnasium wrapper), Stable-Baselines3

## 3. Grading Criteria

The final project grade has two parts. Detailed standards are on Canvas.

- Project Report (35%)
  - Problem Definition & Literature Review (20%)
  - Methodology (30%)
  - Results & Analysis (30%)
  - Clarity & Presentation (20%)
- Project Presentation (25%)
  - Clarity & Engagement (30%)
  - Content & Technical Depth (50%)
  - Q&A Handling (20%)

## 目前想象中的最终报告蓝图

1. 通俗地讲清楚环境是怎么定义的、为什么这么定义、对 PPO 来说主要挑战是什么。
2. 讲清楚 MDP 设置、所用算法、整个训练流程。
3. 根据学界对强化学习超参数与结果的普遍认识，先对实验结果做预测，再实际做实验，对比结果与预测的区别。
4. （进阶）如果有区别，分析原因。

## 任务记录

- 每次用户交付一个新任务，在仓库根目录的 `tasks/` 下按用户所在时区的日期创建 `task-YYYYMMDD/`。同一天有多个任务时，依次用 `task-YYYYMMDD-02/`、`task-YYYYMMDD-03/` 等目录。
- 每个任务目录创建 `instructions.md` 和 `report/report.md`。开始前把用户的任务要求、约束和交付物记入 `instructions.md`。用户补充或修改同一任务时，更新该任务的记录。
- 按记录完成任务。代码、测试和配置可以在主仓库的对应目录中修改，优先复用已有模块、配置和入口。任务记录里可链接到相关文件。
- 完成后在 `report/report.md` 写明实际交付、验证结果和仍待解决的事项（如有）。
