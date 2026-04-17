# GPU 优化迭代（Rounds）文档规范

本目录定义“GPU 优化迭代”的轮次（round）组织方式、每轮必须产出的中文文档、以及与指标框架的对齐方式。目标是让后续优化工作在“可追踪、可复现、可决策”的轨道上推进。

统一指标口径请先阅读：[metrics-framework.md](../governance/metrics-framework.md)。

上层入口：

- 仓库级入口：[MAIN.md](../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../governance/workflow-protocol.md)

## 0. 当前轮次快速入口

仓库级运行索引与当前状态以 [MAIN.md](../../../MAIN.md) 为准；本页额外提供现有轮次文档的快速跳转，避免“轮次入口”只停留在规范层。

- R0001 控制面初始化：[round-overview.md](./round-0001-init/round-overview.md) / [summary.md](./round-0001-init/summary.md) / [close.md](./round-0001-init/close.md)
- R0002 Potter 入口归一化：[round-overview.md](./round-0002-potter-entry-normalization/round-overview.md) / [summary.md](./round-0002-potter-entry-normalization/summary.md) / [close.md](./round-0002-potter-entry-normalization/close.md)
- R0003 控制面严格复查加固：[round-overview.md](./round-0003-control-plane-hardening/round-overview.md) / [summary.md](./round-0003-control-plane-hardening/summary.md) / [close.md](./round-0003-control-plane-hardening/close.md)
- R0004 活跃入口互链加固：[round-overview.md](./round-0004-entry-link-hardening/round-overview.md) / [summary.md](./round-0004-entry-link-hardening/summary.md) / [close.md](./round-0004-entry-link-hardening/close.md)
- R0005 控制面回链补强：[round-overview.md](./round-0005-control-plane-backlinks/round-overview.md) / [summary.md](./round-0005-control-plane-backlinks/summary.md) / [close.md](./round-0005-control-plane-backlinks/close.md)
- R0006 控制面总说明入口块补齐：[round-overview.md](./round-0006-control-plane-readme-backlink/round-overview.md) / [summary.md](./round-0006-control-plane-readme-backlink/summary.md) / [close.md](./round-0006-control-plane-readme-backlink/close.md)
- 新增 round 时，先更新 `MAIN.md` 的“每轮运行索引”，再把本节补齐到最新轮次。

## 1. 轮次命名与结构

- 轮次编号采用 4 位数字：`round-0001-<slug>`。
- `slug` 用英文短语表达主题，例如：`init`、`profiling`、`kernel-fusion`。
- 每轮一个目录：`docs/codex-potter/iterations/round-XXXX-<slug>/`。

每轮目录建议包含（最小集合）：

- `round-overview.md`：本轮目标、范围、输出物、退出标准（exit criteria）。
- `plan.md`：本轮计划、阅读清单、任务拆分、DoD 与验证计划。
- `code.md`：本轮编码/执行记录、实际落点、与计划偏差。
- `test.md`：本轮验证记录、命令、环境信息、失败项与回归风险。
- `next-round-suggestions.md`：对下一轮的建议列表、决断点、依赖与风险。
- `summary.md`：本轮总结、关键决策、交接清单与下一轮阅读入口。
- `commit.md`：本轮提交计划、实际提交、审阅重点与未提交状态。
- `close.md`：本轮结束记录、索引更新、下一轮起点。

可选（后续轮次若需要可增加）：

- `results/`：原始日志与汇总结果（benchmark 输出、profiling 报告等）。
- `artifacts/`：视觉对比图/视频、diff 图等。
- `decisions.md`：关键决策记录（Decision Record），尤其是“误差换速度”的取舍。

导航要求：

- 每份 round 文档标题下都应保留统一的“上层入口”区块；最低要求是回链到 `MAIN.md`、控制面总说明、工作流协议、轮次索引其中之一，推荐直接保留四入口完整版。
- `plan.md` / `code.md` / `test.md` / `commit.md` / `close.md` / `summary.md` 可直接继承模板中的“上层入口”区块；`round-overview.md` 与 `next-round-suggestions.md` 需手工补同样的回链。

## 2. 每轮必须回答的问题（Checklist）

`round-overview.md` 必须明确回答：

- 本轮要优化什么，为什么是现在做（动机/痛点/瓶颈证据）？
- 本轮的“成功定义”（成功指标与阈值）是什么？
- 本轮的范围（in-scope）与不做什么（out-of-scope）？
- 本轮产出物（deliverables）有哪些，谁负责？
- 本轮退出标准（exit criteria）：如何判断可以进入下一轮？

`code.md` 必须明确回答：

- 实际做了哪些改动，分别落在哪些文件/模块？
- 哪些地方偏离了 `plan.md`，为什么偏离？
- 哪些改动仍停留在本地状态，没有进入可提交集合？

`commit.md` 必须明确回答：

- 本轮预计如何拆分提交，实际又是如何拆分的？
- 每个提交解决了什么问题，审阅时先看哪里？
- 是否存在故意不提交的本地状态（例如 `.codexpotter`）？

`close.md` 必须明确回答：

- `MAIN.md` 与轮次索引是否已更新？
- 下一轮从哪个文件开始读、从哪个任务继续最省力？
- 本轮是否真正满足退出标准，还是只是阶段性结束？

每轮的结果汇报（若本轮有实现变更）必须对齐统一 Scorecard（见指标框架）。

## 3. 下一轮建议的写法（Template）

`next-round-suggestions.md` 中每条建议应按统一格式书写，避免“拍脑袋待办”：

```text
建议标题：<一句话描述>
动机：<当前瓶颈/痛点/风险>
假设：<为什么这会带来收益，预期影响的指标>
方案范围：<要改哪些模块/路径，是否需要实验开关>
评测计划：<按 metrics framework 的速度/误差/视觉/工程化分别写>
验收门槛：<P0 门槛，不满足则不合入>
决断点：<何时做 Go/No-Go，谁拍板，需要哪些证据>
回滚与降级：<如何安全落地>
依赖与风险：<依赖数据/工具/环境，潜在风险与缓解>
```

## 4. 决断点（Decision Gates）

为了保证迭代效率，每轮建议至少包含 1 个“硬决断点”：

- Gate A（是否继续投入）：必须基于初步 profiling/bench 数据。
- Gate B（是否合入/上线）：必须基于完整 Scorecard 与可复现证据。

常见的阻断条件（P0）：

- 速度提升不显著或波动过大，无法解释来源。
- 误差超过阈值，且没有业务确认接受范围。
- 视觉出现伪影，且无法用开关隔离或回滚。
- 工程化不可复现、不可回滚、或引入高维护成本。

## 5. 如何把 `.worktrees/gpu-stageii-foundation` 纳入评估

`.worktrees/gpu-stageii-foundation` 作为“候选实现资产”处理，原则是**先评估，再决定采纳方式**：

- 评估方式 1：将其作为完整 Candidate，与 Baseline 跑统一 Scorecard。
- 评估方式 2：拆分为若干“可 cherry-pick 的变更批次”，每批次独立评测与 gate。
- 产出要求：必须在对应 round 文档中声明该资产的评估范围、对比口径、与决策出口（采纳/部分采纳/不采纳）。
- 风险控制：默认不直接把 worktree 整体合入主干；更偏向“可控粒度的 cherry-pick + 回滚开关”。
