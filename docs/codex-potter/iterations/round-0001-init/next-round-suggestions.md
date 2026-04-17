# 下一轮建议（Round 0002+ Suggestions）与决断点

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-16

本文档用于给出下一轮及后续几轮的建议写法、建议清单的最小模板，以及必须设置的决断点（decision gates）。它不是待办列表，而是“可评估、可拍板”的建议集合。

统一指标口径与 Scorecard 模板见：[metrics-framework.md](../../governance/metrics-framework.md)。

## 1. 建议条目模板（必须按此写）

复制以下模板作为每条建议的正文：

```text
建议标题：
动机：
假设：
方案范围：
评测计划：
- 速度：
- 误差：
- 视觉：
- 工程化：
验收门槛（P0）：
决断点（Go/No-Go）：
回滚与降级：
依赖与风险：
```

## 2. 决断点写法（强制要求）

每条建议至少包含 1 个硬决断点，推荐拆成两级：

- Gate A（探索继续与否）：在做完 profiling 或最小 bench 后决定是否继续投入。
- Gate B（合入/默认开启与否）：在完整 Scorecard 与可复现证据齐全后决定是否合入或默认开启。

决断点必须写清：

- 触发时机（例如“拿到第一版 Nsight profile 后”/“跑完固定样例集后”）。
- 证据清单（必须提供哪些数据/产物）。
- 负责人（谁拍板，谁提供证据）。
- 失败处理（不满足门槛时是放弃、改方案、还是降级/开关上线）。

## 3. 如何把 `.worktrees/gpu-stageii-foundation` 作为候选实现资产纳入评估

当建议涉及 `.worktrees/gpu-stageii-foundation` 时，必须在“方案范围/评测计划/风险”中写明：

- 引用方式（全量对比）：将 worktree 视为一个整体 Candidate，与 Baseline 直接对比 Scorecard。
- 引用方式（分批采纳）：将 worktree 的变更拆分为若干批次（cherry-pick），每批次单独评测并设置 gate。
- 决策出口（采纳）：明确合入策略（直接合入/分批合入/默认关闭开关）。
- 决策出口（部分采纳）：列出采纳子集与拒绝子集。
- 决策出口（不采纳）：写明原因（指标不达标/工程化风险过高/与主干方向不一致等）。
- 风险控制：默认要求提供回滚开关或可 revert 的最小粒度 PR。
- 风险控制：如果误差或视觉存在争议，必须强制“默认关闭 + 明确试验范围”。

## 4. 初版建议清单（用于 Round 0002+）

以下是面向“GPU 优化迭代”的建议方向。它们刻意写成可直接落到下一轮文档的形式，但具体内容仍需下一轮负责人补齐实测数据与证据。

### 建议 1：建立可复现的基线 benchmark 与 Scorecard 自动汇总

建议标题：建立最小可复现 benchmark（baseline/candidate 对比）与 Scorecard 自动汇总

动机：没有可复现 bench，后续任何“提速/提质”都无法稳定评估，容易陷入主观争论与重复劳动。

假设：统一输入与统计口径后，可以显著降低性能波动带来的误判，并提高迭代速度。

方案范围：新增或完善最小 bench 入口（命令 + 固定 workload），输出原始日志与 scorecard 汇总（格式对齐 metrics framework）。

评测计划：

- 速度：采集 E2E latency（p50/p90/p99）与 throughput；报告 warm-up、采样次数与波动。
- 误差：选定 Reference，输出至少 1 个误差指标与稳定性结论（NaN/Inf/崩溃）。
- 视觉：生成固定样例集对比图或视频，并给出人工审阅结论。
- 工程化：提供一键复现说明（环境信息、命令、输出路径）；回滚策略清晰。

验收门槛（P0）：能在同一台机器上复现 baseline 与任一 candidate 的 scorecard；口径一致；输出完整。

决断点（Go/No-Go）：完成最小 bench 后，如果结果波动无法控制或无法解释来源，则暂停后续优化轮次，优先解决可复现性问题。

回滚与降级：不涉及业务逻辑变更，主要是工具链；若影响构建则拆分/隔离到独立模块。

依赖与风险：依赖固定样例集与环境记录；风险是“口径不统一”导致后续结果不可比，需强制用 metrics framework 的模板。

### 建议 2：做一次全链路 profiling，明确 Top N 瓶颈并设定优化优先级

建议标题：全链路 profiling，产出 Top N 瓶颈清单与优化路线图

动机：没有瓶颈证据的优化通常低效，且容易引入复杂度与误差风险。

假设：通过 Nsight/日志分段可以快速锁定大头耗时，优先做收益最高的 1-2 个点。

方案范围：对固定 workload 做端到端与 kernel 级 profiling；输出瓶颈排序与可行动建议。

评测计划：

- 速度：基线 E2E 与 kernel 时间占比；记录热点算子/调用栈。
- 误差：profiling 不改结果，但要验证“profiling 开关不改变数值输出”。
- 视觉：同误差，确认输出一致。
- 工程化：profiling 方法可复现（命令、工具版本、输出文件）。

验收门槛（P0）：产出 Top N 瓶颈清单，每项包含证据（截图/报告/日志）与可落地的下一步。

决断点（Go/No-Go）：若 Top N 瓶颈无法稳定复现（热点飘忽），先回到建议 1 解决可复现性。

回滚与降级：profiling 相关变更必须可关闭；不得影响默认路径性能。

依赖与风险：依赖工具链与驱动；风险是测量扰动与环境噪声。

### 建议 3：评估 `.worktrees/gpu-stageii-foundation` 作为候选实现资产

建议标题：把 `.worktrees/gpu-stageii-foundation` 作为 Candidate asset 做一次系统性评估

动机：该 worktree 可能包含可复用的工程基础或性能实现，但如果不按统一口径评估，采纳风险不可控。

假设：通过统一 Scorecard，可以快速判断“全量采纳/部分采纳/不采纳”，并明确最小风险落地路径。

方案范围：

- 方式 A：全量对比 baseline vs worktree（把其视作 Candidate）。
- 方式 B：拆分为若干批次 cherry-pick（按风险/收益分组），分批评测与 gate。

评测计划：

- 速度：对同一 workload 跑 scorecard；若有显著提升，进一步解释性 profiling 定位来源。
- 误差：对 Reference 跑误差指标与稳定性；若引入近似算法，必须写阈值与业务确认需求。
- 视觉：固定样例集对比产物 + 人工结论。
- 工程化：对比依赖变化、构建步骤、API 破坏；必须有回滚策略与开关建议。

验收门槛（P0）：形成明确决策（采纳/部分采纳/不采纳），并给出证据与风险缓解方案。

决断点（Go/No-Go）：如果误差或视觉出现争议，默认 `No-Go（默认开启）`，只能 `Go（合入但默认关闭）` 或 `部分采纳`。

回滚与降级：建议以分批 cherry-pick + feature flag 为默认落地策略。

依赖与风险：与主干差异较大时，合并成本与回归风险上升；需要拆分粒度与逐批 gate 控制。
