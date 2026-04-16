# Round 0001 (init) 概览

日期：2026-04-16

本轮为“初始化轮”（init round），目标是为后续 GPU 优化迭代建立**统一的指标框架与轮次文档规范**，并明确如何把 `.worktrees/gpu-stageii-foundation` 作为候选实现资产纳入评估。

## 1. 本轮目标（Goals）

- 建立统一指标框架：速度、误差、视觉效果、工程化四大类指标的口径、测量要求与输出格式。
- 建立轮次规范：每轮必备文档、建议写法、决断点（decision gates）的最小可执行规范。
- 明确候选资产评估方法：把 `.worktrees/gpu-stageii-foundation` 作为候选实现资产纳入后续评估的统一流程与要求。

## 2. 本轮范围（In Scope）

- 文档初始化：
  - 指标框架文档（metrics framework）初版。
  - 迭代轮次组织与写作规范（iterations README）初版。
  - Round 0001 的概览与下一轮建议初版。
- 评估流程初始化：
  - 明确后续轮次如何产出统一 Scorecard（即使本轮不产出 bench 数据，也要把模板落地）。
  - 明确 `.worktrees/gpu-stageii-foundation` 的评估口径、决策出口、与风险控制原则。

## 3. 不在本轮范围（Out of Scope）

- 不做任何实际 GPU 性能优化实现（不改 kernel、不改算法、不改工程构建）。
- 不要求本轮生成可运行 benchmark 工具或 CI 流水线（后续轮次再补齐）。

## 4. 本轮输出物（Deliverables）

P0（必须产出）：

- `docs/codex-potter/governance/metrics-framework.md`
- `docs/codex-potter/iterations/README.md`
- `docs/codex-potter/iterations/round-0001-init/round-overview.md`
- `docs/codex-potter/iterations/round-0001-init/plan.md`
- `docs/codex-potter/iterations/round-0001-init/test.md`
- `docs/codex-potter/iterations/round-0001-init/summary.md`
- `docs/codex-potter/iterations/round-0001-init/next-round-suggestions.md`

P1（可选，若团队在本轮就已具备条件可补充，但不作为阻断项）：

- 用于后续轮次复用的 Scorecard 示例（填充真实数据的样例）。

## 5. Exit Criteria（本轮退出标准）

满足以下条件即可结束本轮并进入下一轮：

- 指标框架已定义四大类指标的 P0 口径与统一 Scorecard 模板。
- 轮次文档规范已定义：每轮最小文档集合、下一轮建议模板、决断点规则。
- `.worktrees/gpu-stageii-foundation` 的“作为候选资产”评估口径已写清：评估方式、产出要求、风险控制与决策出口。

## 6. `.worktrees/gpu-stageii-foundation` 纳入评估（本轮结论）

本轮只定义“如何评估”，不对该资产本身做技术结论。统一要求如下：

- 该 worktree 视为 Candidate asset，可用于全量对比或拆分 cherry-pick 后分批对比。
- 后续任何轮次若引用该资产，必须在 round 文档中写明：
  - 资产描述：它解决什么问题、包含哪些关键变更。
  - 评估策略：全量 vs 分批；每批的验收门槛（P0）。
  - 决策出口：采纳/部分采纳/不采纳，以及理由。
