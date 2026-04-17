# 下一轮建议（Next Round Suggestions）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 建议 1：进入 `round-0005-benchmark-baseline`

- 动机：控制面入口现在已能稳定续跑，继续停留在 docs-only 会偏离 GPU 主目标。
- 假设：先建立最小 benchmark 和 scorecard，后续 profiling 与优化才有统一 gate。
- 方案范围：固定 workload、固定环境、固定输出指标，把“更快 / 更准 / 更工程化”落成第一版量化表。
- 评测计划：记录最小可复现命令、基线耗时、关键误差指标、产物路径。
- 验收门槛：至少形成一条稳定可复跑 benchmark 命令和一份可比较的 scorecard 草稿。
- 决断点：如果连 workload 与输入资产都无法固定，应先补 runbook，而不是盲目优化代码。
- 回滚与降级：若短期无法拿到完整私有资产，可先用 toy workload 建立骨架。
- 依赖与风险：依赖可共享的测试资产、明确的计时口径与输出对齐方式。

## 建议 2：对固定 workload 做 profiling

- 动机：在没有 Top N 热点前，任何 GPU 优化都会变成拍脑袋式改动。
- 假设：profiling 可以验证旧文档中关于 `moshpp/chmosh.py` / `mosh_stageii(...)` 的瓶颈判断是否仍成立。
- 方案范围：限定在基线 workload 上采样 CPU/GPU 时间占比、同步点与热点函数。
- 评测计划：记录 profiling 命令、热点表和初步解释。
- 验收门槛：至少产出一份 Top N 瓶颈清单，并能映射到具体模块。
- 决断点：若热点不在预期路径，应调整后续优化计划。
- 回滚与降级：profiling 失败时先保留 benchmark skeleton，不阻塞基线建设。
- 依赖与风险：依赖环境稳定、计时方式一致，且不能被首次加载开销污染。

## 建议 3：统一评估 `.worktrees/gpu-stageii-foundation`

- 动机：仓库已有 candidate asset，但没有经过统一 gate 审视。
- 假设：在 benchmark / profiling 基线就位后，可以更客观地决定采纳、部分采纳或暂不采纳。
- 方案范围：把该 worktree 视作 candidate，与 baseline 跑同一套 scorecard。
- 评测计划：比较速度、误差、稳定性与工程化成本。
- 验收门槛：形成明确的 Go / No-Go 结论，而不是停留在“看起来有方向”。
- 决断点：若误差或维护成本过高，应拆分为可 cherry-pick 的小批次。
- 回滚与降级：保持主干不直接吸收整个 worktree。
- 依赖与风险：依赖可复现 benchmark、清晰的基线 commit 与统一的验证口径。
