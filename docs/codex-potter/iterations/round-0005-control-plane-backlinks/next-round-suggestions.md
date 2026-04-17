# Round 0005 下一轮建议

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 建议 1：进入 benchmark baseline round

- 动机：控制面导航已经收口到足够稳定的程度，继续做 docs-only 严格复查的收益明显下降；真实 GPU 路线仍缺少可执行 benchmark gate。
- 假设：只要先固定 workload、命令与 scorecard，后续 profiling / candidate 评估就能从“感觉优化”切到“证据驱动优化”。
- 方案范围：建立 `round-0006-benchmark-baseline`，先做可复现 workload、指标表与最小结果产物。
- 评测计划：按 `metrics-framework.md` 覆盖速度 / 误差 / 工程化至少一项 P0 指标。
- 验收门槛：至少有一条可复现 benchmark 命令和一版 scorecard 草案。
- 依赖与风险：若缺 benchmark 数据或私有资产，需要先定义最小公开样例或 toy workload。

## 建议 2：把 docs-only 修正降为被动项

- 动机：连续多轮控制面修补已经解决大多数导航问题，再主动扩张 docs-only 收口会挤占真实 GPU 迭代。
- 假设：把文档修正降为“发现阻断时再补”，能把主要精力转回 benchmark / profiling / candidate 评估。
- 方案范围：后续仅在出现新的续跑阻断时插入 docs-only round；平时把文档修正并入真实功能 round。
- 验收门槛：下一轮不再以纯文档修正作为主目标。

## 建议 3：用统一 scorecard 评估 `.worktrees/gpu-stageii-foundation`

- 动机：仓库已存在 torch stageii 候选资产，但还没有统一证据说明应整体采纳、局部 cherry-pick 还是继续观望。
- 假设：先有 benchmark baseline，再对候选资产做 A/B 评估，能更快形成 Go/No-Go 决策。
- 方案范围：整理 candidate 与 baseline 的对比命令、样例集和误差阈值。
- 验收门槛：形成一页可复现的 candidate 评估记录，而不是口头判断。
