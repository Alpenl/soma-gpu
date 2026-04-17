# 下一轮建议（Round 0004+ Suggestions）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

R0003 已把控制面初始化与严格复查加固正式收口。下一轮不应继续扩张 docs-only 工作，而应回到 GPU 路线的真实优化目标。

## 1. 建议 1：建立最小可复现 benchmark 与 Scorecard 基线

建议标题：固定一组 workload，产出 baseline scorecard

动机：控制面已经够用了，但后续优化还缺第一版可比较的数据基线。

假设：只要固定 workload、环境记录与结果格式，后续 profiling 和 candidate 评估就能进入可比较状态。

方案范围：

- 建立最小 bench 入口
- 固定至少一组真实 workload
- 输出统一 scorecard

评测计划：

- 速度：E2E latency、throughput
- 误差：至少 1 个对齐指标
- 视觉：固定样例对比
- 工程化：环境记录、一键复现说明

验收门槛（P0）：

- baseline 能稳定复现，且 scorecard 字段完整

决断点（Go/No-Go）：

- 若结果波动大且无法解释，先暂停后续优化，回头修正 benchmark 口径

回滚与降级：

- 该轮优先只增加 bench/记录链路，不修改默认业务路径

依赖与风险：

- 依赖固定样例与环境记录；风险是 workload 不稳定

## 2. 建议 2：对固定 workload 做全链路 profiling

建议标题：输出 Top N 瓶颈清单

动机：没有热点排序，就无法合理决定 GPU 优化优先级。

假设：一次系统性 profiling 足以排除大部分低收益方向。

方案范围：

- E2E 分段耗时
- kernel/op 热点定位
- 输出 Top N 瓶颈清单

评测计划：

- 速度：热点耗时占比
- 误差：profiling 前后结果一致性
- 视觉：固定样例无退化
- 工程化：工具、命令、输出路径可复现

验收门槛（P0）：

- Top N 瓶颈清单可复现，且每项都有证据和下一步建议

决断点（Go/No-Go）：

- 若热点分布不稳定，先回到建议 1 修正基线

回滚与降级：

- profiling 开关默认关闭，不影响正常路径

依赖与风险：

- 依赖 Nsight 或等价工具；风险是采样扰动

## 3. 建议 3：评估 `.worktrees/gpu-stageii-foundation`

建议标题：把 `gpu-stageii-foundation` 当作 candidate asset 做统一评估

动机：仓库中最接近真实收益的候选资产已经存在，不评估就无法判断采纳价值。

假设：统一 scorecard 后，可以较快判断全量采纳、部分采纳还是暂不采纳。

方案范围：

- 方式 A：全量 Candidate 对比
- 方式 B：分批 cherry-pick 对比

评测计划：

- 速度：同 workload 下跑 scorecard
- 误差：对 Reference 做数值比对
- 视觉：固定样例对比
- 工程化：依赖变化、接口变化、回滚策略

验收门槛（P0）：

- 必须形成明确结论：采纳 / 部分采纳 / 不采纳

决断点（Go/No-Go）：

- 若误差或视觉有争议，默认不能直接默认开启

回滚与降级：

- 默认推荐分批 cherry-pick + feature flag

依赖与风险：

- 与主干差异较大；需要严格控制合入粒度
