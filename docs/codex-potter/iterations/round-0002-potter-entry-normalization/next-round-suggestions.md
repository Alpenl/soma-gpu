# 下一轮建议（Round 0003+ Suggestions）

日期：2026-04-16

本轮已经解决了 Potter 入口归一化问题。接下来不建议继续把精力花在工具入口上，而应该回到 GPU 路线的真实目标：速度、误差、视觉效果与工程化。

## 1. 建议 1：建立最小可复现 benchmark 与 Scorecard 汇总

建议标题：建立 baseline/candidate 对比 bench，产出第一版真实 Scorecard

动机：没有真实 bench，当前所有 Gate 只是制度，没有数据。

假设：只要先固定 workload、环境记录和输出格式，就能让后续 profiling 与 candidate 评估不再漂。

方案范围：

- 建立一条最小 bench 入口
- 固定至少一组 workload
- 输出统一 scorecard

评测计划：

- 速度：E2E latency、throughput
- 误差：至少 1 个对齐指标
- 视觉：固定样例对比
- 工程化：环境记录、一键复现说明

验收门槛（P0）：

- baseline 和任一 candidate 能在同一台机器上复现同口径 scorecard

决断点（Go/No-Go）：

- 若 bench 结果波动大且无法解释，则先暂停后续优化，优先解决可复现性

回滚与降级：

- 该轮优先只加工具链，不改默认业务路径

依赖与风险：

- 依赖固定样例与环境记录；风险是 workload 口径不稳定

## 2. 建议 2：做一次全链路 profiling

建议标题：对固定 workload 做 Top N 瓶颈 profiling

动机：没有热点排序，优化优先级容易拍脑袋。

假设：一次系统性 profiling 足以筛掉大部分低收益优化方向。

方案范围：

- E2E 分段耗时
- kernel/op 热点定位
- 输出 Top N 瓶颈清单

评测计划：

- 速度：热点耗时占比
- 误差：profiling 前后输出一致性
- 视觉：对比产物不变
- 工程化：工具、命令、输出路径可复现

验收门槛（P0）：

- 产出 Top N 瓶颈清单，且每项都有证据与下一步建议

决断点（Go/No-Go）：

- 若热点分布不稳定，则先回到建议 1 修正 bench 口径

回滚与降级：

- profiling 开关默认关闭，不影响正常路径

依赖与风险：

- 依赖 Nsight 或等价工具；风险是采样扰动

## 3. 建议 3：评估 `.worktrees/gpu-stageii-foundation`

建议标题：把 `gpu-stageii-foundation` 作为 candidate asset 做统一评估

动机：当前最接近真实收益的候选资产已经存在，不评估就等于浪费已有工程投入。

假设：统一 scorecard 后，可以快速判断全量采纳、部分采纳还是暂不采纳。

方案范围：

- 方式 A：全量对比
- 方式 B：分批 cherry-pick 对比

评测计划：

- 速度：同 workload 下跑 scorecard
- 误差：对 Reference 做数值比对
- 视觉：固定样例对比
- 工程化：依赖变化、接口变化、回滚策略

验收门槛（P0）：

- 必须形成明确结论：采纳 / 部分采纳 / 不采纳

决断点（Go/No-Go）：

- 若误差或视觉有争议，默认不能“直接默认开启”

回滚与降级：

- 默认推荐分批 cherry-pick + feature flag

依赖与风险：

- 与主干差异较大；需要控制合入粒度
