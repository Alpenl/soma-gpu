# 下一轮建议（Next Round Suggestions）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

## 建议 1：建立 benchmark baseline round

动机：控制面入口规则现已基本收敛，继续主动做 docs-only 修补的收益明显下降；真实 GPU 路线仍缺少可复现的基线与 scorecard。

假设：先固定 workload、命令与输出指标，可以为后续 profiling 和 candidate 评估提供统一门槛。

方案范围：新增一个以 benchmark baseline 为主题的 round，优先覆盖关键路径、输入资产假设、统计口径与结果存档位置。

评测计划：对齐 metrics framework，至少补速度与工程化两项基线证据。

验收门槛：能复现跑出第一版 benchmark，且结果可以回写到 round 文档。

决断点：若 benchmark 输入资产仍不清晰，先停在“基线定义”而不要继续写 GPU 优化代码。

回滚与降级：若私有资产暂不可用，可先用 toy workload 或缩小数据切片定义最小基线。

依赖与风险：需要明确 workload、数据位置与现有可运行命令。

## 建议 2：用统一 scorecard 评估 `.worktrees/gpu-stageii-foundation`

动机：候选实现资产已经存在，但尚未进入主仓库的统一 gate。

假设：先做 candidate 评估，比盲目重写更能减少返工。

方案范围：把 worktree 中的 torch stageii 能力拆成可评测的 candidate 或变更批次，并按 metrics framework 记录结论。

评测计划：至少覆盖速度、误差与工程化三项。

验收门槛：形成明确的采纳 / 部分采纳 / 暂不采纳结论。

决断点：若 candidate 无法在统一 benchmark 上复现，则先补 benchmark，不进入合并讨论。

回滚与降级：保持评估与主仓库实现解耦，不直接整树合入。

依赖与风险：依赖 benchmark baseline 已经建立。

## 建议 3：若入口审计再次回归，抽出自动 lint

动机：本次 R0006 说明纯人工 review 仍可能漏掉单页例外。

假设：把“上层入口”块检查脚本固化成可复用 lint，比持续手工 round 更省成本。

方案范围：仅在未来再次出现入口块回归时，考虑把当前 ad-hoc 校验脚本整理成仓库脚本或 CI 检查。

评测计划：验证脚本能稳定识别缺失入口块与断链。

验收门槛：对现有 control-plane 文档零误报、零漏报。

决断点：若未来 2 到 3 轮都不再出现导航回归，则无需为此新增正式工具链。

回滚与降级：保持脚本为本地工具，不强制接入 CI。

依赖与风险：若规则继续变化，脚本口径也要同步维护。
