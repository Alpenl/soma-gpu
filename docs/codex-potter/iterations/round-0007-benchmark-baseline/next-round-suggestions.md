# 下一轮建议（Next Round Suggestions）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

## 建议 1：把 committed foundation 拆到干净 candidate 分支

动机：`.worktrees/gpu-stageii-foundation` 已提交的 foundation 有明确测试，但当前 worktree 叠了未提交 sequence/render 扩展，不能直接作为统一 candidate。

假设：先 cherry-pick `9e686ea` 及其依赖提交到干净分支，再接同一套 benchmark/scorecard，比继续在 dirty tree 上讨论 merge 更稳。

方案范围：只引入 torch foundation 的已提交文件与测试，不带本地 dirty 的 sequence/render 扩展。

评测计划：在干净 candidate 分支上复跑 foundation 关键测试，并把 public benchmark 扩到“sample parse + candidate smoke”双 workload。

验收门槛：形成 `main` vs clean candidate 的同口径 scorecard，而不是只有工程化 audit。

决断点：若 cherry-pick 后仍无法得到统一 workload，就继续停在 foundation 层，不进入 merge 讨论。

回滚与降级：保持 candidate 分支与 `main` 解耦，必要时整批回退。

依赖与风险：依赖对 worktree 提交边界的精确识别；若继续在 dirty tree 上追加改动，会再次污染评估口径。

## 建议 2：补齐 mesh / mp4 环境后，把 benchmark 扩到产物链

动机：round-0007 已证明 public benchmark 可以先从 stageii-ingest 起步，但目标链路仍是从 mocap 到 mesh/mp4。

假设：在拿到 `psbody`、licensed `model.npz` 和 `blender` 后，可以把同一套 scorecard 扩展到 mesh export / mp4 render，不必重做控制面。

方案范围：优先验证 `save_smplx_verts.py` 和 `src/soma/render/parameters_to_mesh.py` 的公开最小路径，再决定是否接 Blender 输出 mp4。

评测计划：对齐 metrics framework，至少补速度、产物、工程化三类，并记录每个阻塞点的解除方式。

验收门槛：同一份 scorecard 中出现真实 mesh 或 mp4 产物路径，不再只有 blocked stages。

决断点：若依赖仍不完整，就维持 stageii-ingest baseline，不编造 full E2E 数字。

回滚与降级：保持 stageii-ingest harness 不变，新增 mesh/mp4 workload 作为扩展项。

依赖与风险：依赖外部环境和受许可模型资产，不是仅靠仓库改动就能完成。

## 建议 3：在统一 workload 上做第一次 profiling

动机：当前 round-0007 修复了 benchmark 口径，但还没有对 `moshpp/chmosh.py` 的热点做正式 profiling。

假设：有了稳定 workload 之后，profiling 才不会变成“量不同东西”。

方案范围：先针对 released `stageii` 路线做一次最小 profiling，输出 Top N 热点，不进入优化代码。

评测计划：保留 benchmark harness，新增 profiling 命令、热点表和解释性记录。

验收门槛：能明确回答“下一轮先投优化器循环、marker attachment，还是 I/O / render glue”。

决断点：若 profiling 面仍受环境阻塞，就先扩 benchmark 面，不进入热点优化。

回滚与降级：profiling 资产只作为文档与日志，不改 released 路线。

依赖与风险：依赖至少一个稳定、可重复的 execution workload；否则热点结论不可信。
