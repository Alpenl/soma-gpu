# Round 0003 (control-plane-hardening) 概览

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

本轮目标是把 `round-0002-potter-entry-normalization` 之后发生的控制面严格复查修正，正式回填为一个可续跑的 round，避免仓库索引只停留在 R0002，而真实提交历史已经继续前进。

## 1. 本轮目标（Goals）

- 把 `78ab839` 到 `0bf9512` 这组控制面修正文档化，并说明各提交解决了什么缺口。
- 新建 `round-0003-control-plane-hardening` 八件套，并把 `MAIN.md` 与 `docs/codex-potter/iterations/README.md` 的运行索引更新到 R0003。
- 复核当前活跃控制面入口仍满足：入口单一、相对路径可移植、示例轮次可直接续跑。

## 2. 本轮范围（In Scope）

- 文档回填：`docs/codex-potter/iterations/round-0003-control-plane-hardening/*`
- 索引更新：`MAIN.md`、`docs/codex-potter/iterations/README.md`
- 本地运行记录同步：`.codexpotter/projects/2026/04/16/1/MAIN.md` 与 `.codexpotter/kb/*`

## 3. 不在本轮范围（Out of Scope）

- 不新增 GPU benchmark、profiling 或实现变更
- 不重写 `round-0001` / `round-0002` 的历史结论，只在本轮解释为何插入一次严格复查加固
- 不修改 `.worktrees/gpu-stageii-foundation` 或其他业务代码

## 4. 本轮输出物（Deliverables）

P0（必须产出）：

- `docs/codex-potter/iterations/round-0003-control-plane-hardening/round-overview.md`
- `docs/codex-potter/iterations/round-0003-control-plane-hardening/plan.md`
- `docs/codex-potter/iterations/round-0003-control-plane-hardening/code.md`
- `docs/codex-potter/iterations/round-0003-control-plane-hardening/test.md`
- `docs/codex-potter/iterations/round-0003-control-plane-hardening/next-round-suggestions.md`
- `docs/codex-potter/iterations/round-0003-control-plane-hardening/summary.md`
- `docs/codex-potter/iterations/round-0003-control-plane-hardening/commit.md`
- `docs/codex-potter/iterations/round-0003-control-plane-hardening/close.md`
- 更新后的 `MAIN.md`
- 更新后的 `docs/codex-potter/iterations/README.md`

P1（本地但不进 git）：

- 更新后的 `.codexpotter/projects/2026/04/16/1/MAIN.md`
- 更新后的 `.codexpotter/kb/README.md`
- 更新后的 `.codexpotter/kb/project-audit.md`

## 5. Exit Criteria（本轮退出标准）

满足以下条件即可结束本轮：

- `MAIN.md` 的“每轮运行索引”已新增 R0003，且与真实提交演进一致
- `docs/codex-potter/iterations/README.md` 的快速入口已包含 R0003
- 本轮 `summary.md` / `commit.md` 能说明 R0002 之后那组严格复查提交解决了哪些控制面缺口
- 本轮 `test.md` 已记录索引、链接、结构与入口一致性的验证结果
