# Round 0004 (entry-link-hardening) 概览

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

本轮目标是补齐活跃控制面入口之间仍然缺失的互链，让 `MAIN.md` 已声明的 4 个主入口真正形成稳定导航网，而不是只在规则层面要求“互链”。

## 1. 本轮目标（Goals）

- 补上 `docs/codex-potter/governance/workflow-protocol.md` 对轮次索引的回链。
- 补上 `docs/codex-potter/iterations/README.md` 对控制面总说明与工作流协议的回链。
- 把这次 docs-only 严格复查登记为正式的 R0004，避免再次出现“有修正提交、无 round 记录”的历史断层。

## 2. 本轮范围（In Scope）

- 活跃入口修正：`docs/codex-potter/governance/workflow-protocol.md`、`docs/codex-potter/iterations/README.md`
- 运行索引更新：`MAIN.md`
- 新增本轮八件套：`docs/codex-potter/iterations/round-0004-entry-link-hardening/*`
- 本地 runtime / KB 同步：`.codexpotter/projects/2026/04/16/1/MAIN.md`、`.codexpotter/kb/*`

## 3. 不在本轮范围（Out of Scope）

- 不改 GPU 代码、benchmark、profiling 或 `.worktrees/gpu-stageii-foundation`
- 不重写 `round-0003` 的历史结论，只解释为什么在其后仍需一次最小 docs-only 收口
- 不扩散到非活跃历史文档的措辞清理

## 4. 本轮输出物（Deliverables）

P0（必须产出）：

- `docs/codex-potter/iterations/round-0004-entry-link-hardening/round-overview.md`
- `docs/codex-potter/iterations/round-0004-entry-link-hardening/plan.md`
- `docs/codex-potter/iterations/round-0004-entry-link-hardening/code.md`
- `docs/codex-potter/iterations/round-0004-entry-link-hardening/test.md`
- `docs/codex-potter/iterations/round-0004-entry-link-hardening/next-round-suggestions.md`
- `docs/codex-potter/iterations/round-0004-entry-link-hardening/summary.md`
- `docs/codex-potter/iterations/round-0004-entry-link-hardening/commit.md`
- `docs/codex-potter/iterations/round-0004-entry-link-hardening/close.md`
- 更新后的 `MAIN.md`
- 更新后的 `docs/codex-potter/governance/workflow-protocol.md`
- 更新后的 `docs/codex-potter/iterations/README.md`

P1（本地但不进 git）：

- 更新后的 `.codexpotter/projects/2026/04/16/1/MAIN.md`
- 更新后的 `.codexpotter/kb/README.md`
- 更新后的 `.codexpotter/kb/project-audit.md`

## 5. Exit Criteria（本轮退出标准）

满足以下条件即可结束本轮：

- `MAIN.md` 的运行索引已新增 R0004
- `workflow-protocol.md` 与 `iterations/README.md` 的缺链已补齐
- `docs/codex-potter/iterations/README.md` 的快速入口已包含 R0004
- 本轮 `test.md` 已记录互链、格式与结构校验结果
