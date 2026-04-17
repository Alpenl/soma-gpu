# Round 0006 (control-plane-readme-backlink) 概览

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

本轮目标是收掉 R0005 后残留的最后一个入口一致性缺口：`docs/codex-potter/README.md` 仍缺少统一“上层入口”区块。该文件本身是四大主入口之一，若继续保留例外，会让“所有 control-plane 页面标题下都有入口块”的规则留下一个活口。

## 1. 本轮目标（Goals）

- 为 `docs/codex-potter/README.md` 补齐统一“上层入口”区块。
- 把这次最小 docs-only 严格复查登记为正式的 R0006，保持 round 历史与 git 提交链同步。
- 复跑入口块、链接、索引与八件套校验，确认控制面入口规则现已无活跃例外。

## 2. 本轮范围（In Scope）

- `docs/codex-potter/README.md` 的入口块补丁
- `MAIN.md` 与 `docs/codex-potter/iterations/README.md` 的 R0006 索引更新
- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/*` 八件套
- 本地 runtime / KB 同步：`.codexpotter/projects/2026/04/16/1/MAIN.md`、`.codexpotter/kb/*`

## 3. 不在本轮范围（Out of Scope）

- 不扩展到新的模板、协议或额外的控制面规则设计
- 不修改 GPU 核心代码、benchmark、profiling 或 `.worktrees/gpu-stageii-foundation`
- 不回写历史 round 的结论，只新增一轮最小收尾记录

## 4. 本轮输出物（Deliverables）

P0（必须产出）：

- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/round-overview.md`
- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/plan.md`
- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/code.md`
- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/test.md`
- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/next-round-suggestions.md`
- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/summary.md`
- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/commit.md`
- `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/close.md`
- 更新后的 `docs/codex-potter/README.md`
- 更新后的 `MAIN.md`
- 更新后的 `docs/codex-potter/iterations/README.md`

P1（本地但不进 git）：

- 更新后的 `.codexpotter/projects/2026/04/16/1/MAIN.md`
- 更新后的 `.codexpotter/kb/README.md`
- 更新后的 `.codexpotter/kb/project-audit.md`

## 5. Exit Criteria（本轮退出标准）

满足以下条件即可结束本轮：

- `docs/codex-potter/**/*.md` 不再存在缺少“上层入口”区块的活跃控制面文档
- `docs/codex-potter/README.md` 已显式加入与其他入口页一致的回链块
- `MAIN.md` 与 `iterations/README.md` 已正式收录 R0006
- 本轮 `test.md` 已记录格式、入口块、链接与八件套校验结果
