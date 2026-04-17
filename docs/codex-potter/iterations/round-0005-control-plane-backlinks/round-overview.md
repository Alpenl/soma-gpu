# Round 0005 (control-plane-backlinks) 概览

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

本轮目标是把此前只写成纯文本路径或局部互链的 control-plane 文档系统性补成可点击回链，并把“标题下保留上层入口区块”的做法正式写回规范，避免导航再次退化。

## 1. 本轮目标（Goals）

- 为 `docs/codex-potter/` 下缺少主入口回链的治理 / 模板 / round / spec / plan 文档补齐统一的“上层入口”区块。
- 在 `MAIN.md`、`iterations/README.md`、`resume-and-handoff.md`、`templates/README.md` 固化这一规则，防止未来 round 再次漏链。
- 把这次 docs-only 严格复查登记为正式的 R0005，保持 round 历史与 git 提交链同步。

## 2. 本轮范围（In Scope）

- 批量修正文档回链：`docs/codex-potter/governance/*.md`、`templates/*.md`、`plans/*.md`、`specs/*.md`、`iterations/round-*/*.md`
- 规范补强：`MAIN.md`、`docs/codex-potter/governance/resume-and-handoff.md`、`docs/codex-potter/iterations/README.md`、`docs/codex-potter/templates/README.md`
- 新增本轮八件套：`docs/codex-potter/iterations/round-0005-control-plane-backlinks/*`
- 本地 runtime / KB 同步：`.codexpotter/projects/2026/04/16/1/MAIN.md`、`.codexpotter/kb/*`

## 3. 不在本轮范围（Out of Scope）

- 不改 GPU 核心代码、benchmark、profiling 或 `.worktrees/gpu-stageii-foundation`
- 不重写历史 round 的结论，只补导航回链与必要的规则说明
- 不把 `round-overview.md` / `next-round-suggestions.md` 额外扩展成新模板体系

## 4. 本轮输出物（Deliverables）

P0（必须产出）：

- `docs/codex-potter/iterations/round-0005-control-plane-backlinks/round-overview.md`
- `docs/codex-potter/iterations/round-0005-control-plane-backlinks/plan.md`
- `docs/codex-potter/iterations/round-0005-control-plane-backlinks/code.md`
- `docs/codex-potter/iterations/round-0005-control-plane-backlinks/test.md`
- `docs/codex-potter/iterations/round-0005-control-plane-backlinks/next-round-suggestions.md`
- `docs/codex-potter/iterations/round-0005-control-plane-backlinks/summary.md`
- `docs/codex-potter/iterations/round-0005-control-plane-backlinks/commit.md`
- `docs/codex-potter/iterations/round-0005-control-plane-backlinks/close.md`
- 更新后的 `MAIN.md`
- 更新后的 `docs/codex-potter/governance/resume-and-handoff.md`
- 更新后的 `docs/codex-potter/iterations/README.md`
- 更新后的 `docs/codex-potter/templates/README.md`

P1（本地但不进 git）：

- 更新后的 `.codexpotter/projects/2026/04/16/1/MAIN.md`
- 更新后的 `.codexpotter/kb/README.md`
- 更新后的 `.codexpotter/kb/project-audit.md`

## 5. Exit Criteria（本轮退出标准）

满足以下条件即可结束本轮：

- `docs/codex-potter/` 下当前生效文档不再缺少对主入口的可点击回链
- `MAIN.md` / `iterations/README.md` / `resume-and-handoff.md` / `templates/README.md` 已写明“上层入口”规则
- `MAIN.md` 与 `iterations/README.md` 已正式收录 R0005
- 本轮 `test.md` 已记录回链、格式、链接与八件套校验结果
