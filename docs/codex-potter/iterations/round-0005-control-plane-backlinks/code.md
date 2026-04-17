---
round:
  id: "round-0005-control-plane-backlinks"
  date: "2026-04-17"
  status: "done"
repo:
  branch: "main"
---

# 本轮编码 / 执行记录（Code Log）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 1. 实际做了哪些改动

- 用一次性脚本为 43 个既有 `docs/codex-potter/*.md` 文档补齐统一“上层入口”区块，覆盖：
  - `governance/metrics-framework.md`
  - `governance/resume-and-handoff.md`
  - `templates/*.md`
  - `plans/*.md`
  - `specs/*.md`
  - `iterations/round-0001-*/*.md` 到 `iterations/round-0004-*/*.md`
- 手工补充规则与索引：
  - `MAIN.md`：新增 R0005，并把“新增文档必须至少回链其一”细化为“标题下保留统一上层入口区块”
  - `docs/codex-potter/governance/resume-and-handoff.md`：在轻审标准与交接检查清单中加入入口审计
  - `docs/codex-potter/iterations/README.md`：新增 R0005 快速入口，并把“上层入口”区块写成 round 级规范
  - `docs/codex-potter/templates/README.md`：要求模板保留入口块，`round-overview.md` / `next-round-suggestions.md` 手工补齐同样回链
- 新建 `docs/codex-potter/iterations/round-0005-control-plane-backlinks/` 八件套，正式登记本轮 docs-only 收口。

## 2. 与计划偏差

- 原计划只需为现有文档补链；执行中确认 `round-overview.md` / `next-round-suggestions.md` 没有模板来源，因此额外在 `iterations/README.md` 与 `templates/README.md` 补了防复发说明。
- 没有新增独立模板文件。原因：本轮目标是导航补强与规则固化，继续扩模板体系会把 docs-only 收口扩张成新的设计工作。

## 3. 关键实现说明

- 既有文档数量较多，且相对路径取决于目录深度；批量插入“上层入口”区块比手工逐页 patch 更稳妥，因此使用一次性脚本统一生成相对路径。
- 主入口页 `docs/codex-potter/README.md`、`governance/workflow-protocol.md`、`iterations/README.md` 本身已具备入口属性，没有重复插入区块；只在需要的规范页上补写规则。

## 4. 未提交本地状态

- `.codexpotter/projects/2026/04/16/1/MAIN.md`
- `.codexpotter/kb/README.md`
- `.codexpotter/kb/project-audit.md`

原因：这些文件用于 runtime 状态与本地知识捕获，按约定不进 git。
