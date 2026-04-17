---
round:
  id: "round-0005-control-plane-backlinks"
  date: "2026-04-17"
  status: "done"
repo:
  branch: "main"
  base_commit: "acb6ce16c9f034f603b0614201186bf26fa11baa"
roles:
  orchestrator: "codex"
  workers: []
scope_tags:
  - "docs"
  - "control-plane"
  - "navigation"
---

# 本轮计划（Plan）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 0. 本轮一句话目标

为现有 control-plane 文档补齐可点击主入口回链，并把这一约束固化进规范与 round 历史。

## 1. 强制阅读清单（开始前必须完成）

- [x] `MAIN.md`：确认主入口互链约定已要求新增文档至少回链主入口其一
- [x] `docs/codex-potter/iterations/README.md`：确认当前只定义了八件套结构，尚未把“上层入口”区块写成显式规则
- [x] `docs/codex-potter/governance/workflow-protocol.md`：确认四大主入口本身已互链
- [x] `docs/codex-potter/governance/resume-and-handoff.md`：确认轻审与交接检查清单尚未显式要求“上层入口”区块
- [x] `docs/codex-potter/templates/README.md`：确认 6 份模板目前可承载入口块，但 `round-overview.md` / `next-round-suggestions.md` 没有模板来源
- [x] `docs/codex-potter/iterations/round-0004-entry-link-hardening/{summary,test,close}.md`：确认 R0004 只修了四大主入口，尚未扩展到 round/template/governance 其余页面

阅读结论（要点）：

- `MAIN.md` 已给出“新增文档必须至少回链主入口其一”的规则，但现存大量文档只写了纯文本路径或局部互链，没有可点击主入口回链。
- 缺口主要集中在治理补充页、模板、历史 round 八件套，以及 plan/spec 文档；这说明问题不在某一页，而在规则没有落成统一写法。
- 由于 `round-overview.md` / `next-round-suggestions.md` 没有模板来源，只改 6 份模板不足以完全防复发，仍需把规则写进轮次规范与交接规范。

## 2. 范围（In Scope）与非目标（Out of Scope）

In Scope（本轮要做）：

- 批量补齐 `docs/codex-potter/**/*.md` 中缺失的“上层入口”区块
- 在主入口与规范页写明该区块的最低要求
- 新建 `round-0005-control-plane-backlinks` 八件套并更新运行索引

Out of Scope（本轮明确不做）：

- 不新增 GPU benchmark / profiling / candidate 评估代码
- 不改变历史 round 的核心结论，只补导航和规则
- 不额外创建 `round-overview` / `next-round-suggestions` 模板文件

约束（必须遵守）：

- 只修改 control-plane Markdown 文档与本地 runtime / KB 记录
- 不把 `.codexpotter/` 提交进 git
- 不回退用户已有更改

## 3. 产物与完成定义（Deliverables & DoD）

本轮必须产物：

- `round-overview.md`
- `plan.md`（本文件）
- `code.md`
- `test.md`
- `next-round-suggestions.md`
- `summary.md`
- `commit.md`
- `close.md`
- 文档改动：`MAIN.md`、`docs/codex-potter/governance/resume-and-handoff.md`、`docs/codex-potter/iterations/README.md`、`docs/codex-potter/templates/README.md`、以及缺链文档批量补丁

完成定义（DoD），需可验收：

- [x] 缺链的 control-plane 文档已统一补上“上层入口”区块
- [x] 规则层文档已写明未来 round 如何保持这一入口块
- [x] `MAIN.md` 与 `iterations/README.md` 已收录 R0005
- [x] `code.md` 记录了批量补丁与规则补丁的实际落点
- [x] `test.md` 里至少有 1 条可执行验证命令，且覆盖回链与结构检查
- [x] `commit.md` 记录分支名与 commit hash（至少一个）
- [x] `summary.md` 记录分支名与 commit hash（至少一个）
- [x] `close.md` 指明下一轮入口与退出判断

## 4. 任务拆分与派发（面向子代理）

本轮未启用子代理，主会话直接完成最小 docs-only 收口。

1. 任务：审计缺失的主入口回链
   - 目标：枚举 `docs/codex-potter/` 下当前没有可点击主入口回链的文档集合
   - 允许修改：无（只读扫描）
   - 验证：Python 脚本输出缺口列表

2. 任务：补齐入口块并固化规则
   - 目标：为缺链文档统一插入“上层入口”区块，并在规范页写清楚最低要求
   - 允许修改：`docs/codex-potter/**/*.md`、`MAIN.md`
   - 验证：缺口扫描归零，相关 diff 无格式问题

3. 任务：登记 R0005 并复跑验证
   - 目标：新增本轮八件套、更新索引、记录测试结果
   - 允许修改：`docs/codex-potter/iterations/round-0005-control-plane-backlinks/*`、`MAIN.md`、`docs/codex-potter/iterations/README.md`
   - 验证：八件套齐全、快速入口可跳转、链接校验通过

## 5. 风险清单（Risks）与应对

- 风险：批量补丁可能误伤已稳定入口页
  - 影响：产生重复或错误的回链区块
  - 触发信号：主入口文档 diff 出现意外变更
  - 应对：dry-run 先枚举目标集，主入口页列入跳过名单

- 风险：只改现有文档，不改规则，后续 round 仍会复发
  - 影响：下一轮新增文档继续漏回链
  - 触发信号：未来 round 再出现“纯文本路径但无回链”的页面
  - 应对：同步在 `MAIN.md`、`iterations/README.md`、`resume-and-handoff.md`、`templates/README.md` 固化写法

## 6. 验证计划（Test Plan）

最小验证（必须）：

- `git diff --check`：确认无格式问题
- Python 脚本：确认 `docs/codex-potter/**/*.md` 不再缺少主入口 Markdown 回链
- Python 脚本：确认 `README.md`、`MAIN.md` 与 `docs/codex-potter/**/*.md` 的本地 Markdown 链接均可解析
- Python 脚本：确认 `round-0001` 到 `round-0005` 的八件套齐全

扩展验证（有时间就做）：

- `rg -n "控制面回链补强|round-0005-control-plane-backlinks"`：确认索引与快速入口已同步到 R0005

## 7. 提交计划（Git Plan）

本轮预计提交策略：

- 分支：`main`
- 提交拆分：先提交回链补丁与 R0005 目录，再回填 commit metadata
- 提交信息约定：`docs: add round-0005 control plane backlinks`、`docs: record round-0005 commit metadata`
