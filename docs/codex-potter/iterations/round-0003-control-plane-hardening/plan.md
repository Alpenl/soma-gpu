---
round:
  id: "round-0003-control-plane-hardening"
  date: "2026-04-17"
  status: "done"
repo:
  branch: "main"
  base_commit: "216d47fb8d07025126fc6e2933e5e3332671d534"
  head_commit: "6eeae6d43f481e79295b01730d63b74964e9bd3a"
roles:
  orchestrator: "main-session"
  workers:
    - "none (strict review executed locally)"
scope_tags:
  - "docs"
  - "control-plane"
  - "audit"
  - "handoff"
---

# 本轮计划（Plan）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 0. 本轮一句话目标

把 `round-0002-potter-entry-normalization` 之后的控制面严格复查修正，回填成正式的 R0003 交接包与运行索引，避免“真实提交继续前进，但 round 文档停在旧状态”。

## 1. 强制阅读清单（开始前必须完成）

- [x] `MAIN.md`：确认仓库级控制面入口仍只记录到 R0002。
- [x] `docs/codex-potter/iterations/round-0002-potter-entry-normalization/summary.md`：确认 R0002 结论是“入口归一化完成，下一轮应回到 benchmark / profiling”。
- [x] `docs/codex-potter/iterations/round-0002-potter-entry-normalization/commit.md`：确认 R0002 只覆盖到 `4fb8c2a` / `216d47f`，未包含后续严格复查提交。
- [x] `docs/codex-potter/iterations/round-0002-potter-entry-normalization/close.md`：确认当前历史文档尚未把严格复查加固登记为新 round。
- [x] `docs/codex-potter/governance/workflow-protocol.md`：确认协议要求每轮都要产出八件套并更新 `MAIN.md` 运行索引。
- [x] `docs/codex-potter/governance/resume-and-handoff.md`：确认本轮需要记录交接包、验证与提交边界。
- [x] `.codexpotter/projects/2026/04/16/1/MAIN.md`：确认 runtime progress file 的 Done 已记录多次严格复查修正，但 git 中缺少对应 round 目录。

阅读结论（要点）：

- `78ab839` 到 `0bf9512` 这组提交都属于控制面严格复查加固，但仓库索引仍停留在 R0002。
- 继续把这些修正只记在 runtime progress file 中，会让 git 内的 round 历史与真实提交链长期分叉。
- 最小且可审计的补救方式不是重写 R0002，而是新增一个回填型 R0003，把严格复查加固正式纳入轮次体系。

## 2. 范围（In Scope）与非目标（Out of Scope）

In Scope（本轮要做）：

- 新建 `docs/codex-potter/iterations/round-0003-control-plane-hardening/` 八件套
- 更新 `MAIN.md` 与 `docs/codex-potter/iterations/README.md`，让运行索引和快速入口指向 R0003
- 记录并解释 `78ab839`、`5391542`、`b4d786d`、`4d390bc`、`c6ede83`、`0bf9512` 这组提交的作用
- 在本地 progress file / KB 中写回严格复查结论

Out of Scope（本轮明确不做）：

- 不进入 benchmark、profiling 或 GPU 路线代码修改
- 不把每个严格复查提交都再拆成独立 round
- 不修改 `round-0001` / `round-0002` 的目标与结论

约束（必须遵守）：

- 只改控制面文档与本地 `.codexpotter` 状态
- 所有描述必须以真实 git 提交与实际文件落点为准，不编造未发生的执行路径
- 下一轮建议必须重新回到真实 GPU 路线，而不是继续无限扩张初始化任务

## 3. 产物与完成定义（Deliverables & DoD）

本轮必须产物：

- `MAIN.md`
- `docs/codex-potter/iterations/README.md`
- `docs/codex-potter/iterations/round-0003-control-plane-hardening/*`
- `.codexpotter/projects/2026/04/16/1/MAIN.md`（本地 runtime 状态）
- `.codexpotter/kb/README.md` 与 `.codexpotter/kb/project-audit.md`（本地知识捕获）

完成定义（DoD），需可验收：

- [x] `MAIN.md` 已收录 R0003
- [x] `docs/codex-potter/iterations/README.md` 已新增 R0003 快速入口
- [x] R0003 八件套齐全
- [x] `summary.md` 与 `commit.md` 已覆盖严格复查修正链的关键提交
- [x] `test.md` 至少记录 4 条可执行验证命令或等价验证步骤
- [x] 本地 progress file / KB 已记录本轮审计结论

## 4. 任务拆分与派发（面向子代理）

1. 任务：还原严格复查提交链
   - 目标：定位 R0002 之后所有控制面修正提交及其文件落点
   - 允许修改：无
   - 禁止修改：仓库任何 tracked 文件
   - 验证：`git log` / `git show --stat` 能列出完整提交链与改动范围

2. 任务：补建 R0003 交接包与运行索引
   - 目标：新增 `round-0003-control-plane-hardening` 八件套，并更新 `MAIN.md` / `iterations/README.md`
   - 允许修改：`MAIN.md`、`docs/codex-potter/iterations/README.md`、`docs/codex-potter/iterations/round-0003-control-plane-hardening/*`
   - 禁止修改：GPU 代码、历史 round 的目标与结论
   - 验证：八件套齐全、索引可跳转、内容能解释提交链

3. 任务：补做严格复查与知识捕获
   - 目标：重跑入口/链接/结构校验，并把结果写回 `.codexpotter`
   - 允许修改：`.codexpotter/projects/2026/04/16/1/MAIN.md`、`.codexpotter/kb/*`
   - 验证：active docs 扫描、Markdown 链接校验、八件套计数、git history 核对通过

## 5. 风险清单（Risks）与应对

- 风险：继续把严格复查修正视为“只需写到 progress file 的隐性步骤”
  - 影响：git 内控制面历史与实际提交链继续分叉
  - 触发信号：`MAIN.md` / `iterations/README.md` 长期停在 R0002，但 git log 已有更多控制面提交
  - 应对：用一个回填型 R0003 统一收口，而不是继续追加隐性修正

- 风险：为追求“历史干净”而重写 R0002
  - 影响：历史结论被篡改，审计链变差
  - 触发信号：试图把 `78ab839` 之后的提交重新塞回 R0002
  - 应对：保留 R0002 原样，在 R0003 说明“为什么插入这一轮”

## 6. 验证计划（Test Plan）

最小验证（必须）：

- `git log --oneline -- MAIN.md README.md docs/codex-potter | head -n 12`
- `rg -n -- "--rounds <N>|--rounds 10" README.md MAIN.md docs/codex-potter/README.md docs/codex-potter/governance docs/codex-potter/iterations .codexpotter/projects/2026/04/16/1/MAIN.md`
- `python3` 脚本校验 `README.md`、`MAIN.md` 与 `docs/codex-potter/**/*.md` 的本地链接
- `python3` 脚本校验 `round-0001` / `round-0002` / `round-0003` 的八件套完整性

扩展验证（有时间就做）：

- 重新扫描活跃入口的绝对路径与旧术语
- 核对 progress file 中的 `git_commit` 与最新仓库基线是否对齐

## 7. 提交计划（Git Plan）

本轮预计提交策略：

- 分支：`main`
- 提交拆分：先提交 R0003 交接包与索引，再补一次 metadata/验证回填
- 提交信息约定：`docs: add round-0003 control plane hardening` / `docs: record round-0003 commit metadata`
