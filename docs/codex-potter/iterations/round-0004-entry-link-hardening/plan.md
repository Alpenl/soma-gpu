---
round:
  id: "round-0004-entry-link-hardening"
  date: "2026-04-17"
  status: "done"
repo:
  branch: "control-plane-entry-links"
  base_commit: "216d47fb8d07025126fc6e2933e5e3332671d534"
  head_commit: "archived in: docs: consolidate control-plane hardening history"
roles:
  orchestrator: "main-session"
  workers:
    - "none (strict review executed locally)"
scope_tags:
  - "docs"
  - "control-plane"
  - "navigation"
  - "handoff"
---

# 本轮计划（Plan）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 0. 本轮一句话目标

补齐活跃控制面入口的互链网，并把这次最小 docs-only 严格复查正式登记为 R0004，避免控制面规则与实际导航再次分叉。

## 1. 强制阅读清单（开始前必须完成）

- [x] `MAIN.md`：确认仓库主入口已声明 4 个主入口需要保持互链。
- [x] `docs/codex-potter/README.md`：确认控制面总说明已同时链接 `MAIN.md`、协议页与轮次页。
- [x] `docs/codex-potter/governance/workflow-protocol.md`：确认活跃协议页仍缺少轮次页回链。
- [x] `docs/codex-potter/iterations/README.md`：确认轮次入口页仍缺少对控制面总说明与协议页的回链。
- [x] `docs/codex-potter/iterations/round-0003-control-plane-hardening/summary.md`：确认 R0003 已宣称初始化任务收口，但严格复查仍需优先修正会误导续跑的活跃入口缺口。
- [x] `.codexpotter/projects/2026/04/16/1/MAIN.md`：确认当前 runtime progress file 尚未把这次缺链问题登记为显式任务。

阅读结论（要点）：

- 缺口不在历史文档，而在当前真正会被续跑入口直接命中的活跃页面。
- 若只改两处入口文档却不新增 round，git 历史会再次出现“修了活口径，但 round 索引没记”的断层。
- 因此本轮的最小可审计方案是：补互链 + 新增 R0004 + 更新本地 runtime / KB。

## 2. 范围（In Scope）与非目标（Out of Scope）

In Scope（本轮要做）：

- 在 `docs/codex-potter/governance/workflow-protocol.md` 补 `iterations/README.md` 回链
- 在 `docs/codex-potter/iterations/README.md` 补 `docs/codex-potter/README.md` 与 `workflow-protocol.md` 回链
- 新建 `docs/codex-potter/iterations/round-0004-entry-link-hardening/` 八件套
- 更新 `MAIN.md` 与 `docs/codex-potter/iterations/README.md` 的 round 索引
- 在 `.codexpotter` 中回写审计结论

Out of Scope（本轮明确不做）：

- 不进入 benchmark baseline、profiling 与 GPU 核心实现
- 不继续扩张 docs-only 范围到非活跃历史文档
- 不改动 `docs/codex-potter/README.md` 现有结论

约束（必须遵守）：

- 只改控制面文档与本地 runtime 记录
- 描述必须以当前仓库文件与实时校验结果为准
- 下一轮建议必须重新回到 benchmark / profiling，而不是继续长期停留在 docs-only

## 3. 产物与完成定义（Deliverables & DoD）

本轮必须产物：

- `MAIN.md`
- `docs/codex-potter/governance/workflow-protocol.md`
- `docs/codex-potter/iterations/README.md`
- `docs/codex-potter/iterations/round-0004-entry-link-hardening/*`
- `.codexpotter/projects/2026/04/16/1/MAIN.md`
- `.codexpotter/kb/README.md`
- `.codexpotter/kb/project-audit.md`

完成定义（DoD），需可验收：

- [x] 4 个主入口之间不再存在当前严格复查发现的缺链
- [x] `MAIN.md` 与轮次 README 已收录 R0004
- [x] R0004 八件套齐全
- [x] `test.md` 至少记录 4 条可执行验证命令或等价验证步骤
- [x] 本地 progress file / KB 已记录本轮审计结论

## 4. 任务拆分与派发（面向子代理）

1. 任务：定位活跃入口缺链
   - 目标：确认缺口只存在于 `workflow-protocol.md` 与 `iterations/README.md`
   - 允许修改：无
   - 验证：`rg` 命中结果能证明互链不完整

2. 任务：补入口互链并登记 R0004
   - 目标：修正文档导航，新增 round 目录并更新索引
   - 允许修改：`MAIN.md`、`docs/codex-potter/governance/workflow-protocol.md`、`docs/codex-potter/iterations/README.md`、`docs/codex-potter/iterations/round-0004-entry-link-hardening/*`
   - 禁止修改：GPU 代码、历史 round 结论
   - 验证：入口文档互链与 round 快速入口可直接跳转

3. 任务：补做验证与知识捕获
   - 目标：把互链、链接和结构验证结果写回 `.codexpotter`
   - 允许修改：`.codexpotter/projects/2026/04/16/1/MAIN.md`、`.codexpotter/kb/*`
   - 验证：`git diff --check`、Markdown 链接校验、八件套计数、互链扫描通过

## 5. 风险清单（Risks）与应对

- 风险：把互链规则写在 `MAIN.md`，但入口页本身长期不执行
  - 影响：续跑者进入某一入口页后仍需靠记忆或回退寻找其它入口
  - 触发信号：`MAIN.md` 声明的主入口在互链扫描里无法彼此命中
  - 应对：只修活跃入口，不扩散到历史文档；并把修正登记为正式 round

- 风险：继续产生 docs-only 提交却不新增 round
  - 影响：git 历史与 round 历史再次分叉
  - 触发信号：运行索引停在 R0003，但控制面提交已经继续前进
  - 应对：用 R0004 做最小收口，之后立即回到 benchmark 路线

## 6. 验证计划（Test Plan）

最小验证（必须）：

- `rg -n 'MAIN\\.md|docs/codex-potter/README\\.md|workflow-protocol\\.md|iterations/README\\.md' MAIN.md docs/codex-potter/README.md docs/codex-potter/governance/workflow-protocol.md docs/codex-potter/iterations/README.md`
- `git diff --check -- MAIN.md docs/codex-potter/governance/workflow-protocol.md docs/codex-potter/iterations/README.md docs/codex-potter/iterations/round-0004-entry-link-hardening`
- `python3` 脚本校验 4 个主入口文档的本地 Markdown 链接
- `python3` 脚本校验 `round-0001` / `round-0002` / `round-0003` / `round-0004` 的八件套完整性

扩展验证（有时间就做）：

- 重新扫描活跃入口旧口径与绝对路径，确认本轮没有引入新分叉
- 核对 `MAIN.md` 与 `iterations/README.md` 的 R0004 索引是否一致

## 7. 提交计划（Git Plan）

本轮预计提交策略：

- 分支：`control-plane-entry-links`
- 提交拆分：先提交 R0004 文档主体与索引，再补一次 metadata/验证记录
- 提交信息约定：`docs: add round-0004 entry link hardening` / `docs: record round-0004 commit metadata`
