---
round:
  id: "round-0006-control-plane-readme-backlink"
  date: "2026-04-17"
  status: "done"
repo:
  branch: "main"
  base_commit: "d559eb003997175847cb7a66bf295b17f7ba6e11"
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

补齐 `docs/codex-potter/README.md` 缺失的统一“上层入口”区块，并把这次最小 docs-only 收尾正式登记为 R0006。

## 1. 强制阅读清单（开始前必须完成）

- [x] `MAIN.md`：确认主入口互链约定与当前运行索引已推进到 R0005
- [x] `docs/codex-potter/README.md`：确认该文件是唯一未采用统一“上层入口”区块的控制面文档
- [x] `docs/codex-potter/iterations/README.md`：确认轮次快速入口当前停留到 R0005
- [x] `docs/codex-potter/governance/workflow-protocol.md`：确认其他主入口页已经在标题下保留入口块
- [x] `docs/codex-potter/iterations/round-0005-control-plane-backlinks/{summary,test,close}.md`：确认 R0005 已把“所有 control-plane 页面应保留入口块”写成明确目标

阅读结论（要点）：

- 经过 `rg --files-without-match "上层入口：" docs/codex-potter -g '*.md'` 复查，`docs/codex-potter/README.md` 是唯一仍缺少统一入口块的控制面文档。
- 该缺口不是功能性 bug，但会让“所有 control-plane 页面统一回到主入口”的规则保留一个活跃例外；继续放着不管，会削弱后续入口审计的可信度。
- 由于修复范围极小，但仍会产生新的 git 提交，因此需要登记一个最小 R0006，避免 round 历史与提交链再次脱节。

## 2. 范围（In Scope）与非目标（Out of Scope）

In Scope（本轮要做）：

- 给 `docs/codex-potter/README.md` 增加统一“上层入口”区块
- 更新 `MAIN.md` 与 `docs/codex-potter/iterations/README.md`，把本轮登记为 R0006
- 新建 `docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/` 八件套

Out of Scope（本轮明确不做）：

- 不新增 benchmark / profiling / scorecard 产物
- 不再扩张控制面规则或模板范围
- 不回改历史 round 的结论或建议

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
- 文档改动：`docs/codex-potter/README.md`、`MAIN.md`、`docs/codex-potter/iterations/README.md`

完成定义（DoD），需可验收：

- [x] `docs/codex-potter/README.md` 已采用统一“上层入口”区块
- [x] `rg --files-without-match "上层入口：" docs/codex-potter -g '*.md'` 返回空结果
- [x] `MAIN.md` 与 `iterations/README.md` 已收录 R0006
- [x] `code.md` 记录了实际变更落点与计划偏差
- [x] `test.md` 至少记录 1 条可执行验证命令，并覆盖入口块、链接与结构检查
- [x] `commit.md` 记录分支名与主体提交 hash（至少一个）
- [x] `summary.md` 记录分支名与主体提交 hash（至少一个）
- [x] `close.md` 指明下一轮入口与退出判断

## 4. 任务拆分与派发（面向子代理）

本轮未启用子代理，主会话直接完成最小 docs-only 收口。

1. 任务：复核剩余入口块缺口
   - 目标：确认当前只剩 `docs/codex-potter/README.md` 未采用统一入口块
   - 允许修改：无（只读扫描）
   - 验证：`rg --files-without-match "上层入口：" docs/codex-potter -g '*.md'`

2. 任务：补入口块并登记 R0006
   - 目标：修正 `docs/codex-potter/README.md`，同步更新 `MAIN.md`、`iterations/README.md` 和新 round 目录
   - 允许修改：`docs/codex-potter/README.md`、`MAIN.md`、`docs/codex-potter/iterations/README.md`、`docs/codex-potter/iterations/round-0006-control-plane-readme-backlink/*`
   - 验证：diff 无格式问题，索引与快速入口可命中 R0006

3. 任务：复跑验证并记录结果
   - 目标：确认入口块、链接与八件套完整性校验全部通过
   - 允许修改：`test.md`、`summary.md`、`commit.md`、`close.md`
   - 验证：校验命令记录为 PASS，结果可复现

## 5. 风险清单（Risks）与应对

- 风险：对 `docs/codex-potter/README.md` 补块时写错相对路径
  - 影响：把一个导航修复引成新的断链
  - 触发信号：本地 Markdown 链接校验报错
  - 应对：复跑链接校验，只允许使用仓库相对路径

- 风险：只修 README 而不登记 round
  - 影响：git 历史新增提交，但 round 历史仍停在 R0005
  - 触发信号：`MAIN.md` / `iterations/README.md` 与 `git log` 断层
  - 应对：本轮最小范围内仍新增 R0006 八件套与索引

## 6. 验证计划（Test Plan）

最小验证（必须）：

- `git diff --check -- MAIN.md docs/codex-potter`
- `rg --files-without-match "上层入口：" docs/codex-potter -g '*.md'`
- `python3` 脚本校验 `README.md`、`MAIN.md` 与 `docs/codex-potter/**/*.md` 的本地 Markdown 链接
- `python3` 脚本校验 `round-0001` 到 `round-0006` 的八件套是否齐全
- `rg -n "R0006|round-0006-control-plane-readme-backlink|控制面总说明入口块补齐" MAIN.md docs/codex-potter/iterations/README.md`

## 7. 提交计划（Git Plan）

本轮预计提交策略：

- 分支：`main`
- 提交拆分：先提交 README 入口块补丁、R0006 目录与索引，再回填主体提交 hash
- 提交信息约定：`docs: add round-0006 control plane readme backlink`、`docs: record round-0006 commit metadata`
