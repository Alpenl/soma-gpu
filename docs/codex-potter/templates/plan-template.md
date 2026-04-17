---
round:
  id: "YYYY-MM-DD-<topic>"
  date: "YYYY-MM-DD"
  status: "draft | in_progress | done"
repo:
  branch: "<branch>"
  base_commit: "<commit-hash>"
  head_commit: "<optional: commit-hash>"
roles:
  orchestrator: "<name or handle>"
  workers:
    - "<worker-1>"
    - "<worker-2>"
scope_tags:
  - "docs"
  - "gpu"
  - "stageii"
---

# 本轮计划（Plan）

上层入口：

- 仓库级入口：[MAIN.md](../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../iterations/README.md)

## 0. 本轮一句话目标

用一句话描述“本轮结束时你希望仓库变成什么样”。

示例：补齐 CodexPotter 的续跑规范与中文模板，保证下一轮可以按文档闭环继续推进。

---

## 1. 强制阅读清单（开始前必须完成）

> 把阅读结果写成要点，而不是只写“已读”。

- [ ] `MAIN.md`：读出当前目标/优先级/约束
- [ ] 上一轮 `summary.md`：读出遗留与下一轮建议
- [ ] 上一轮 `test.md`：读出验证缺口与复现方式
- [ ] 上一轮 `plan.md`：读出原计划与偏离点
- [ ] `docs/codex-potter/governance/resume-and-handoff.md`：读出本轮必须产出的交接要求
- [ ] （按需）`docs/如何将原生MoSh++改造成GPU版.md`：读出背景与瓶颈
- [ ] （按需）相关分支/工作树的计划文档：读出可复用资产与风险

阅读结论（要点）：

- TODO

---

## 2. 范围（In Scope）与非目标（Out of Scope）

In Scope（本轮要做）：

- TODO

Out of Scope（本轮明确不做）：

- TODO

约束（必须遵守）：

- TODO（例如“只允许改某些文件”“不改 API 行为”“不引入新依赖”等）

---

## 3. 产物与完成定义（Deliverables & DoD）

本轮必须产物：

- `round-overview.md`
- `plan.md`（本文件）
- `code.md`（本轮编码 / 执行记录）
- `test.md`（本轮测试记录）
- `next-round-suggestions.md`（本轮下一轮建议）
- `summary.md`（本轮总结与交接）
- `commit.md`（本轮提交记录）
- `close.md`（本轮结束记录）
- 代码/配置/文档改动（若有）

完成定义（DoD），需可验收：

- [ ] TODO
- [ ] TODO
- [ ] `code.md` 记录了实际落点与计划偏离
- [ ] `test.md` 里至少有 1 条可执行验证命令，或写清楚不可执行原因与替代验证
- [ ] `commit.md` 记录分支名与 commit hash（至少一个），或明确“本轮无提交”
- [ ] `summary.md` 记录分支名与 commit hash（至少一个）
- [ ] `close.md` 指明下一轮入口与退出判断

---

## 4. 任务拆分与派发（面向子代理）

> 每个任务必须有边界：允许改哪些文件/模块，以及“怎么验证”。

任务列表（建议按“可并行”拆分）：

1. 任务：<标题>
   - 目标：TODO
   - 允许修改：`<paths/modules>`
   - 禁止修改：`<paths/modules>`
   - 验证：`<command>`，期望：<expected>
   - 回执要求：按汇报模板提交（见 `resume-and-handoff.md` 的“汇报格式”）

2. 任务：<标题>
   - 目标：TODO
   - 允许修改：TODO
   - 验证：TODO

---

## 5. 风险清单（Risks）与应对

- 风险：TODO
  - 影响：TODO
  - 触发信号：TODO
  - 应对：TODO

---

## 6. 验证计划（Test Plan）

> 这里写“计划要怎么验证”；执行后的记录写到 `test.md`。

最小验证（必须）：

- TODO（例如单测、lint、脚本 smoke、文档自检）

扩展验证（有时间就做）：

- TODO

---

## 7. 提交计划（Git Plan）

本轮预计提交策略：

- 分支：`<branch>`
- 提交拆分：TODO（例如“先文档后代码”“按模块拆分提交”）
- 提交信息约定：TODO（例如 `docs: ...` / `feat: ...`）
