---
round:
  id: "round-0001-init"
  date: "2026-04-16"
  status: "done"
repo:
  branch: "main"
  base_commit: "35529ead68eaa4358ec95bf6471027448d95467c"
roles:
  orchestrator: "main-session"
  workers:
    - "Hegel"
    - "Noether"
    - "Descartes"
    - "Kierkegaard"
    - "Nietzsche"
scope_tags:
  - "docs"
  - "control-plane"
  - "gpu-stageii"
---

# 本轮计划（Plan）

## 0. 本轮一句话目标

建立本仓库的 CodexPotter 控制面入口、治理规则、轮次规范与首轮交接包，使下一轮可以按文档闭环继续推进 GPU 线路优化。

## 1. 强制阅读清单（开始前必须完成）

- [x] `MAIN.md`：本轮为初始化轮，主控入口由本轮建立。
- [x] 上一轮 `summary.md`：无，上轮不存在；本轮承担“零到一初始化”职责。
- [x] 上一轮 `test.md`：无，上轮不存在。
- [x] 上一轮 `plan.md`：无，上轮不存在。
- [x] `docs/codex-potter/governance/resume-and-handoff.md`：本轮需要建立续跑与交接制度。
- [x] `docs/如何将原生MoSh++改造成GPU版.md`：确认长期目标是优先推进 GPU StageII。
- [x] `.worktrees/gpu-stageii-foundation/docs/superpowers/plans/2026-04-15-gpu-stageii-foundation.md`：确认当前已有 torch stageii 候选实现资产。

阅读结论（要点）：

- 本仓库长期目标是优化 GPU 路线，而不是泛化的控制面平台建设。
- 当前最有价值的候选资产是 `.worktrees/gpu-stageii-foundation`，后续应按统一指标口径评估。
- 本轮必须先把“如何迭代”写清楚，再进入真实 bench、profiling 和代码更改。

## 2. 范围（In Scope）与非目标（Out of Scope）

In Scope（本轮要做）：

- 建立 `MAIN.md` 与 `docs/codex-potter/` 文档体系
- 固化治理文档：协议、指标、续跑规范
- 固化模板：`plan/test/summary`
- 建立 `iterations/round-0001-init/` 并补齐最小交接包

Out of Scope（本轮明确不做）：

- 不修改 GPU 核心实现或测试
- 不建立完整 benchmark harness
- 不对 `.worktrees/gpu-stageii-foundation` 做技术采纳结论

约束（必须遵守）：

- 主会话只做调度与轻审阅
- 每个阶段使用新的子代理
- 文档必须中文、结构化、可续跑
- 轮次目录命名与交接包口径必须统一

## 3. 产物与完成定义（Deliverables & DoD）

本轮必须产物：

- `MAIN.md`
- `docs/codex-potter/README.md`
- `docs/codex-potter/governance/*.md`
- `docs/codex-potter/specs/2026-04-16-control-plane-design.md`
- `docs/codex-potter/plans/2026-04-16-init-control-plane-plan.md`
- `docs/codex-potter/templates/*.md`
- `docs/codex-potter/iterations/README.md`
- `docs/codex-potter/iterations/round-0001-init/*`

完成定义（DoD），需可验收：

- [x] 仓库存在根级 `MAIN.md`
- [x] 治理、模板、spec、plan、iterations 文档齐备
- [x] `round-0001-init` 包含 `round-overview.md`、`plan.md`、`test.md`、`summary.md`、`next-round-suggestions.md`
- [x] `test.md` 至少记录 1 条可执行验证命令
- [x] 轮次路径口径统一为 `docs/codex-potter/iterations/round-XXXX-<slug>/`

## 4. 任务拆分与派发（面向子代理）

1. 任务：主入口与协议
   - 目标：建立 `MAIN.md`、控制面 README、工作流协议
   - 允许修改：`MAIN.md`、`docs/codex-potter/README.md`、`docs/codex-potter/governance/workflow-protocol.md`
   - 禁止修改：GPU 代码与测试目录
   - 验证：检查互链与职责边界是否明确

2. 任务：指标与首轮规范
   - 目标：建立指标框架、iterations 规范、Round 0001 概览与下一轮建议
   - 允许修改：`docs/codex-potter/governance/metrics-framework.md`、`docs/codex-potter/iterations/**`
   - 验证：确认四大指标、Gate A/B、candidate asset 评估口径齐全

3. 任务：续跑规范与模板
   - 目标：建立 resume/handoff 规范与模板三件套
   - 允许修改：`docs/codex-potter/governance/resume-and-handoff.md`、`docs/codex-potter/templates/**`
   - 验证：确认下一轮阅读清单、汇报格式、模板字段完整

4. 任务：spec 与实施计划
   - 目标：固化控制面设计 spec 和本轮计划
   - 允许修改：`docs/codex-potter/specs/**`、`docs/codex-potter/plans/**`
   - 验证：确认 spec 不越界到 GPU 实现，计划与真实初始化目标一致

## 5. 风险清单（Risks）与应对

- 风险：轮次目录命名出现多套口径
  - 影响：下一轮无法确定标准入口
  - 触发信号：`runs/rollouts/iterations` 并存
  - 应对：统一收敛到 `iterations/round-XXXX-<slug>/`

- 风险：计划文档偏离真实仓库目标
  - 影响：后续子代理会被错误引导到无关工作
  - 触发信号：出现泛化“控制面服务组件/部署接口”类内容
  - 应对：把计划重写为“文档初始化计划”，只服务本仓库 GPU 迭代控制面

## 6. 验证计划（Test Plan）

最小验证（必须）：

- `git status --short`
- `find docs/codex-potter -type f | sort`
- `rg -n 'rollouts/|runs/' MAIN.md docs/codex-potter`

扩展验证（有时间就做）：

- 抽查关键文档内容与互链
- 核对 `round-0001-init/summary.md` 是否可作为下一轮阅读入口

## 7. 提交计划（Git Plan）

本轮预计提交策略：

- 分支：`main`
- 提交拆分：单次文档初始化提交即可
- 提交信息约定：`docs: initialize codex potter control plane`
