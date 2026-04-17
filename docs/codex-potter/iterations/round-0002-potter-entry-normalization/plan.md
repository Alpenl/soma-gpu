---
round:
  id: "round-0002-potter-entry-normalization"
  date: "2026-04-16"
  status: "done"
repo:
  branch: "main"
  base_commit: "462f95d457e6eafcc963e613eba7b08e15241c93"
  head_commit: "4fb8c2af7449d6850cfaeb45839287b1524a2408"
roles:
  orchestrator: "main-session"
  workers:
    - "Aristotle (errored)"
    - "Bohr (errored)"
scope_tags:
  - "docs"
  - "codex-potter"
  - "resume"
---

# 本轮计划（Plan）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 0. 本轮一句话目标

把 CodexPotter 的 runtime progress file 与仓库根控制面入口归一化，明确标准 `resume` 命令，并同步本地内部项目状态到当前基线。

## 1. 强制阅读清单（开始前必须完成）

- [x] `MAIN.md`：确认当前仓库级控制面入口与轮次索引
- [x] `docs/codex-potter/iterations/round-0001-init/summary.md`：确认初始化轮已经完成，下一步应进入真实续跑准备
- [x] `docs/codex-potter/iterations/round-0001-init/next-round-suggestions.md`：确认长期重点仍是 benchmark/profiling/gpu-stageii-foundation 评估
- [x] `docs/codex-potter/governance/resume-and-handoff.md`：确认续跑制度需要写清入口
- [x] `/tmp/CodexPotter-src/docs/wiki/cli.md`：确认 `resume PROJECT_PATH` 的 candidate 解析算法
- [x] `/tmp/CodexPotter-src/cli/src/workflow/resume.rs`：确认 `derive_project_workdir` 要求 progress file 位于 `.codexpotter`
- [x] `.codexpotter/projects/2026/04/16/1/MAIN.md`：确认当前内部 progress file 仍停留在初始化前状态

阅读结论（要点）：

- `resume` 的解析规则允许命中仓库根 `MAIN.md`，但随后会因为该文件不位于 `.codexpotter` 内而无法推导 workdir。
- 因此，仓库根 `MAIN.md` 只能作为人类入口；真正给 Potter runtime 用的，必须是 `.codexpotter/projects/.../MAIN.md`。
- 当前内部 progress file 仍记录旧的基线提交 `35529ea...`，需要同步到当前提交并指向新的仓库文档体系。

## 2. 范围（In Scope）与非目标（Out of Scope）

In Scope（本轮要做）：

- 通过 CodexPotter 源码与官方文档确认 `resume` 入口规则
- 在仓库文档中固化当前标准续跑命令与 runtime progress file 路径
- 更新 `.codexpotter/projects/2026/04/16/1/MAIN.md`
- 建立 `round-0002-potter-entry-normalization` 交接包

Out of Scope（本轮明确不做）：

- 不改动 GPU 路线实现代码
- 不升级 CodexPotter 到新版本
- 不修改 `.codexpotter/projects/2026/04/16/1/potter-rollout.jsonl`

约束（必须遵守）：

- 仓库内可提交变更只限文档
- `.codexpotter` 变更仅作本地 runtime 同步，不进 git
- 不把仓库根 `MAIN.md` 伪装成 runtime progress file

## 3. 产物与完成定义（Deliverables & DoD）

本轮必须产物：

- `MAIN.md` 中新增标准 `resume` 入口说明
- `docs/codex-potter/README.md` 与 `resume-and-handoff.md` 中新增入口区分说明
- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/` 八件套
- 更新后的 `.codexpotter/projects/2026/04/16/1/MAIN.md`

完成定义（DoD），需可验收：

- [x] 能用源码或官方文档定位 `resume` 的 candidate 解析规则
- [x] 能用源码定位 `derive_project_workdir` 对 `.codexpotter` 的要求
- [x] 仓库文档里出现当前标准续跑命令
- [x] 内部 progress file 已同步到提交 `462f95d457e6eafcc963e613eba7b08e15241c93`
- [x] 本轮 `test.md` 至少记录 4 条验证命令或等价验证步骤
- [x] 本轮目录已补齐 `code.md`、`commit.md`、`close.md`

## 4. 任务拆分与派发（面向子代理）

1. 任务：只读探测 `resume` 规则
   - 目标：确认 `PROJECT_PATH` 解析与 workdir 推导规则
   - 允许修改：无
   - 验证：源码路径与关键实现定位
   - 说明：本轮尝试了两个只读探测子代理，但均因上游 502 失败，后续由主会话本地完成

2. 任务：仓库入口文档更新
   - 目标：把人类入口与 runtime 入口区分写入仓库文档
   - 允许修改：`MAIN.md`、`docs/codex-potter/README.md`、`docs/codex-potter/governance/resume-and-handoff.md`
   - 验证：关键路径与标准 `resume` 命令可见

3. 任务：本地 runtime progress file 同步
   - 目标：更新 `.codexpotter/projects/2026/04/16/1/MAIN.md`
   - 允许修改：该单文件
   - 验证：front matter 与任务状态更新到当前基线

4. 任务：本轮交接包
   - 目标：写完 `overview/plan/code/test/next/summary/commit/close`
   - 允许修改：本轮目录下八个文件
   - 验证：下一轮可直接按阅读清单继续

## 5. 风险清单（Risks）与应对

- 风险：仍有人把仓库根 `MAIN.md` 直接传给 `resume`
  - 影响：续跑失败，误以为 Potter 未配置好
  - 触发信号：使用 `codex-potter resume /home/alpen/DEV/soma-gpu` 或 `.../MAIN.md`
  - 应对：在根入口、README 与 handoff 规范中显式写清

- 风险：内部 progress file 本地修改后未同步
  - 影响：`resume` 仍会按照旧初始化状态继续
  - 触发信号：front matter 仍指向 `35529ea...`
  - 应对：更新 `.codexpotter/projects/2026/04/16/1/MAIN.md` 到当前基线

## 6. 验证计划（Test Plan）

最小验证（必须）：

- `sed -n '95,140p' /tmp/CodexPotter-src/docs/wiki/cli.md`
- `sed -n '659,680p' /tmp/CodexPotter-src/cli/src/workflow/resume.rs`
- `sed -n '1,160p' .codexpotter/projects/2026/04/16/1/MAIN.md`
- `git diff --check -- MAIN.md docs/codex-potter`
- `git status --short`

扩展验证（有时间就做）：

- 尝试一次只读 `resume` 入口探测，确认 `2026/04/16/1` 可作为标准 short path

## 7. 提交计划（Git Plan）

本轮预计提交策略：

- 分支：`main`
- 提交拆分：单次文档归一化提交
- 提交信息约定：`docs: normalize codex potter resume entry`
