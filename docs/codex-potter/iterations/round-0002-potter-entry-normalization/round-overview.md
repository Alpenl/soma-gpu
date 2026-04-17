# Round 0002 (potter-entry-normalization) 概览

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-16

本轮目标是把 **人类控制面入口** 与 **CodexPotter runtime 入口** 明确区分并同步，避免后续 `resume` 时把仓库根 `MAIN.md` 误当成 progress file。

## 1. 本轮目标（Goals）

- 确认 CodexPotter 的 `resume PROJECT_PATH` 解析规则与工作目录推导规则。
- 固化默认续跑示例入口：`codex-potter resume 2026/04/16/1 --yolo --rounds 10`；若需不同轮数，只替换最后的 `10`。
- 同步本地 `.codexpotter/projects/2026/04/16/1/MAIN.md` 到当前基线提交与当前任务状态。
- 把上述规则写入仓库根入口与续跑规范，避免再次混淆。

## 2. 本轮范围（In Scope）

- 源码/官方文档级别的规则核对
- 文档更新：
  - `MAIN.md`
  - `docs/codex-potter/README.md`
  - `docs/codex-potter/governance/resume-and-handoff.md`
- 本地 runtime progress file 同步：
  - `.codexpotter/projects/2026/04/16/1/MAIN.md`
- 新建本轮交接包：
  - `round-overview.md`
  - `plan.md`
  - `code.md`
  - `test.md`
  - `next-round-suggestions.md`
  - `summary.md`
  - `commit.md`
  - `close.md`

## 3. 不在本轮范围（Out of Scope）

- 不升级 CodexPotter 版本
- 不做 benchmark、profiling 或 GPU 核心实现修改
- 不修改 `potter-rollout.jsonl`

## 4. 本轮输出物（Deliverables）

P0（必须产出）：

- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/round-overview.md`
- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/plan.md`
- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/code.md`
- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/test.md`
- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/summary.md`
- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/next-round-suggestions.md`
- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/commit.md`
- `docs/codex-potter/iterations/round-0002-potter-entry-normalization/close.md`
- 更新后的 `MAIN.md`
- 更新后的 `docs/codex-potter/governance/resume-and-handoff.md`

P1（本地但不进 git）：

- 更新后的 `.codexpotter/projects/2026/04/16/1/MAIN.md`

## 5. Exit Criteria（本轮退出标准）

满足以下条件即可结束本轮：

- 已用源码或官方文档确认：仓库根 `MAIN.md` 不能直接作为 `resume` runtime 入口。
- 已在仓库文档中写清标准 `resume` 命令与当前 progress file 路径。
- `.codexpotter/projects/2026/04/16/1/MAIN.md` 已同步到当前基线提交，并指向仓库根控制面入口。
- 本轮验证记录已能支撑下一轮直接使用正确的续跑命令。
