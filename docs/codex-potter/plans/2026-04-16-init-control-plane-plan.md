# 2026-04-16 初始化实施计划（CodexPotter 控制面）

上层入口：

- 仓库级入口：[MAIN.md](../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../iterations/README.md)

> 目的：把本仓库的调度式工作流固化为可续跑的中文文档体系。  
> 范围：只做控制面初始化与首轮交接包，不做 GPU 核心代码修改。

## 1. 初始化目标

本轮计划只服务一件事：让后续 GPU 优化轮次能够按统一文档协议启动、执行、验证、总结和续跑。

具体目标：

1. 建立仓库级主入口：`MAIN.md`
2. 建立控制面总说明与治理文档：
   - `docs/codex-potter/README.md`
   - `docs/codex-potter/governance/*.md`
3. 建立首轮迭代目录与最小交接包：
   - `docs/codex-potter/iterations/round-0001-init/`
4. 建立模板与续跑规则：
   - `docs/codex-potter/templates/*.md`
5. 明确 `.worktrees/gpu-stageii-foundation` 在后续轮次中的评估口径。

## 2. 本轮范围

### 2.1 In Scope

- 控制面入口与互链规范
- 轮次目录规范与模板
- 指标框架与建议/决断点规范
- `round-0001-init` 的 `overview/plan/code/test/next/summary/commit/close` 文档
- 文档层面的轻量验证

### 2.2 Out of Scope

- 不改动 `moshpp/`、`src/`、`tests/` 等 GPU 相关实现文件
- 不建立真实 benchmark harness
- 不对 `.worktrees/gpu-stageii-foundation` 做技术结论，只定义其评估入口

## 3. 计划创建的文件树

```text
MAIN.md
docs/codex-potter/
  README.md
  governance/
    workflow-protocol.md
    metrics-framework.md
    resume-and-handoff.md
  specs/
    2026-04-16-control-plane-design.md
  plans/
    2026-04-16-init-control-plane-plan.md
  templates/
    README.md
    plan-template.md
    code-template.md
    test-template.md
    commit-template.md
    close-template.md
    summary-template.md
  iterations/
    README.md
    round-0001-init/
      round-overview.md
      plan.md
      code.md
      test.md
      next-round-suggestions.md
      summary.md
      commit.md
      close.md
```

## 4. 本轮执行步骤

### 步骤 1：固化设计输入

- 以 `docs/codex-potter/specs/2026-04-16-control-plane-design.md` 作为控制面设计约束源。
- 明确本轮是文档初始化轮，不做 GPU 核心开发。

### 步骤 2：并行起草治理文档

- 主会话拆分 ownership 并派发新的子代理。
- 子代理并行起草主入口、总说明、协议、指标、模板、首轮规范。

### 步骤 3：整合并规范口径

- 统一轮次目录命名为 `docs/codex-potter/iterations/round-XXXX-<slug>/`
- 统一最小交接包为：
  - `round-overview.md`
  - `plan.md`
  - `code.md`
  - `test.md`
  - `next-round-suggestions.md`
  - `summary.md`
  - `commit.md`
  - `close.md`

### 步骤 4：补齐首轮交接包

- 为 `round-0001-init` 补全实际 `plan/code/test/next/summary/commit/close`
- 把本轮真实执行信息、验证命令、结论与遗留写入交接包

### 步骤 5：轻量验证

- 确认关键文件存在
- 确认互链路径不再混用 `runs/rollouts/iterations`
- 确认 `round-0001-init` 可作为下一轮的阅读入口

## 5. 验证计划

最小验证：

1. `git status --short`：确认改动边界仅为文档
2. `find docs/codex-potter -type f | sort`：确认文件树齐备
3. `rg -n 'rollouts/|runs/' MAIN.md docs/codex-potter`：确认轮次路径口径已统一

扩展验证：

- 抽查关键文档内容与互链
- 检查 `round-0001-init/summary.md` 能否独立作为下一轮入口

## 6. 完成标准

满足以下条件即可视为初始化完成：

1. 仓库根目录存在 `MAIN.md`
2. `docs/codex-potter/` 下已形成治理、模板、spec、plan、iterations 四类文档
3. `round-0001-init` 目录具备最小交接包
4. 轻量验证完成且无明显结构冲突
5. 下一轮可以从 `MAIN.md -> iterations/README.md -> round-0001-init/close.md` 这条路径开始续跑

## 7. 已知风险与下一步

- 目前只有文档与规则，没有真实 benchmark 数据，因此后续 round 需要先补基线评测链路。
- `.worktrees/gpu-stageii-foundation` 已被纳入评估口径，但尚未进入统一 scorecard 比较。
- 下一轮建议优先做“基线 bench + profiling + candidate asset 评估”中的一项。
