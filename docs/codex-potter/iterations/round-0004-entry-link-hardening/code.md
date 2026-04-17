# 本轮编码 / 执行记录（Code / Execution Log）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 0. 本轮实际执行摘要

- 补齐 `workflow-protocol.md` 对 `iterations/README.md` 的回链。
- 补齐 `iterations/README.md` 对 `docs/codex-potter/README.md` 与 `workflow-protocol.md` 的回链，并把快速入口扩展到 R0004。
- 更新 `MAIN.md` 运行索引，新增 R0004。
- 新建 `round-0004-entry-link-hardening` 八件套，并把本次 docs-only 严格复查写成可续跑记录。

## 1. 实际改动落点

- `MAIN.md`
  - 新增 R0004 行，正式把本轮纳入仓库级运行索引。
  - 收敛主入口互链段的总结句，从依赖数量的“三者必须保持互链”改为稳态表述“上述入口必须保持互链”。

- `docs/codex-potter/governance/workflow-protocol.md`
  - 在“上层入口”补充轮次索引链接，避免协议页成为单向入口。

- `docs/codex-potter/iterations/README.md`
  - 新增“上层入口”区块，补回对控制面总说明与工作流协议的回链。
  - 在快速入口中新增 R0004。

- `docs/codex-potter/iterations/round-0004-entry-link-hardening/*`
  - 记录本轮目标、验证、提交与交接。

- `.codexpotter/projects/2026/04/16/1/MAIN.md`
  - 本地同步本轮任务状态与 Done 结论。

- `.codexpotter/kb/*`
  - 写回活跃入口互链缺口与代码位置，供后续严格复查直接复用。

## 2. 与计划的偏离（Plan Drift）

- 与原计划基本一致。
- 唯一偏离点：`round-0003` 的下一轮建议原本主张直接进入 benchmark / profiling；本轮仍然插入 docs-only 收口。
  - 原因：本次发现的是活跃入口缺链，会直接影响续跑导航，优先级高于继续扩展业务工作。
  - 控制：只做最小互链修正，不新增新的制度层功能。

## 3. 关键决策

- 决策：只修活跃入口，不追溯清理所有历史文档
  - 原因：本轮问题是“当前入口不能完整互跳”，不是全仓历史文案重写。

- 决策：新增 R0004，而不是把修改回写到 R0003
  - 原因：这次严格复查是在 R0003 关闭之后发现的新缺口；继续回改历史会再次模糊 round 边界。

## 4. 阶段内自检

- 已在独立 worktree `control-plane-entry-links` 中完成文档修改，避免直接在 `main` 上扩散改动。
- 基线校验确认 worktree 初始无格式错误，4 个主入口文档的本地 Markdown 链接可解析。
- 详细验证结果见 `test.md`。
