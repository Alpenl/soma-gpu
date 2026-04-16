# soma-gpu 控制面主控文档（MAIN）

> 这是仓库级“控制面”入口。它不承担技术细节的长篇叙述，主要负责：目标对齐、工作流协议、每轮运行索引、以及把主入口与子文档互链起来。

## 0. 快速入口

- 总说明（本项目的控制面总览）：[docs/codex-potter/README.md](./docs/codex-potter/README.md)
- 工作流协议（每轮固定阶段与子代理约束）：[docs/codex-potter/governance/workflow-protocol.md](./docs/codex-potter/governance/workflow-protocol.md)
- 轮次规范与当前轮目录：[docs/codex-potter/iterations/README.md](./docs/codex-potter/iterations/README.md)
- CodexPotter 标准续跑命令：`codex-potter resume 2026/04/16/1 --yolo --rounds 10`
- CodexPotter runtime progress file（本地 gitignored）：`.codexpotter/projects/2026/04/16/1/MAIN.md`

## 1. 项目目标（GPU 线路）

本仓库的 GPU 线路目标明确聚焦四件事：

1. **更快**：关键路径上尽可能把瓶颈迁移到 GPU，减少 Python/CPU 端开销，支持批处理与并行。
2. **更准**：与原始参考实现（或权威基线）在数值一致性/误差边界上可解释、可回归、可监控。
3. **效果更好**：不仅追求“能跑”，更追求稳定性、鲁棒性、边界条件处理与可复现性带来的整体效果提升。
4. **更工程化**：可测试、可持续迭代、可追踪（日志/指标/产物）、可审阅（变更小步、文档可续跑）。

## 2. 主会话职责边界（只做调度与轻审阅）

主会话（你现在正在看的这个对话线程）必须刻意保持“控制面”属性：

- 只做：任务拆解、子代理派发、阶段门禁、轻审阅（风险点与接口边界）、合并与收尾。
- 不做：长时间沉浸式编码、单点深挖、把所有实现细节都塞进主会话。
- 产物导向：每轮必须落地为可执行的计划、可验证的改动、可回放的文档记录。

## 3. 每轮必做约束（可续跑）

每一轮（Round）开始时，必须先完成以下动作：

1. **先读上一轮文档**：上一轮的 `总结/决策/未决项/下一轮建议` 是本轮输入，禁止跳过。
2. **按固定阶段推进**：计划 -> 编码 -> 测试 -> 下一轮建议 -> 文档总结 -> 提交 -> 结束。
3. **每阶段必须新子代理**：阶段内可多名子代理并行，但不得复用前一阶段的同名代理来“偷跑”阶段门禁。
4. **尽量并行**：能拆就拆，能并行就并行；主会话只做排程与收敛。

详细规则见：[docs/codex-potter/governance/workflow-protocol.md](./docs/codex-potter/governance/workflow-protocol.md)。

## 4. 仓库现状与约束（截至 2026-04-16）

- `main` 工作区当前无未提交修改（干净），但相对 `origin/main` 处于 `ahead` 状态（本地领先若干提交）。
- 存在 worktree：`.worktrees/gpu-stageii-foundation/`，聚焦 torch 的 stageii（该 worktree 的职责与协作方式见工作流协议）。

> 约定：控制面文档不强制要求你必须在 `main` 上开发；鼓励用 worktree 承载中大型改动，减少上下文与冲突成本。

## 5. 每轮运行索引（手动维护，支持续跑）

以下索引用于把“每轮的输入/输出”串起来。后续每轮请更新本表，保持可追踪。

| 轮次 | 日期 | 主题 | 上一轮指针 | 本轮产物指针 | 状态 |
|---|---|---|---|---|---|
| R0001 | 2026-04-16 | 控制面初始化 | 无 | `docs/codex-potter/iterations/round-0001-init/` | 完成 |
| R0002 | 2026-04-16 | Potter 入口归一化 | `round-0001-init` | `docs/codex-potter/iterations/round-0002-potter-entry-normalization/` | 完成 |

后续每轮的标准产物目录：`docs/codex-potter/iterations/round-XXXX-<slug>/`。

## 6. 主入口互链约定

- 仓库级入口：`MAIN.md`
- 控制面说明入口：`docs/codex-potter/README.md`
- 治理与协议入口：`docs/codex-potter/governance/workflow-protocol.md`
- 轮次入口：`docs/codex-potter/iterations/README.md`

三者必须保持互链；新增文档必须至少回链到上述其一，避免“孤儿文档”。

## 7. 人类入口与 Runtime 入口的区别

- 仓库根 `MAIN.md` 是**人类与主会话**使用的控制面入口，已经提交到 git。
- `.codexpotter/projects/.../MAIN.md` 是 **CodexPotter runtime progress file**，供 `codex-potter resume` 使用，默认不进 git。
- 对当前仓库，标准续跑命令是：`codex-potter resume 2026/04/16/1 --yolo --rounds <N>`。
- 不要把仓库根 `MAIN.md` 直接当成 `resume` 的 `PROJECT_PATH`；根据 CodexPotter 源码，runtime progress file 必须位于 `.codexpotter/projects/...` 目录下。
