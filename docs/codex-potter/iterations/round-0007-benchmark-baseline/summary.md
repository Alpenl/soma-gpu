---
round:
  id: "round-0007-benchmark-baseline"
  date: "2026-04-17"
  status: "done"
repo:
  branch: "main"
  commits:
    - "pending-round-close"
artifacts:
  docs:
    - "round-overview.md"
    - "plan.md"
    - "code.md"
    - "test.md"
    - "next-round-suggestions.md"
    - "summary.md"
    - "commit.md"
    - "close.md"
    - "results/scorecard.md"
  code:
    - "benchmark_stageii_public.py"
    - "utils/stageii_benchmark.py"
    - "tests/test_stageii_benchmark.py"
    - "results/public-stageii-benchmark.json"
---

# 本轮总结与交接（Summary & Handoff）

上层入口：

- 仓库级入口：[MAIN.md](../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 0. 本轮结论（TL;DR）

- 已新增公开可复跑的 stageii baseline harness：`benchmark_stageii_public.py`，并生成 `results/public-stageii-benchmark.json`。
- 第一版 scorecard 已落地：public workload 基于 `support_data/tests/mosh_stageii.pkl`，速度 `2.8471 ms`、repeatability `0.0`、核心数值稳定，但 mesh/mp4 明确 blocked。
- `.worktrees/gpu-stageii-foundation` 的 foundation 关键测试通过（`19 passed, 1 warning`），但由于 worktree 仍是 dirty tree，本轮结论是“committed foundation 部分采纳，dirty sequence/render 扩展暂不采纳”。
- 下一轮最推荐动作不是直接进热点优化，而是先把 committed foundation 落到干净 candidate 分支，并在同一 scorecard 下扩 workload。

---

## 1. 产出清单（Deliverables）

文档：

- `round-overview.md`：定义 round-0007 为什么先做 public benchmark skeleton，而不是伪装 full E2E
- `plan.md`：记录 public workload、blocked stages 与 candidate 结论的任务拆分
- `code.md`：记录 harness、scorecard、索引和 worktree 决策
- `test.md`：记录测试、结果 JSON 和 worktree foundation 验证
- `next-round-suggestions.md`：提出 clean candidate、mesh/mp4 扩展和 profiling 三条下一轮建议
- `summary.md`：本文件
- `commit.md`：记录提交策略
- `close.md`：记录索引更新与下一轮起点
- `results/scorecard.md`：第一版 public scorecard

代码/配置/其他：

- `benchmark_stageii_public.py`：public benchmark CLI
- `utils/stageii_benchmark.py`：兼容加载、计时、repeatability、blocked-stage 汇总
- `tests/test_stageii_benchmark.py`：benchmark harness 测试
- `results/public-stageii-benchmark.json`：实际测量结果
- `MAIN.md`：新增 R0007
- `docs/codex-potter/iterations/README.md`：新增 R0007 快速入口
- `.codexpotter/projects/2026/04/17/1/MAIN.md`、`.codexpotter/kb/*`：本地状态与上下文同步（不进 git）

---

## 2. 验证结论（Verification）

- 最小验证：PASS
- 关键说明：
  - public benchmark 已可复跑并生成 JSON 报告
  - foundation candidate 的关键测试已通过
  - full mesh/mp4 public path 仍然 blocked，但阻塞条件已被明确记录，而不是继续隐含在口头环境里

---

## 3. 关键决策与取舍（Decisions）

- 决策：先以 public stageii-ingest 作为 round-0007 baseline，而不是追求 full mp4 baseline
  - 备选方案：继续尝试直接跑 mesh/mp4 路径
  - 取舍原因：当前机器缺少 README 历史环境、`psbody`、licensed `model.npz` 和 `blender`；强推 full E2E 只会得到不可复现文档

- 决策：为 public benchmark 新增独立兼容加载器，而不是依赖 `MoSh.load_as_amass_npz(...)`
  - 备选方案：修当前环境直到 `MoSh` 导入通过
  - 取舍原因：round-0007 的 P0 是 benchmark 口径，不是环境大修；独立加载器能更快稳定 public baseline

- 决策：`.worktrees/gpu-stageii-foundation` 只部分采纳
  - 备选方案：把当前 worktree 当作整体 candidate 或整体暂不采纳
  - 取舍原因：committed foundation 已有测试与边界，dirty 扩展没有；整体采纳风险过高，整体否决又会丢掉已验证基座

---

## 4. 风险与遗留（Risks & TODO）

风险：

- 当前 scorecard 仍停在 stageii-ingest，不能替代 full mesh/mp4 指标
- public harness 目前主要验证 legacy sample，modern stageii format 仍需补覆盖
- worktree 的 dirty 扩展如果继续增长，会持续拖慢 candidate 收敛

遗留 TODO（下一轮可直接接手）：

1. 将 committed foundation 拆到干净 candidate 分支，并对齐同一 scorecard
2. 在补齐 mesh/render 环境后，把 benchmark 扩到 mesh export / mp4
3. 在统一 workload 上做第一次 profiling，而不是继续只做架构判断

---

## 5. 下一轮建议（Next Round Suggestions）

1. 先做 clean candidate，再谈 `gpu-stageii-foundation` 的 merge 或大规模评估。
2. 解锁 `psbody` / `model.npz` / `blender` 后，再扩展产物链 benchmark，不要把 blocked stages 长期留在文档里。
3. benchmark 面稳定后立即进入 profiling，回答真实热点在哪一层。

---

## 6. 子代理回执（Worker Reports）

### Worker: explorer attempts

- 做了什么：多次尝试对主仓库、round 文档和 worktree 做只读探索
- 改了哪些文件：无
- 怎么验证的：子代理通知显示多次上游 `502`；主会话改用本地审计与命令验证完成收敛
- 风险与疑点：子代理基础设施在本轮不稳定，导致“计划阶段必须新子代理”的执行质量下降
- 未完成项：若下轮子代理恢复稳定，建议重新把 candidate 审计与 profiling 计划外包给新代理

### Worker: worktree foundation verification

- 做了什么：对 `.worktrees/gpu-stageii-foundation` 的 foundation 关键测试子集做工程化验证
- 改了哪些文件：无
- 怎么验证的：`python3 -m pytest tests/test_smplx_torch_wrapper.py tests/test_transformed_lm_torch.py tests/test_gmm_prior_torch.py tests/test_stageii_backend.py tests/test_stageii_torch_smoke.py -q`
- 风险与疑点：worktree 仍是 dirty tree；通过的只是 foundation 子集，不代表 sequence/render 扩展已被统一 benchmark gate 覆盖
- 未完成项：需要 clean candidate 分支承接 committed foundation

---

## 7. 交接给下一轮：强制阅读清单（Handoff Reading List）

- `MAIN.md`
- `docs/codex-potter/governance/metrics-framework.md`
- `docs/codex-potter/governance/workflow-protocol.md`
- `docs/codex-potter/governance/resume-and-handoff.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/round-overview.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/plan.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/code.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/test.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/results/scorecard.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/summary.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/next-round-suggestions.md`
- `benchmark_stageii_public.py`
- `utils/stageii_benchmark.py`
