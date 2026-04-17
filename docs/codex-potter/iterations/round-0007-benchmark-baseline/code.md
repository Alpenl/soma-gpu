---
round:
  id: "round-0007-benchmark-baseline"
  date: "2026-04-17"
  status: "done"
repo:
  branch: "main"
execution:
  mode: "mixed"
  worktree: "main + .worktrees/gpu-stageii-foundation (read/test only)"
---

# 本轮编码 / 执行记录（Code / Execution Log）

上层入口：

- 仓库级入口：[MAIN.md](../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 1. 实际执行概览

- 本阶段目标：建立最小可复现 benchmark 与第一版 scorecard，并完成 `.worktrees/gpu-stageii-foundation` 的采纳边界判断。
- 实际完成：新增 public stageii benchmark harness、生成 JSON 报告、补齐 round-0007 八件套与 scorecard、更新轮次索引，并给出 worktree 的部分采纳结论。
- 当前状态：round-0007 可关闭；下一轮应在更完整环境下扩展 benchmark 面，而不是直接进入热点优化。

## 2. 改动落点（What Changed）

- `tests/test_stageii_benchmark.py`：新增 TDD 测试，固定 legacy stageii sample 归一化与报告格式。
- `utils/stageii_benchmark.py`：实现独立于 `MoSh.load_as_amass_npz(...)` 的兼容加载、计时、repeatability 和 blocked-stage 汇总。
- `benchmark_stageii_public.py`：新增 CLI，输出 JSON 报告。
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/*`：新增 R0007 八件套与 `results/scorecard.md`。
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/results/public-stageii-benchmark.json`：本轮实际 benchmark 结果。
- `MAIN.md`：新增 R0007 运行索引。
- `docs/codex-potter/iterations/README.md`：新增 R0007 快速入口。

## 3. 与计划的偏离（Drift From Plan）

- 原计划：建立从 mocap/mcp 输入到 mesh/mp4 的第一版 benchmark/scorecard。
- 实际偏离：本轮只把公开可复跑的 stageii-ingest workload 固化为 baseline，没有声称 mesh/mp4 已覆盖。
- 偏离原因：README 中历史 conda 环境缺失，`blender` 不在 `PATH`，`psbody` 与完整 `model.npz` 资产也不在公开 repo 中；继续追求 full E2E 只会制造不可复现文档。
- 是否影响 DoD：不影响。按照 workflow 的优先级，本轮更重要的是先修复 benchmark 口径与复现性。

## 4. 关键实现 / 文档决策

- 决策：不用 `MoSh.load_as_amass_npz(...)` 直接构建 public benchmark
  - 原因：当前环境中 `moshpp.mosh_head` 通过 `mocap_interface` 间接依赖 `body_visualizer.mesh`，导入即失败。
  - 影响：public harness 改为独立兼容加载器，但其职责仅限 stageii-ingest baseline，不篡改 released 主逻辑。

- 决策：把 `markers_obs` 的 NaN 从“全局数值不稳定”里拆出来，单独记为 `markers_obs_nan_count`
  - 原因：sample 中存在合法缺失 marker，直接把 NaN 记成 `all_finite = false` 会误导 scorecard。
  - 影响：误差栏能同时表达“核心参数稳定”与“观测存在缺失值”。

- 决策：`.worktrees/gpu-stageii-foundation` 只部分采纳
  - 原因：已提交 foundation 有明确边界和测试，dirty 的 sequence/render 层仍在本地扩张，整体整树评估风险过高。
  - 影响：下一轮应按 committed foundation -> clean branch -> unified benchmark 的顺序推进，而不是直接 merge 当前 worktree。

## 5. 阶段内验证（Local Checks）

- `python3 -m pytest tests/test_stageii_benchmark.py -q`
  - 结果：PASS（`2 passed in 0.06s`）

- `python3 benchmark_stageii_public.py --input support_data/tests/mosh_stageii.pkl --output docs/codex-potter/iterations/round-0007-benchmark-baseline/results/public-stageii-benchmark.json --warmup-runs 1 --measured-runs 5`
  - 结果：PASS
  - 摘要：`latency_ms.mean=2.8471`，`throughput_ops_s=351.2321`，`repeatability.max_abs_diff=0.0`

- 在 `.worktrees/gpu-stageii-foundation` 中：
  - `python3 -m pytest tests/test_smplx_torch_wrapper.py tests/test_transformed_lm_torch.py tests/test_gmm_prior_torch.py tests/test_stageii_backend.py tests/test_stageii_torch_smoke.py -q`
  - 结果：PASS（`19 passed, 1 warning in 0.79s`）

## 6. 未完成项与交接点

- 公开 benchmark 仍未覆盖 mesh export 和 mp4 render；scorecard 已把阻塞条件写清。
- worktree 的 dirty sequence/render 扩展尚未进入统一 benchmark；只能算下一轮候选，不算本轮合并对象。
- 子代理基础设施在本轮多次返回上游 `502`；若下一轮仍不稳定，建议继续把“主会话最小收敛 + 文档审计链”作为兜底策略。
