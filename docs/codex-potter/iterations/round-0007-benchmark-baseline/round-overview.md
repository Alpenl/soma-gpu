# Round 0007 (benchmark-baseline) 概览

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

本轮目标是把 GPU 路线从“只有方向，没有 gate”的状态，推进到“至少有一个公开可复跑的 baseline harness 和第一版 scorecard”。由于当前机器缺少 README 里记录的历史 conda 环境，`blender` 不在 `PATH`，且公开仓库不包含完整的 SMPL-X mesh 导出资产，本轮不伪装成 full E2E mp4 benchmark，而是先把 `support_data/tests/mosh_stageii.pkl` 固化为 public stageii-ingest workload，再用同一轮文档明确 worktree 候选的采纳边界。

## 1. 本轮目标（Goals）

- 新增一个公开可复跑的 benchmark harness：`benchmark_stageii_public.py`。
- 固定 `support_data/tests/mosh_stageii.pkl` 为 round-0007 的最小 public workload，并输出 JSON 报告。
- 产出第一版 scorecard，至少覆盖速度、误差、产物/阻塞、工程化。
- 对 `.worktrees/gpu-stageii-foundation` 给出采纳 / 部分采纳 / 暂不采纳结论，不再把它当作“整树直接合入”的候选。

## 2. 本轮范围（In Scope）

- `tests/test_stageii_benchmark.py`：锁定 legacy stageii sample 归一化与 benchmark 报告格式。
- `utils/stageii_benchmark.py`：实现独立于 `MoSh.load_as_amass_npz(...)` 的 public stageii-ingest 归一化与 benchmark 汇总。
- `benchmark_stageii_public.py`：CLI 入口，生成 round-0007 结果 JSON。
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/*`：完整八件套与 `results/scorecard.md`。
- `MAIN.md` 与 `docs/codex-potter/iterations/README.md`：新增 R0007 索引。

## 3. 不在本轮范围（Out of Scope）

- 不宣称已经跑通 `convert_tennis.py -> save_smplx_verts.py -> parameters_to_mesh.py -> mesh_to_video_standard.py` 的 full public mp4 benchmark。
- 不在本轮直接合并 `.worktrees/gpu-stageii-foundation` 的 dirty sequence/render 扩展。
- 不在本轮进入热点 profiling、内核优化或 torch backend 接线。

## 4. 本轮输出物（Deliverables）

P0（必须产出）：

- `docs/codex-potter/iterations/round-0007-benchmark-baseline/round-overview.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/plan.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/code.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/test.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/next-round-suggestions.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/summary.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/commit.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/close.md`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/results/public-stageii-benchmark.json`
- `docs/codex-potter/iterations/round-0007-benchmark-baseline/results/scorecard.md`
- `benchmark_stageii_public.py`
- `utils/stageii_benchmark.py`
- `tests/test_stageii_benchmark.py`
- 更新后的 `MAIN.md`
- 更新后的 `docs/codex-potter/iterations/README.md`

P1（本地但不进 git）：

- 更新后的 `.codexpotter/projects/2026/04/17/1/MAIN.md`
- 更新后的 `.codexpotter/kb/README.md`
- 新增 `.codexpotter/kb/round-0007-benchmark-context.md`

## 5. Exit Criteria（本轮退出标准）

满足以下条件即可结束本轮：

- public workload 与复现命令被固定到 repo 内可见资产，而不是继续依赖口头环境说明。
- 第一版 scorecard 已写出速度、误差、产物/阻塞、工程化四类字段。
- `test.md` 清楚记录：public benchmark 可以跑，mesh/mp4 public path 仍被依赖阻塞。
- `.worktrees/gpu-stageii-foundation` 的结论已经从“候选树”收敛为“committed foundation 部分采纳，dirty 扩展暂不采纳”。
