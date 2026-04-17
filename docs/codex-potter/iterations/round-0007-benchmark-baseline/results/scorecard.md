# Round 0007 Public Scorecard

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../../README.md)

数据来源：

- `results/public-stageii-benchmark.json`
- `.worktrees/gpu-stageii-foundation` foundation 测试子集结果

硬件/环境：

- GPU: 未参与本轮 public benchmark；`torch 2.11.0+cu130` 可用
- OS: `Linux-6.8.0-94-generic-x86_64-with-glibc2.35`
- Python: `3.10.12`
- 关键依赖: `numpy 2.2.6` / `pytest 9.0.2`
- 阻塞依赖: `body_visualizer.mesh` 缺失、`psbody` 缺失、public repo 不含完整 `model.npz`、`blender` 缺失

Workload：

- 数据集/样例集：`support_data/tests/mosh_stageii.pkl`
- 样例格式：`legacy_stageii_pkl`
- 关键参数：`frames=581`、`pose_dim=156`、`marker_count=67`、`latent_marker_count=67`
- warm-up：1 次
- 采样：5 次，统计：mean/stdev/min/max

对比结论（Baseline -> Candidate）：

- 速度：
  - public stageii ingest latency_ms: `2.8471`（stdev `0.3734`）
  - throughput_ops_s: `351.2321`
  - Candidate 说明：`.worktrees/gpu-stageii-foundation` 当前尚未接到同一 public workload 的可比较执行链，本轮不宣称速度提升

- 误差：
  - repeatability.max_abs_diff: `0.0`
  - 核心数值稳定性：`all_finite=true`
  - 观测缺失：`markers_obs_nan_count=60`
  - Candidate 说明：当前 worktree 的 foundation 通过工程化测试，但本轮没有在统一 public workload 上得出数值误差结论

- 视觉 / 产物：
  - 已生成产物：`results/public-stageii-benchmark.json`
  - 未生成产物：mesh / png / mp4
  - 阻塞原因：缺 `psbody`、缺 licensed `model.npz`、缺 `blender`

- 工程化：
  - 可复现：PASS
    - 主仓库命令：`python3 benchmark_stageii_public.py --input support_data/tests/mosh_stageii.pkl --output docs/codex-potter/iterations/round-0007-benchmark-baseline/results/public-stageii-benchmark.json --warmup-runs 1 --measured-runs 5`
  - 候选工程化证据：PASS
    - worktree foundation 测试：`19 passed, 1 warning in 0.79s`
  - 可回滚：PASS
    - 本轮新增内容是独立 benchmark harness 和文档，不影响 released 主执行路径

决策：

- 结论：`部分采纳`
- 采纳对象：`.worktrees/gpu-stageii-foundation` 的 committed foundation（`9e686ea feat: add torch stageii backend` 及其已提交的 wrapper/prior/attachment/smoke/backend selector）
- 暂不采纳：dirty 的 sequence/render 扩展层（当前 `git status --short` 中的未提交修改与未跟踪文件）
- 决断点：
  - Go：将 committed foundation cherry-pick 到干净分支，在同一 benchmark/scorecard 下继续扩展 candidate workload
  - No-Go：在未获得 clean candidate 和 full mesh/render 环境前，不讨论整树 merge
