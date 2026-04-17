---
round:
  id: "round-0007-benchmark-baseline"
  date: "2026-04-17"
repo:
  branch: "main"
  head_commit: "e67cc13"
environment:
  os: "Linux-6.8.0-94-generic-x86_64-with-glibc2.35"
  python: "3.10.12"
  cuda: "torch 2.11.0+cu130"
  gpu: "未直接参与本轮 public benchmark"
---

# 本轮测试记录（Test Log）

上层入口：

- 仓库级入口：[MAIN.md](../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

> 原则：记录“别人拿到这份文档，能否复现你说的结论”。成功与失败都必须写。

## 1. 测试范围（What We Tested）

- 新增 `tests/test_stageii_benchmark.py` 的单元测试
- public stageii baseline harness 的 JSON 报告生成
- `.worktrees/gpu-stageii-foundation` committed foundation 的关键测试子集

不在范围（Not Tested）：

- `save_smplx_verts.py` 的真实 mesh 导出
- `src/soma/render/parameters_to_mesh.py` 和 `mesh_to_video_standard.py` 的 mp4 路径
- 任何依赖私有 tennis 数据或 README 历史 conda 环境的命令

---

## 2. 环境信息（Environment）

- OS：`Linux-6.8.0-94-generic-x86_64-with-glibc2.35`
- Python：`3.10.12`
- 关键依赖版本：`numpy 2.2.6`、`torch 2.11.0+cu130`、`pytest 9.0.2`
- 硬件信息（如相关）：本轮 public benchmark 只测 pickle ingest，不直接触发 GPU 内核
- 额外缺口：
  - README 里记录的 `/home/user416/miniconda3/envs/soma/bin/python` 在当前机器不存在
  - `blender` 不在 `PATH`
  - `psbody` 不可导入
  - `body_visualizer.mesh` 缺失，导致当前环境不能直接导入 `MoSh.load_as_amass_npz(...)`

---

## 3. 执行记录（Commands & Results）

1. 命令：
   - `python3 -m pytest tests/test_stageii_benchmark.py -q`
   - 期望：新 benchmark harness 的测试全部通过
   - 结果：PASS
   - 关键输出摘要：`2 passed in 0.06s`

2. 命令：
   - `python3 benchmark_stageii_public.py --input support_data/tests/mosh_stageii.pkl --output docs/codex-potter/iterations/round-0007-benchmark-baseline/results/public-stageii-benchmark.json --warmup-runs 1 --measured-runs 5`
   - 期望：生成 JSON 报告，并写出可复用的 baseline 指标
   - 结果：PASS
   - 关键输出摘要：
     - sample：`legacy_stageii_pkl`，`gender=male`，`surface_model_type=smplh`
     - workload：`frames=581`，`pose_dim=156`，`marker_count=67`，`latent_marker_count=67`
     - 速度：`latency_ms.mean=2.8471`，`stdev=0.3734`，`throughput_ops_s=351.2321`
     - 误差：`repeatability.max_abs_diff=0.0`
     - 数值：`all_finite=true`，但 `markers_obs_nan_count=60`
     - blocked stages：`mosh_head_loader`、`mesh_export`、`mp4_render`

3. 命令：
   - 在 `.worktrees/gpu-stageii-foundation` 中运行：
   - `python3 -m pytest tests/test_smplx_torch_wrapper.py tests/test_transformed_lm_torch.py tests/test_gmm_prior_torch.py tests/test_stageii_backend.py tests/test_stageii_torch_smoke.py -q`
   - 期望：committed foundation 的关键测试子集通过，给部分采纳结论补工程化证据
   - 结果：PASS
   - 关键输出摘要：`19 passed, 1 warning in 0.79s`；warning 来自 `moshpp/tools/c3d.py` 的 “No analog data found in file.”

4. 命令：
   - `python3 -c 'import numpy, torch, pytest; print({...})'`
   - 期望：补齐 `test.md` 所需环境版本
   - 结果：PASS
   - 关键输出摘要：`python=3.10.12`、`numpy=2.2.6`、`torch=2.11.0+cu130`、`pytest=9.0.2`

---

## 4. 失败项与排查（If Any）

失败项清单：

- 直接基于 `MoSh.load_as_amass_npz(...)` 构建 public benchmark 失败
- README 所述 conda 环境不存在
- `blender` / `psbody` / full `model.npz` 资产缺失，无法把本轮 public benchmark 扩到 mesh/mp4

已做排查：

- `MoSh` 路线失败不是 sample 损坏，而是导入链经过 `body_visualizer.mesh.psbody_mesh_sphere`
- `support_data/tests/mosh_stageii.pkl` 本身可通过纯 `pickle` 兼容加载稳定解析
- worktree foundation 测试可以跑通，说明当前阻塞主要在 public E2E 环境，而不是 torch foundation 自测面

下一步建议：

- 在获得完整 mesh/render 环境前，继续把 public benchmark 锁在 stageii-ingest 层
- 下一轮先扩展 candidate 接线与 modern stageii sample 覆盖，再决定是否进入 profiling

---

## 5. 回归风险点（Regression Watchlist）

- `utils/stageii_benchmark.py` 目前以 public sample 为主，后续若引入 modern `stageii_debug_details` 产物，需要回归 modern format 兼容性
- 当前 `markers_obs_nan_count` 是样例真实现象；若未来 sample 换版，scorecard 不能再默认固定为 `60`
- worktree 的 dirty sequence/render 层未经统一 benchmark gate，下一轮若误把它当作“已验证 candidate”会直接污染决策口径
