# GPU 逐帧求解器重写实验归档

Date: 2026-04-21  
Related doc: [GPU逐帧求解器提速分析.md](./GPU逐帧求解器提速分析.md)

## 归档范围

本文归档 2026-04-20 到 2026-04-21 这几轮围绕 GPU `stageii` 逐帧路径所做的高风险求解器重写、相关代码接线、测试结果，以及已经明确否掉的方向。

这里刻意区分两类内容：

- **已落库并保留在当前工作区的改动**
- **只做过真实 benchmark、但未保留为当前主线实现的实验方向**

## 当前工作区里实际保留的改动

### 1. Batched Independent L-BFGS 路径

相关文件：

- `moshpp/optim/batch_frame_fit_torch.py`
- `moshpp/optim/__init__.py`
- `moshpp/chmosh_torch.py`
- `tests/test_batch_frame_fit_torch.py`
- `tests/test_chmosh_torch.py`

实际内容：

- 新增 `fit_stageii_frames_batched_torch(...)`
- 在 `chmosh_torch.py` 中接入 `runtime.frame_solver=batched_lbfgs`
- 当 `sequence_chunk_size == 1` 且 `frame_batch_size > 1` 时，允许逐帧主线改走 batched solver
- 为 batched 路径补了独立测试

### 2. `adaptive_exact` 逐帧自适应路由

相关文件：

- `moshpp/chmosh_torch.py`
- `tests/test_chmosh_torch.py`

实际内容：

- 新增 `runtime.frame_solver=adaptive_exact`
- 支持在逐帧主线中混合使用：
  - frame 0 exact solve
  - adaptive fast path
  - exact fallback
- 新增 `stageii_debug_details["adaptive_frame_solver_stats"]`
- 新增 fast path 相关 runtime 选项：
  - `adaptive_fast_refine_iters`
  - `adaptive_fast_max_eval`
  - `adaptive_residual_threshold_mm`
  - `adaptive_transl_velocity_alpha`
  - `adaptive_latent_velocity_alpha`
  - `adaptive_anchor_stride`
  - `adaptive_fast_optimizer`
  - `adaptive_fast_lr`
  - `adaptive_pose_corrector_iters`
  - `adaptive_pose_corrector_lr`
  - `adaptive_pose_corrector_body_dofs`
  - `adaptive_fallback_batch_size`

### 3. 最新一轮 forward-gate 系列扩展

相关文件：

- `moshpp/chmosh_torch.py`
- `tests/test_chmosh_torch.py`
- `docs/superpowers/specs/2026-04-21-adaptive-forward-gate-design.md`
- `docs/superpowers/plans/2026-04-21-adaptive-forward-gate.md`

实际内容：

- 将 `adaptive_exact` 的 fast path 从“弱化版 `fit_stageii_frame_torch(...)`”改成了真正的：
  - predictor
  - verifier
  - exact fallback
- fast path 现在不再直接调用优化器，而是：
  1. 用上一个已解状态做 latent/transl 外推
  2. 调 `evaluate_stageii_frame(...)` 做纯前向评估
  3. 做一次固定 pose 的 translation correction
  4. 再次前向评估
  5. 只有 residual 通过阈值才接受，否则走 exact fallback
- 当前代码还保留了两条实验性扩展入口，用于复现实验结论：
  - fast path 上的 low-DOF pose corrector
  - reject 后的 batched exact fallback 尝试
- diagnostics 中新增：
  - `fast_initial_residual_mean_mm`
  - `fast_initial_residual_p90_mm`
  - `fast_pose_residual_mean_mm`
  - `fast_pose_correction_mean_deg`
  - `fast_translation_correction_mean_mm`
  - `fast_translation_correction_p90_mm`
  - `batched_fallback_batches`
  - `batched_fallback_frames`

## 测试归档

### 已通过的回归测试

本轮最终状态：

```bash
pytest -q tests/test_frame_fit_torch.py tests/test_batch_frame_fit_torch.py tests/test_chmosh_torch.py
```

结果：

- `69 passed`
- `1 warning`（`c3d.py` 提示 `No analog data found in file`，不影响本任务）

### 关键新增测试点

`tests/test_chmosh_torch.py` 目前覆盖了这些 adaptive 场景：

- `adaptive_exact` 路由是否被正确选中
- fast path accept 时是否绕开 exact fallback
- fast residual 超阈值时是否回退到 exact solve
- translation correction 是否在 accept 前实际生效
- pose correction 是否在 accept 前实际生效
- batched fallback 是否按配置尝试批量 exact rescue
- `adaptive_fast_optimizer` / `adaptive_fast_lr` 配置是否正确传入 fast path
- `adaptive_pose_corrector_*` / `adaptive_fallback_batch_size` 配置是否正确传入 adaptive 路径

## Benchmark 归档

## 一、第一轮：Batched Independent L-BFGS 接线结果

这轮的目标是先把“多帧共享 body forward/backward”接进逐帧主线，验证有没有低风险吞吐收益。

### a3 小窗口 smoke

数据：

- 质量基准：`data/a3.pkl`
- smoke：前 24 帧

结果：

| 配置 | 时间 | Marker residual mean | Marker residual p90 |
|---|---:|---:|---:|
| 旧单帧路径 | 17.99 s / 24 帧 | 4.736 mm | 8.893 mm |
| 新 batched LBFGS | 17.25 s / 24 帧 | 4.703 mm | 8.900 mm |

结论：

- 质量没有下降，均值略好
- 速度只有约 `1.04x`
- 这条线不足以把整条 `4090-haonan-73` 压到 1 小时以内

## 二、第二轮：exact 路径保守优化探索（未保留为主线）

这轮主要尝试“保持 exact 语义不变，只优化 warmup/history 复用”。

### 真实序列探索脚本结果（早期 exploratory harness）

数据：

- `4090-haonan-73.c3d` 前 100 帧

结果：

| 配置 | 时间 | Marker residual mean |
|---|---:|---:|
| exact baseline | 49.63 s / 100 帧 | 4.4642 mm |
| persist warmup history | 45.96 s / 100 帧 | 4.4546 mm |
| persist warmup + transl init 0.5 | 43.03 s / 100 帧 | 4.4875 mm |
| persist warmup + warmup10 + refine60 + transl init 0.5 | 39.08 s / 100 帧 | 4.6117 mm |

结论：

- 纯 exact 保守优化能拿到的安全收益大约是 `7%`
- 速度仍然在 `2.4h` 量级，不足以满足目标
- 这轮没有作为主线保留

## 三、第三轮：`adaptive_exact` 原型探索（部分结论被吸收到主线）

这轮开始从“安全 exact”转向“预测 + 守门 + fallback”的混合结构。

### 早期 no-anchor prototype（探索结论，非当前代码）

真实 `real100` 探索结果：

| 原型 | 时间 | Marker residual mean |
|---|---:|---:|
| `noanchor_r12_thr4.70` | 19.64 s / 100 帧 | 4.5647 mm |
| `noanchor_r12_thr4.60` | 20.26 s / 100 帧 | 4.5343 mm |
| `noanchor_r10_thr4.70` | 15.83 s / 100 帧 | 4.5954 mm |

这轮非常重要，因为它第一次证明了：

- **“一小时量级的吞吐”并非完全不可能**
- 但问题转成了 **质量略退化**

这个结论直接推动了后面的 `adaptive_exact` 代码接线。

## 四、第四轮：已落库的 `adaptive_exact`（旧 fast-solver 版本）

这一轮是已经接入 `chmosh_torch.py` 的 `adaptive_exact`，fast path 仍然是一个削弱版 solver，不是纯前向 gate。

### 当前可比 benchmark harness 下的结果

为了避免不同 exploratory harness 之间数字不可比，这里只保留同一入口下的对比结果。

#### `real100`

资产源：

- 参考 stageii 资产：`/home/alpen/DEV/tmp/native_zh73_smoke20/work/input/wolf001/bbb/cpu_reference/4090-haonan-73_cpu_reference_stageii.pkl`
- mocap：`data/4090-haonan-73.c3d`

结果：

| 配置 | 时间 | Marker residual mean | fast accept | fallback exact |
|---|---:|---:|---:|---:|
| exact baseline | 77.17 s / 100 帧 | 4.5881 mm | - | - |
| `adaptive_exact`（旧 fast-solver 版） | 46.36 s / 100 帧 | 4.5989 mm | 59 | 40 |

#### `a3_80`

资产源：

- `data/a3.pkl`
- raw mocap：`/home/alpen/DEV/tmp/codex_a3_noanchor_verify2/a3_raw_mocap.pkl`

结果：

| 配置 | 时间 | Marker residual mean | fast accept | fallback exact |
|---|---:|---:|---:|---:|
| exact baseline | 50.77 s / 80 帧 | 4.6066 mm | - | - |
| `adaptive_exact`（旧 fast-solver 版） | 41.70 s / 80 帧 | 4.6099 mm | 42 | 37 |

补充 sweep 结论：

- `adaptive_fast_refine_iters=6~8` 比 `12` 略好
- 最好一档约 `41.89 s / 100 帧`
- 即便如此，外推仍然是 `2h+`，远高于 1 小时目标

## 五、第五轮：forward-gate 重写（当前最新代码）

这轮把 fast path 真正改成了纯前向 verifier，不再让 fast frames 支付优化器开销。

### `real100`

结果：

| 配置 | 时间 | Marker residual mean | fast accept | fallback exact |
|---|---:|---:|---:|---:|
| exact baseline | 77.17 s / 100 帧 | 4.5881 mm | - | - |
| forward-gate `adaptive_exact` | 46.40 s / 100 帧 | 4.6002 mm | 45 | 54 |

diagnostics：

- `fast_initial_residual_mean_mm = 4.6340`
- `fast_residual_mean_mm = 4.6302`
- `fast_translation_correction_mean_mm = 0.1869`

### `a3_80`

结果：

| 配置 | 时间 | Marker residual mean | fast accept | fallback exact |
|---|---:|---:|---:|---:|
| exact baseline | 50.77 s / 80 帧 | 4.6066 mm | - | - |
| forward-gate `adaptive_exact` | 41.64 s / 80 帧 | 4.6137 mm | 29 | 50 |

diagnostics：

- `fast_initial_residual_mean_mm = 4.6470`
- `fast_residual_mean_mm = 4.6429`
- `fast_translation_correction_mean_mm = 0.1916`

结论：

- 这轮重写**没有**把 wall-clock 压到新的量级
- 原因不是 fast path 还有 optimizer 开销，而是：
  - **pose predictor 依然不够准**
  - translation correction 只能修一部分误差
  - exact fallback 仍然过多

## 六、第五轮附加 sweep：anchor stride

为验证“固定 exact anchor 是否能改善 predictor 漂移”，做了小规模 sweep。

### `real100`

| `adaptive_anchor_stride` | 时间 | Marker residual mean | anchor exact | fast accept | fallback exact |
|---|---:|---:|---:|---:|---:|
| 0 | 46.10 s / 100 帧 | 4.6002 mm | 0 | 45 | 54 |
| 4 | 43.98 s / 100 帧 | 4.6031 mm | 24 | 40 | 35 |
| 8 | 44.36 s / 100 帧 | 4.6059 mm | 12 | 39 | 48 |

结论：

- `anchor_stride=4` 是这组里最好的平衡点
- 但提升幅度仍然太小
- anchor 只能部分抑制漂移，不能从根本上解决 1 小时目标问题

## 七、第六轮：回到提速分析文档方案 A，直接测 `real-mcp-baseline`（已否）

这轮不是继续拧 per-frame，而是回到提速分析文档原本的低风险路线：直接评估现有 chunked sequence 主线能不能替代 per-frame。

### `4090-haonan-73` matched real 300 帧结果

资产：

- benchmark：`/home/alpen/DEV/tmp/codex_real_mcp_baseline_eval_20260421/4090-haonan-73_chunked_eval_benchmark.json`
- candidate stageii：`/home/alpen/DEV/tmp/native_zh73_smoke20/work/input/wolf001/4090-haonan-73_chunked_eval_20260421_stageii.pkl`
- reference per-frame：`/home/alpen/DEV/tmp/gpu_optimize_loop/gpu_perframe_baseline_stageii.pkl`

结果：

| 配置 | 时间 | Marker residual mean | Marker residual p90 |
|---|---:|---:|---:|
| `real-mcp-baseline` chunked sequence | 25.44 s / 300 帧 | 25.20 mm | 41.54 mm |
| per-frame reference | - | 4.75 mm | 8.70 mm |

mesh/stageii 对照：

- `reference_delta_mean_mm = +20.45 mm`
- mean residual 相对 per-frame 约 `5.30x`
- `mesh_compare.frame_delta_l2.mean = 9.97`
- candidate `mesh_accel_l2.mean = 2.20`
- reference `mesh_accel_l2.mean = 0.18`

结论：

- 这条路的 wall-clock 很有吸引力，但质量退化不是“略差”，而是明显不可接受
- 因此 **不能** 因为现有 chunked baseline 而停掉 per-frame 路径的后续工作
- 提速分析文档里的“方案 A 已实测完成，结论是否定”

## 八、第七轮：low-DOF pose corrector 原型（已否）

这轮尝试验证“在 fast verifier 前插一个极便宜的 pose 修正器”能不能把更多帧留在 fast path。

### `real100 / a3_80` benchmark

资产：

- 汇总：`/home/alpen/DEV/tmp/codex_pose_corrector_bench_20260421/summary.json`

#### `real100`

| 配置 | 时间 | Marker residual mean | fast accept | fallback exact |
|---|---:|---:|---:|---:|
| exact | 67.42 s / 100 帧 | 4.5881 mm | - | - |
| `adaptive_base` | 48.86 s / 100 帧 | 4.6002 mm | 45 | 54 |
| `adaptive_pose_root1` | 66.51 s / 100 帧 | 4.5881 mm | 0 | 99 |
| `adaptive_pose_body8` | 66.48 s / 100 帧 | 4.5881 mm | 0 | 99 |

#### `a3_80`

| 配置 | 时间 | Marker residual mean | fast accept | fallback exact |
|---|---:|---:|---:|---:|
| exact | 53.54 s / 80 帧 | 4.6066 mm | - | - |
| `adaptive_base` | 45.22 s / 80 帧 | 4.6137 mm | 29 | 50 |
| `adaptive_pose_root1` | 54.19 s / 80 帧 | 4.6066 mm | 0 | 79 |
| `adaptive_pose_body8` | 54.14 s / 80 帧 | 4.6066 mm | 0 | 79 |

诊断信号：

- `real100` 上 `fast_pose_residual_mean_mm` 达到 `592-619 mm`
- `a3_80` 上 `fast_pose_residual_mean_mm` 达到 `596-616 mm`
- `fast_pose_correction_mean_deg` 达到 `6757-9067 deg`

结论：

- 这条最小原型没有降低 fallback，反而把 fast path 基本全部打回 exact
- residual 回到 exact 档，不是因为 corrector 成功，而是因为它让绝大多数帧都 fallback 了
- 当前这种“裸梯度一步”的 pose corrector 方向已判死，不值得继续拧超参

## 九、第八轮：batched exact fallback 原型（接线保留，真实效果已否）

这轮目标是：如果 fast reject 仍然很多，就把 reject 帧攒成小批量做 exact rescue，而不是逐帧 fallback。

### 代码接线

已保留在当前工作区里的改动：

- `adaptive_fallback_batch_size` runtime 选项
- `adaptive_exact` reject 分支中的 batched fallback 尝试
- `batched_fallback_batches` / `batched_fallback_frames` diagnostics

对应测试：

- `tests/test_chmosh_torch.py` 中新增 batched fallback 行为测试
- runtime 参数解析测试也已补齐

### `real100 / a3_80` benchmark

资产：

- 汇总：`/home/alpen/DEV/tmp/codex_batched_fallback_bench_20260421/summary.json`
- 诊断 probe：`/home/alpen/DEV/tmp/codex_batched_fallback_bench_20260421/real100_probe20_diagnostic.json`

#### `real100`

| 配置 | 时间 | Marker residual mean | fast accept | fallback exact | batched fallback |
|---|---:|---:|---:|---:|---:|
| exact | 78.71 s / 100 帧 | 4.5881 mm | - | - | - |
| `adaptive_base` | 60.63 s / 100 帧 | 4.6002 mm | 45 | 54 | 0 |
| `adaptive_batched_fallback` | 50.98 s / 100 帧 | 4.6002 mm | 45 | 54 | 0 |

#### `a3_80`

| 配置 | 时间 | Marker residual mean | fast accept | fallback exact | batched fallback |
|---|---:|---:|---:|---:|---:|
| exact | 53.49 s / 80 帧 | 4.6066 mm | - | - | - |
| `adaptive_base` | 45.16 s / 80 帧 | 4.6137 mm | 29 | 50 | 0 |
| `adaptive_batched_fallback` | 47.01 s / 80 帧 | 4.6137 mm | 29 | 50 | 0 |

probe 诊断结论：

- 在真实 `real100` 前 20 帧探针里，每次 2 帧 batched 调用都返回 `fallback_mask=[true, true]`
- `fallback_reasons` 全部是 `line_search_failed`
- 因此真实运行行为不是“批量 fallback 生效但收益小”，而是“batched solver 一上真数据就把自己全退回单帧 exact”

结论：

- `方向 B` 的外层 plumbing 已经接好了，但当前收益为零
- 问题不在 `adaptive_exact` 调度逻辑，而在 `batch_frame_fit_torch.py` 自身的优化稳定性
- 如果继续做这条线，下一轮应直接排查 batched solver 的 line search / fallback 机制，而不是继续改 adaptive 外层

## 已明确否掉的方向

### 1. Adam fast path

已在当前 adaptive 框架下验证：

| fast path | 时间 | fast accept | fallback exact | 结论 |
|---|---:|---:|---:|---|
| Adam, 4 iters, lr=0.05 | 63.84 s / 100 帧 | 0 | 99 | 全量 fallback，方向无效 |
| Adam, 6 iters, lr=0.05 | 63.35 s / 100 帧 | 0 | 99 | 全量 fallback，方向无效 |

结论：

- 在当前目标函数和初始化下，Adam fast path 无法给出可接受 seed
- 这条线已经明确否掉

### 2. 只继续拧 `adaptive_fast_refine_iters`

真实 sweep 表明：

- 把 fast solver 从 `12` 砍到 `6~8` 只能拿到有限改进
- 上限仍然是 `40+ s / 100 帧`
- 不足以再投入更多时间做小调参

### 3. 只做 translation correction 的纯前向 gate

forward-gate 这一轮已经证明：

- translation correction 确实在局部起作用
- 但主要误差不在 translation，而在 **pose predictor**
- 因此“只修平移”不是足够强的 fast 路径

## 当前结论

截至 2026-04-21，这几轮工作的结果可以归纳为：

1. **batched LBFGS 接线**是安全的，但收益太小，不足以满足 1 小时目标。
2. **`real-mcp-baseline` chunked 主线**已经直接测过，质量不可接受，不能替代 per-frame 路径。
3. **纯 exact 路径保守优化**最多只拿到大约 `7%`，不值得继续主攻。
4. **`adaptive_exact` 混合结构**仍然是当前 per-frame 路径里唯一真正有结构性收益的方向，因为它确实能把一部分帧从 exact 主路径里拿出去。
5. 但当前 fast path 无论是“弱化版 solver”、“纯前向 + 平移纠正”，还是“low-DOF pose corrector”，都还不足以把 fallback 压到足够低。
6. **batched exact fallback** 这条线当前也没有跑通真实收益，因为 batched solver 在真数据上系统性 `line_search_failed`。

## 下一轮最值得继续的方向

从当前证据看，后续如果还要继续 per-frame 提速，只剩两类真正值得投入的结构方向：

### 方向 A：更强的 pose predictor / 廉价 pose corrector

目标：

- 在 verifier 前增加一个比当前“裸梯度一步”更稳的 pose 修正器
- 候选形式：
  - 有 trust region / clamp 的极小步校正
  - 更强的 predictor，而不是在线梯度修正
  - 只对少量关键 pose channels 做受约束修正

原因：

- 当前瓶颈已经明确是 pose predictor 不准
- 只修 translation 不够
- 当前 low-DOF pose corrector 原型已经证明，直接裸梯度步会明显过冲

### 方向 B：batched exact fallback

目标：

- 把 fallback 帧攒成小批量
- 但前提是先修通 batched solver 在真实数据上的 line search 稳定性

原因：

- 当前 adaptive 框架里，真正贵的是 fallback exact
- 如果 fallback 不能再少，就只能让 fallback 本身更便宜
- 当前失败点已经定位，不在外层接线，而在 batched solver 内核

## 备注

本文记录的是“这几轮实验的归档状态”，不是最终生产配置建议。真正用于生产时，应优先以最新同入口 benchmark 为准，而不是引用早期 exploratory harness 里的数字。
