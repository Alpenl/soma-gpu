# GPU 逐帧求解器提速分析

Date: 2026-04-20  
Hardware: NVIDIA RTX 4090 D  
Sequence: `wolf001 / 4090-haonan-73` (18919 frames, 10.5 min @ 30fps)

后续几轮高风险求解器重写、真实 benchmark 和已否掉方向的归档见：[GPU逐帧求解器重写实验归档.md](./GPU逐帧求解器重写实验归档.md)

状态更新（2026-04-21）：

- 本文推荐路线里的“方案 A：直接测 `real-mcp-baseline`”已经实测完成
- matched real 300 帧结果为 residual mean `25.20 mm`、p90 `41.54 mm`
- 该质量相对 per-frame reference 明显不可接受，因此方案 A 已判否
- 细节见归档文档中的对应实验条目

## 当前状态

GPU per-frame LBFGS（`chunk_size=1, maxiter=100, refine_lr=0.05`）：

| 指标 | CPU (dogleg) | GPU per-frame (LBFGS) |
|------|------|------|
| Marker residual mean | 4.90 mm | **4.76 mm** ✓ |
| Marker residual p90 | 5.33 mm | **5.01 mm** ✓ |
| 稳态速度 (GPU 空闲) | 0.56 s/帧 | **0.34 s/帧** (1.6x) |
| 稳态速度 (GPU 争抢) | — | ~0.73 s/帧 |

GPU 精度已超过 CPU。速度在 GPU 空闲时快 1.6x，在 GPU 被其他进程占用时反而更慢。

### 证据位置

- CPU 残差来源：`/home/alpen/DEV/tmp/native_zh73_smoke20/work/input/wolf001/bbb/cpu_reference/4090-haonan-73_cpu_reference_stageii.pkl`，前 300 帧逐帧计算 `sqrt(sum((markers_sim - markers_obs)^2, axis=-1)).mean()`
- GPU 残差来源：`/home/alpen/DEV/tmp/gpu_optimize_loop/gpu_perframe_baseline_stageii.pkl`（300 帧），由 `/home/alpen/DEV/tmp/gpu_optimize_loop/run_perframe_300f.py` 产出
- 速度数据：同一脚本的 wall-clock，以及单独的 50 帧 benchmark（跳过前 10 帧 CUDA JIT warmup）
- GPU 争抢速度：全序列运行期间 `nvidia-smi` 显示另一进程占 19.5GB，profile 单帧测得 726ms

## 瓶颈分析

### Forward 耗时分解（单帧，GPU 空闲时）

测量方法：对 frame 50 的 evaluator 各组件分别循环 200 次，`torch.cuda.synchronize()` 后取均值。

| 组件 | 耗时 | 占比 |
|------|------|------|
| `decode_stageii_latent_pose` | 0.06 ms | <1% |
| **Body model forward (SMPL-X)** | **6.90 ms** | **96%** |
| `decode_marker_attachment` | 0.11 ms | 1.5% |
| Pose prior | 0.10 ms | 1.4% |
| Loss computation | 0.01 ms | <1% |
| **Total forward** | **~8.5 ms** | |
| Backward | ~5.0 ms | |
| **Forward + Backward** | **~13.5 ms** | |

每帧实际约 25 次 forward+backward（LBFGS 提前收敛 + strong_wolfe line search 额外 closure 调用），总计 ~340ms/帧。

### Per-frame wall-clock 分解

测量方法：20 帧（frame 10-29），`perf_counter` + `cuda.synchronize`。

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Prep（构建 obs tensor/weights/attachment subset） | 9.7 ms | 1.3% |
| **Solve（LBFGS + GPU forward/backward）** | **726 ms** | **99%** |
| Post（detach + cpu copy） | 0.07 ms | <0.1% |

注：726ms 是 GPU 争抢环境下的数字。GPU 空闲时 solve 约 320ms。

### 核心发现：Body model batch 效率极高

测量方法：对每个 batch size 分别构建 body model，循环 100 次 forward，`cuda.synchronize` 后取均值。

| Batch size | Total time | Per-frame time | 相对 batch=1 |
|---|---|---|---|
| 1 | 7.7 ms | 7.67 ms | 1x |
| 4 | 7.7 ms | 1.92 ms | 4x |
| 8 | 8.2 ms | 1.03 ms | 7.5x |
| 16 | 7.9 ms | 0.49 ms | 15.6x |
| 32 | 7.8 ms | 0.24 ms | **32x** |

Batch=32 完整 forward+backward：12.2ms total，0.38ms/frame（100 次循环均值）。这是每次成功接受 step 的理想下界，不是完整 wall-clock 预测。

### GPU 利用率低的原因

GPU 利用率仅 28%（`nvidia-smi`）。原因不是 CPU prep 慢，而是 LBFGS 的 Python 解释器开销 + CUDA kernel launch latency：每次 LBFGS 迭代中 GPU 只忙 ~13ms，然后等 Python 端跑完 LBFGS two-loop recursion、strong_wolfe bracket/zoom 判断后才发起下一次 GPU 调用。

## 已尝试的 batch 方案及失败原因

1. **Batch LBFGS (sum loss)** — 29mm 精度（`/home/alpen/DEV/soma-gpu/moshpp/optim/batch_frame_fit_torch.py` 早期版本）。原因：LBFGS 维护全局 Hessian 近似，不同帧曲率差异大。
2. **Batch Adam (150-750 iters)** — 6.3-75mm 精度（同文件多个版本）。原因：Adam 收敛慢，需要极多迭代。
3. **Per-frame LBFGS with batched forward (v4 原型)** — 未完成（`moshpp/optim/batch_frame_fit_torch_v4.py`）。原因：PyTorch LBFGS 依赖 `closure` 重算语义，`strong_wolfe` line search 内部做 bracket/zoom 和多次 objective+gradient 重算（`torch/optim/lbfgs.py:40-170`, `332-489`）。

## 可行提速方案

### 方案 A：评估现有 chunked sequence 主线的吞吐（低风险，应先做）

> 2026-04-21 状态：已完成实测，结论是否定。`real-mcp-baseline` 在 matched real 300 帧上的 residual 明显不可接受，不能作为停掉 per-frame 提速工作的依据。具体数字见归档文档。

当前主线已支持 `sequence_chunk_size > 1` 的 batched sequence evaluator（`chmosh_torch.py:1682-1702`），并有多个已验证的 preset：

- `real-mcp-baseline`：`chunk_size=32, overlap=4, seed_refine_iters=5, refine_lr=0.05, sequence_lr=0.05`
- `real-mcp-chunk48ov8-deltapose4`：`chunk_size=48, overlap=8, delta_pose=4, stitch_mode=adaptive_transl_jump_pose_guard`

这些 preset 与 per-frame LBFGS 在目标函数（额外的 velocity/boundary/temporal 项）、优化器（sequence solver 默认 Adam）、初始化策略（seed refine prepass）、数据布局（full attachment + visible_mask）四个维度上都不同，因此 residual 数字不能直接和本文档开头的 per-frame LBFGS 做 apples-to-apples 对比。

这个方案的目标不是"复现 per-frame LBFGS 的精度"，而是"评估现有 chunked 主线在生产场景下是否够用"。

具体做法：
1. 用 `run_stageii_torch_official.py --preset real-mcp-baseline` 跑 300 帧，记录 wall-clock 和 marker residual
2. 用 `run_stageii_torch_pair.py` 做 baseline vs per-frame 的 stageii/mesh 对照
3. 如果 chunked 路径的 residual 可接受，直接用它作为生产配置

注意：之前全序列测得的 6.58mm 是 `real-mcp-quality-video` preset（chunk=96, overlap=16, 重 temporal 权重），不是 `real-mcp-baseline`。两者 residual 可能差异很大，不应混用。需要单独测 `real-mcp-baseline` 的 residual。

### 方案 B：自定义 Batched L-BFGS（高风险研究项）

原理：每帧维护独立的 L-BFGS history buffer `(N, m, D)`，forward 用 batch，Hessian 近似和 line search per-frame。

**注意事项：**
- 当前 per-frame solver 依赖 PyTorch LBFGS 的 `strong_wolfe` closure 重算语义（bracket/zoom、多次 objective+gradient 重算），要在 batch 维度上复现需要重写整个 line search。不与当前 per-frame LBFGS 数学等价。
- 还需要处理可见 marker 变化（每帧不同的 visible marker 集合）和统一 batch 数据布局（当前 per-frame 路径逐帧裁剪 attachment 子集，sequence 路径用 full attachment + visible_mask）。
- `25 × 12.2ms / 32 = 10ms/帧` 是 optimistic lower bound（假设每次 line search 只需一次 forward eval），实际 wall-clock 会更高。

建议先做简化原型：统一 full-marker + visible_mask 布局，用固定 step 或 Armijo line search 验证 wall-clock 和 residual，再决定是否投入 strong-Wolfe 级别实现。

### 方案 C：Batched Gauss-Newton / LM（探索项，需先测 Jacobian 成本）

原理：用 `torch.func.vmap + jacrev` 计算 per-frame Jacobian，解 `(J^T J + λI) δ = J^T r`。

显式 Jacobian 是完全不同于 `loss.backward()` 的 AD 工作负载，计算代价可能远大于一次 backward。在没有 microbenchmark 之前不应估算具体提速倍数。

下一步：用 `torch.func.jacrev` 对当前 evaluator 做一次 Jacobian 计算的 microbenchmark。

### 方案 D：torch.compile（正交优化项，需先排查失败根因）

当前状态：不可用。失败信息：

```
CalledProcessError: Command '['/usr/bin/gcc', '...cuda_utils.c', '-O3', '-shared', '-fPIC', ...]'
returned non-zero exit status 1.
```

当前环境：gcc 11.4.0, torch 2.11.0+cu130, triton 3.6.0。gcc 版本不是问题。真实根因尚未确认——可能是 Triton codegen、CUDA toolchain ABI、或某个内核特化问题。需要先拿到真实编译 stderr 再判断。

主线已接入 `compile_evaluator` 开关（`chmosh_torch.py:1666-1695`）。

下一步：设置 `TORCHDYNAMO_VERBOSE=1` 和 `TORCH_LOGS="+dynamo"` 复现失败并捕获完整 stderr。

## 推荐路线

1. **先评估方案 A**：用现有 `real-mcp-baseline` preset 跑 300 帧 benchmark，通过 `run_stageii_torch_pair.py` 与 per-frame LBFGS 做 stageii/mesh 对照。如果 chunked 路径的 residual 可接受（或通过调参改善），直接用它作为生产配置。
2. **并行排查方案 D**：捕获 `torch.compile` 的真实 stderr，判断修复成本。
3. **如果方案 A 精度不够**：做方案 B 的简化原型——统一数据布局，用简化 line search 验证吞吐，再决定是否投入完整实现。
4. **方案 C 作为长期探索**：先做 Jacobian microbenchmark，有数据后再评估。
