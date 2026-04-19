# 高质量 MCP 到 MP4 导出

这份说明面向“最终视频交付”场景，而不是 speed benchmark。目标是尽量在不改业务代码的前提下，把 `.mcp -> stageii -> OBJ/PC2/preview MP4` 这一条链路跑得更稳、更适合人工观看。

## 适用场景

- 你已经有一份真实 `.mcp`
- 你更在意最终 MP4 里的抖动、手指乱动和压缩伪影
- 你可以接受 solver 比 `real-mcp-baseline` 更慢

如果当前主线目标是快速筛候选、跑 benchmark、比较 quality JSON，请继续使用 `real-mcp-baseline` 或其他 retained candidate；不要默认把这套 final-video 配置拿去当 speed 基线。

## 1. 推荐的 stageii preset

当前仓库内置了一个更偏最终视频质量的 preset：`real-mcp-quality-video`。

它相对 `real-mcp-baseline` 的关键变化是：

- 更少的 chunk 边界：
  - `runtime.sequence_chunk_size=96`
  - `runtime.sequence_chunk_overlap=16`
- 更重的 sequence/refine 求解：
  - `runtime.sequence_seed_refine_iters=10`
  - `runtime.sequence_max_iters=120`
  - `runtime.refine_lr=0.02`
  - `runtime.sequence_lr=0.02`
- 更强的 temporal smoothness：
  - `runtime.sequence_temporal_accel=25.0`
  - `runtime.sequence_pose_accel=0.15`
  - `runtime.sequence_body_accel=0.35`
  - `runtime.sequence_hand_accel=1.25`
- 更保守地贴近 stitched reference：
  - `runtime.sequence_delta_pose=0.25`
  - `runtime.sequence_delta_trans=25.0`
- 更偏 final-video continuity 的 stitch / velocity 约束：
  - `runtime.sequence_chunk_stitch_mode=adaptive_transl_jump_pose_mesh_guard`
  - `runtime.sequence_transl_velocity=32`
  - `runtime.sequence_boundary_transl_velocity_reference=true`
  - `runtime.sequence_boundary_transl_velocity_reference_window=8`
  - `runtime.sequence_boundary_transl_velocity_reference_zero_seam=true`

这些参数的意图不是“让局部动作更自由”，而是：

- 降低 chunk seam 抖动
- 降低整体 body twitch
- 对 hand PCA 也施加时间二阶平滑，减少手指乱跳

## 2. 推荐完整流程

### 2.1 先跑 stageii

```bash
python run_stageii_torch_official.py \
  --mocap-fname "$MCP" \
  --support-base-dir "$SUPPORT" \
  --work-base-dir "$WORK" \
  --preset real-mcp-quality-video \
  --cfg surface_model.gender=male \
  --skip-benchmark
```

说明：

- 如果只是为了最终视频交付，通常建议加 `--skip-benchmark`
- 这条 preset 会比 baseline 慢很多，这是预期行为

### 2.2 导出高质量 preview MP4

如果还要同时导出 OBJ/PC2，可直接走 artifact 入口：

```bash
python export_stageii_artifacts.py \
  --input-pkl "$STAGEII_PKL" \
  --support-base-dir "$SUPPORT" \
  --width 1024 \
  --height 1024 \
  --supersample 2 \
  --ffmpeg-crf 16 \
  --ffmpeg-preset slow
```

默认情况下，`export_stageii_artifacts.py` 会使用 `subject-frontal` 机位：它会根据 stageii 里的 root orientation 自动解出人物正面，而不是固定看世界坐标的某个轴向。

如果只想单独重渲 MP4，不重复导 OBJ/PC2：

```bash
python render_video.py \
  --input-path "$STAGEII_PKL" \
  --model-path "$SUPPORT/smplx/male/model.npz" \
  --output-path output_quality.mp4 \
  --width 1024 \
  --height 1024 \
  --supersample 2 \
  --ffmpeg-crf 16 \
  --ffmpeg-preset slow \
  --force
```

`render_video.py` 现在同样默认使用 `subject-frontal`。如果你想保留旧的世界坐标前视，可显式传：

```bash
--camera-preset frontal
```

## 3. 为什么这些参数会改善最终视频

### 3.1 stageii 侧

当前仓库新增了三类 temporal acceleration 项：

- `runtime.sequence_pose_accel`
- `runtime.sequence_body_accel`
- `runtime.sequence_hand_accel`

它们分别会在 sequence evaluator 里产生：

- `accelP`
- `accelB`
- `accelH`

这和原本只平滑 `transl` 的 `accel` 不同。现在优化器会直接在：

- 全部 latent pose
- body latent pose
- hand latent PCA

上施加二阶时间平滑，而不是只在最终 mesh 上做事后平滑。对于长序列视频，这通常比后处理顶点滤波更自然。

### 3.2 render 侧

当前渲染器新增了：

- `--supersample`
- `--ffmpeg-crf`
- `--ffmpeg-preset`
- `--ffmpeg-path`

作用分别是：

- `--supersample 2`
  - 先按 `2x` 分辨率渲染，再缩回目标分辨率
  - 可以减少边缘锯齿和细节闪烁
- `--ffmpeg-crf 16 --ffmpeg-preset slow`
  - 改用 `ffmpeg/libx264`
  - 避免默认 OpenCV `VideoWriter` 带来的普通压缩质量

## 4. 调参建议

如果第一版结果还不够稳，可以按下面顺序继续试：

### 手指仍然明显乱跳

先增大：

```text
runtime.sequence_hand_accel=2.5
```

如果变得太僵，再回落到：

```text
runtime.sequence_hand_accel=1.5
```

### 身体整体仍然抖

优先加：

```text
runtime.sequence_body_accel=0.6
runtime.sequence_pose_accel=0.25
runtime.sequence_temporal_accel=40
```

### 仍有明显 chunk seam

优先保持 `real-mcp-quality-video` 的大 chunk / 宽 overlap，不建议先把：

- `sequence_chunk_size`
- `sequence_chunk_overlap`

改回 baseline。

## 5. 已知取舍

- 这套 preset 不是速度优先，而是最终视频优先
- 它更擅长压低 twitch / seam，不保证一定能修掉所有局部姿态 basin 问题
- 默认的 `subject-frontal` 会根据 stageii 的 root orientation 跟随人物朝向，因此更适合作为最终视频导出的默认机位
- `frontal` 相机 preset 仍然保留，但它只是世界坐标前视，不一定等于人物真实正面
- 如果自动 `subject-frontal` 仍不符合你的构图目标，继续使用 `--camera-x/--camera-y/--camera-z` 和 `--lookat-*` 手动覆盖

## 6. 建议工作方式

实际使用时，推荐分成两步：

1. 先用 `real-mcp-quality-video` 产出 `stageii.pkl`
2. 先直接看默认的 `subject-frontal` 导出结果
3. 如果构图还不理想，再反复重渲 preview MP4，调机位和编码参数

这样调视频视角、分辨率和编码时，不需要反复重跑长时间的 stageii 求解。
