# 如何将原生 MoSh++ 改造成 GPU 版

## 1. 目的

本文档说明当前公开版原生 `MoSh++` 为什么不能直接切到 GPU，以及如果要在尽量保持原始求解逻辑的前提下做 GPU 加速，应该改哪些模块、按什么顺序改、哪些地方可以保持不动。

本文档面向当前这份公开版代码副本，并以原始 CPU/Chumpy 路线作为对照。

这里说的“GPU 版”特指 `stageii` 拟合过程的 GPU 化，而不是最后渲染视频的 GPU 化。当前仓库里的渲染路径主要仍是 Blender / `bpy` 相关工具，这里不展开讨论渲染后端，因为它不是当前主要瓶颈。

## 2. 当前瓶颈在哪里

当前最慢的是 `moshpp/chmosh.py` 里的 `mosh_stageii(...)`。它的工作方式是：

1. 逐帧读取 marker 观测。
2. 为每一帧构造 `data`、`poseB`、`poseH`、`velo` 等目标项。
3. 先做一次 body warm start。
4. 再做一次 full pose 优化。
5. 每帧都调用 `ch.minimize(..., method="dogleg")` 反复迭代。

关键位置：

- `moshpp/chmosh.py:458`
- `moshpp/chmosh.py:584`
- `moshpp/chmosh.py:651`
- `moshpp/chmosh.py:669`
- `moshpp/chmosh.py:703`

从当前代码路径看，`stageii` 的主要工程瓶颈更可能来自“逐帧的 CPU 优化器循环”，而不是单次 SMPL-X 前向；但如果要把这条结论写成严格性能结论，仍然需要单独做 profiling。

## 3. 为什么现在不能直接开 GPU

原因不是“没配好 CUDA”，而是“数学图和优化器本来就不是 GPU 架构”。

### 3.1 当前模型图是 Chumpy，不是 Torch

`moshpp/models/smpl_fast_derivatives.py` 里直接使用了：

- `chumpy`
- `scipy.sparse`
- `psbody.smpl.fast_derivatives.smplcpp_chumpy`

关键位置：

- `moshpp/models/smpl_fast_derivatives.py:42`
- `moshpp/models/smpl_fast_derivatives.py:44`
- `moshpp/models/smpl_fast_derivatives.py:48`

这条链本质上是：

`marker loss -> chumpy graph -> CPU sparse derivative -> dogleg solver`

它不是 `torch.Tensor` 图，所以没有 `.cuda()`、没有 autograd graph、也没有现成 GPU optimizer 可以直接接。

### 3.2 Marker 变换层也是 CPU/Chumpy

`moshpp/transformed_lm.py` 里的 `TransformedCoeffs` 和 `TransformedLms` 负责把 latent marker 绑定到 body 表面附近。这里用了：

- `NearestNeighbors(..., algorithm="kd_tree")`
- `chumpy` 向量运算

关键位置：

- `moshpp/transformed_lm.py:73`
- `moshpp/transformed_lm.py:120`

这层如果不改，整个 marker-to-mesh 的误差图仍然留在 CPU。

### 3.3 Pose prior 也是 Chumpy 稀疏导数链

`moshpp/prior/gmm_prior_ch.py` 里的 body prior 和 hand prior 不是 torch 实现，而是 Chumpy 版本的 GMM prior。它在导数里还会构造 `scipy.sparse.csc_matrix`。

关键位置：

- `moshpp/prior/gmm_prior_ch.py:42`
- `moshpp/prior/gmm_prior_ch.py:79`
- `moshpp/prior/gmm_prior_ch.py:81`

因此，哪怕只把 body model 换成 GPU，如果 prior 还是这套，求解也断不开 CPU。

### 3.4 仓库里有并行配置，但不是 GPU 加速

仓库中确实有：

- `support_data/conf/parallel_conf/moshpp_parallel.yaml`
- `src/soma/tools/parallel_tools.py`

但公开版的 `run_parallel_jobs(...)` 最终仍然是串行执行：

- `src/soma/tools/parallel_tools.py:77`
- `src/soma/tools/parallel_tools.py:78`

并且默认配置里也是：

- `gpu_count: 0`
- `pool_size: 1`

所以这个“parallel”更像是任务调度接口的壳，不是 released 的多 GPU/多进程求解器。

## 4. 能改成 GPU 吗

能，但要明确分成两种目标。

### 4.1 目标 A：保留原生数学定义，重写执行后端

这是最合理的 GPU 方案。

意思是：

- 保留 marker data term 的定义
- 保留 pose/body/hand prior 的定义
- 保留时间平滑项的定义
- 但把 Chumpy 图和 dogleg 调用，改成 Torch 图和 GPU optimizer

这样结果最有机会接近原生实现，同时可以真正利用 GPU。

### 4.2 目标 B：做一个“近似等价”的新拟合器

这条路更快，但不再是严格意义上的原生 MoSh++。

比如：

- 用 torch SMPL-X body model
- 用 marker 或 joint target 直接做位置拟合
- 加简单的 temporal smoothness
- 加 body/hand regularization

这种实现更容易跑起来，也更容易上 GPU，但输出和原始 MoSh++ 会有方法学差异。

如果目标是“原生路径 GPU 化”，应该优先选目标 A。

## 5. 最小可行改造范围

如果只改最关键路径，建议只重写 `stageii`，不要先碰 `stagei`。

原因：

1. `stageii` 是当前最大瓶颈。
2. `stagei` 通常只做少量形状和 marker layout 相关拟合，工程上往往比 `stageii` 小得多，但这一点仍应以 profiling 结果为准。
3. 先只替换 `stageii`，更容易和当前环境共存，也更便于与原生输出做逐步对照。

建议保留：

- `moshpp/mosh_head.py` 的整体流程入口
- marker layout 文件格式
- `stagei` 结果文件格式

建议重写：

- `moshpp/chmosh.py` 中的 `mosh_stageii(...)`
- `moshpp/models/smpl_fast_derivatives.py`
- `moshpp/transformed_lm.py`
- `moshpp/prior/gmm_prior_ch.py`

## 6. 推荐的新架构

推荐新增这些文件，而不是直接大改原文件：

- `moshpp/chmosh_torch.py`
- `moshpp/models/smplx_torch_wrapper.py`
- `moshpp/transformed_lm_torch.py`
- `moshpp/prior/gmm_prior_torch.py`
- `moshpp/optim/frame_fit_torch.py`

然后在入口处通过配置开关选择：

- `backend: chumpy`
- `backend: torch`

这样做的好处是：

1. 可以和原生实现逐步对照。
2. 出现精度退化时容易回退。
3. 便于做 smoke test 和逐项替换。

## 7. 每个模块要怎么改

### 7.1 Body model：把 SMPL-X 前向和参数表示改成 Torch

当前的 `SmplModelLBS` 建在 Chumpy 上。GPU 版应替换为 torch 版本的 body model。

要求：

1. 输入参数全部使用 `torch.Tensor`。
2. 支持 `pose`、`betas`、`trans`、可选 `expression`。
3. 输出至少包括：
   - `vertices`
   - `joints`
   - 用于 marker surface attachment 的局部几何量
4. 必须支持批量帧处理，即 `B x D` 的 pose 和 trans。

建议：

1. 优先复用 `human_body_prior` 或 `smplx` 的 torch 实现。
2. 单独包一层 wrapper，对齐当前 `stageii` 需要的输出结构。
3. 先支持 `smplx`，其他 model type 后补。

### 7.2 Marker attachment：把 surface marker 变换层改成 Torch

当前 `TransformedCoeffs` 和 `TransformedLms` 的逻辑本质是：

1. 在 canonical body 上为每个 marker 找到近邻顶点。
2. 在局部坐标系中记录 marker 相对 body 的系数。
3. 对任意新 pose，把这些系数映射回当前 body 表面附近的位置。

Torch 版更适合分两步：

1. 预处理阶段：
   - 在 CPU 上做最近邻搜索。
   - 把每个 marker 的 `closest vertex ids` 和局部系数固定下来。
2. 拟合阶段：
   - 只在 GPU 上做局部基向量重建与 marker 位置解码。

需要说明当前公开版里的实际情况：

1. 在 `stageii` 中，`TransformedCoeffs` 绑定的是 `can_model.r` 和固定的 `markers_latent`，最近邻搜索更接近一次性预处理。
2. 真正持续留在 CPU 上的是后续 `TransformedLms` 的 Chumpy 运算，以及整个 Chumpy 图和 `dogleg` 优化过程。
3. 在 `stagei` 中，`TransformedCoeffs(can_body=can_model, markers_latent=markers_latent)` 绑定的是优化变量，最近邻和局部基向量构造在优化过程中可能更频繁触发，因此它比 `stageii` 更值得单独注明。

所以“两步法”的意义不是说当前 `stageii` 每个优化 step 都在重复跑 KD-tree，而是说 GPU 改造时应把“CPU 近邻预处理”和“GPU 上的 marker 解码”清晰拆开。

建议新增：

- `build_marker_attachment(...)`
- `decode_markers_from_attachment(...)`

其中第一步可以保留 CPU，第二步必须纯 torch。

### 7.3 Prior：把 GMM prior 改成 Torch

`gmm_prior_ch.py` 要改成纯 torch 版本。

目标不是完全复刻 Chumpy 接口，而是复刻数值目标：

1. body pose prior
2. hand pose prior
3. 可选 face prior

实现建议：

1. 把 GMM 的 `means`、`precs`、`weights` 预加载成 torch tensor。
2. 不要只复刻“Mahalanobis distance 的标量值”，而要尽量复刻当前 Chumpy 版本的 residual 构造方式。
3. 先做 hard-min 的 mixture 选择，再考虑 soft-min 近似。

注意：

- 如果要求最大限度贴近原实现，应先做 hard-min。
- 当前实现不是简单的标量 NLL，而是最小二乘 residual 向量：
  - `loglikelihoods = sqrt(0.5) * (x - mean) dot prec`
  - mixture 选择规则是 `argmin((logl ** 2).sum() - log(weight))`
  - 输出 residual 还会额外拼接 `sqrt(-log(weight))`
- torch 版如果忽略这些细节，即便“形式上还是 GMM prior”，也会在量级、梯度和最优解上偏离原版。
- 如果优化稳定性比严格一致性更重要，可以在完成对齐版之后，再单独评估 soft-min。

### 7.4 Optimizer：用 Torch 二阶或准二阶方法替代 Dogleg

原生实现是：

- Chumpy graph
- `dogleg`

GPU 版没有必要追求“求解器名字相同”，关键是目标函数和变量块划分尽量一致。

推荐分层实现：

1. 第一版：
   - `LBFGS`
   - 单帧 warm start
   - 单帧 full pose refine
2. 第二版：
   - 在已经复刻单帧或逐帧流程之后，再研究 chunk 内多帧联合优化
   - 把新增 temporal smoothness 明确标记为“方法学变体”
3. 第三版：
   - 研究 Gauss-Newton / LM 风格的自定义二阶近似

为什么不建议第一天就自己写 Dogleg：

1. 工作量大。
2. 数值稳定性难调。
3. 很容易在 trust region 和 line search 上花掉大量时间。

先用 torch `LBFGS` 把数值链跑通，通常是最稳的。

### 7.5 时序优化：先区分“对齐原版”和“主动改方法”

原生 `stageii` 虽然是逐帧推进，但使用了上一帧 pose 的外推项：

- `velo`

如果目标是前文的“目标 A: 尽量保留原生数学定义”，那第一版 torch 实现不应直接跳到分块联合优化，而应先尽量复刻当前的逐帧推进方式，包括：

1. 逐帧处理。
2. rigid init / body warm start / full pose refine 这类阶段划分。
3. 与当前 `velo` 外推项尽量一致的时间约束。

只有当这条路径已经能与原生输出稳定对齐后，才建议进入“目标 B”路线，把 GPU 版改成“分块联合优化”。

例如：

1. 每个 chunk 处理 `60` 到 `240` 帧。
2. chunk 内一起优化：
   - `trans`
   - `body pose`
   - `hand pose`
3. 加入一阶或二阶时序平滑：
   - `pose[t] - pose[t-1]`
   - `pose[t] - 2 * pose[t-1] + pose[t-2]`

这样做的好处是：

1. GPU 能吃到 batch。
2. 手部和抖动会比逐帧独立更稳。
3. 整体速度通常比 CPU 逐帧 dogleg 更有优势。

但也要明确代价：

1. 这已经不再是“只替换执行后端”，而是目标函数和优化策略的实质变化。
2. 输出结果不再与原生逐帧 `dogleg` 严格可比。
3. 原有 loss 权重往往需要重新标定。

## 8. 建议的实施顺序

### 阶段 0：冻结参考基线

先固定一份原生参考输出，用于后续对比：

1. 选一个 100 帧 smoke 序列。
2. 保存原生 `stageii.pkl`。
3. 保存对应关键帧截图。
4. 记录每帧 marker error、hand marker error、时长。

没有这个基线，后面很难判断“更快了但是不是已经跑偏”。

### 阶段 1：只替换 body forward

目标：

1. 用 torch body model 输出与原版尽量一致的 vertices/joints。
2. 不接优化器。
3. 给相同 pose/betas/trans，比较顶点误差。

验收标准：

- 顶点误差足够小
- joint 误差足够小

### 阶段 2：只替换 marker attachment 解码

目标：

1. 给定同一套 marker attachment 参数。
2. 用 torch 版解码 marker 位置。
3. 与 Chumpy 版逐帧对比误差。

验收标准：

- marker 解码误差处于毫米级或更小

### 阶段 3：只替换 data term + prior 计算

目标：

1. 在不做完整优化的情况下，比较同一组 pose 参数下的 loss 数值。
2. 确保：
   - data term 量级一致
   - body prior 量级一致
   - hand prior 量级一致

### 阶段 4：做单帧 GPU 优化

目标：

1. 只拟合一个 frame。
2. 用 torch `LBFGS` 跑通：
   - rigid init
   - body warm start
   - full pose refine

验收标准：

1. 单帧拟合结果不扭曲。
2. marker 误差接近原版。
3. 运行速度开始出现 GPU 优势。

### 阶段 5：做逐帧 GPU 版 `stageii`

目标：

1. 保持与原版相同的逐帧推进结构。
2. 尽量保留现有 `velo` 外推项和阶段式求解顺序。
3. 先验证 torch 后端是否能在长序列上复刻原版趋势。

验收标准：

1. 长序列结果与原版没有明显方法学偏移。
2. marker 误差与可视效果仍接近原版。
3. GPU 后端开始体现吞吐优势。

### 阶段 6：做分块 GPU 优化（方法学变体）

目标：

1. 一次拟合几十到几百帧。
2. 引入时序项。
3. 支持 chunk overlap。

验收标准：

1. 长序列不抖。
2. chunk 拼接连续。
3. 总时长明显优于 CPU 原版。

注意：

1. 这一阶段属于前文“目标 B”路线，不应再宣称自己只是“原生后端替换”。
2. 这一阶段的指标应同时和“原版 CPU 输出”以及“阶段 5 的逐帧 torch 版输出”对比。

### 阶段 7：接回原有入口

最后再把 GPU 版通过配置接回：

- `moshpp/mosh_head.py`

不要一开始就改入口，先把新后端单独跑通。

## 9. 验证指标

要避免“速度快了但人扭曲了”，必须同时验这几类指标。

### 9.1 几何误差

1. marker mean error
2. marker p95 error
3. hand marker mean error
4. hand marker p95 error

### 9.2 动作稳定性

1. pose 一阶差分均值
2. pose 二阶差分均值
3. 手指关节角速度异常峰值

### 9.3 视觉检查

1. 正面视图
2. 侧面视图
3. 多帧抽帧拼图
4. 手部特写

### 9.4 性能

1. 每 100 帧耗时
2. 显存占用
3. 不同 chunk 大小下的吞吐

## 10. 一个现实的工程判断

如果目标是“这周内把当前项目提速”，最现实的通常不是立刻全量 GPU 重写，而是先做这两件事：

1. 评估是否值得补一个真正可用的 CPU 分段并行执行层。
2. 同时开始 stageii-only 的 torch 重写验证。

原因：

1. CPU 分段并行如果能做出来，几乎不改变数学定义，风险最低。
2. 但当前公开版仓库里的 `run_parallel_jobs(...)` 仍然是串行 stub，默认配置也没有 released 的多进程 / 多 GPU 执行能力，因此“做 CPU 分段并行”本身也是额外工程，不应误写成现成功能。
3. GPU 重写真正耗时的是“把原来的 Chumpy 图逐块替换掉”，而不是装 CUDA。
4. 如果直接全量重写，极容易出现数值不稳定、手部发散、长序列拼接断层等问题。

## 11. 建议的目录落位

建议在这份副本中按下面结构推进：

```text
moshpp/
  chmosh.py
  chmosh_torch.py
  transformed_lm.py
  transformed_lm_torch.py
  models/
    smpl_fast_derivatives.py
    smplx_torch_wrapper.py
  prior/
    gmm_prior_ch.py
    gmm_prior_torch.py
  optim/
    frame_fit_torch.py
    chunk_fit_torch.py
tests/
  test_body_forward_consistency.py
  test_marker_attachment_consistency.py
  test_gmm_prior_consistency.py
  test_stageii_single_frame_smoke.py
docs/
  如何将原生MoSh++改造成GPU版.md
```

## 12. 推荐的第一批开发任务

如果现在正式开始做，建议第一批只做下面 5 项：

1. 新建 `smplx_torch_wrapper.py`
2. 新建 `transformed_lm_torch.py`
3. 新建 `gmm_prior_torch.py`
4. 写 3 个 consistency test
5. 写一个“单帧 stageii torch smoke”脚本

不要第一批就做：

1. 全长视频导出
2. 自定义 dogleg
3. 全模型类型兼容
4. 全量替换入口

## 13. 最终结论

当前公开版原生 `MoSh++` 不能通过“改环境”直接变成 GPU 版，根因是：

1. 核心计算图建立在 `Chumpy` 上。
2. 导数和 prior 建立在 `scipy.sparse` 与 `psbody.smpl.fast_derivatives.smplcpp_chumpy` 上。
3. `stageii` 的优化器是逐帧 CPU `dogleg`。

真正可行的 GPU 方案是：

1. 保留原始 loss 定义。
2. 用 torch 重写 `stageii` 的 body model、marker attachment、prior 和 optimizer。
3. 先做单帧，再做分块，再接回原入口。

如果只想要最短时间内看到速度提升，先做 CPU 分段并行；如果要长期可维护、能持续吃到 GPU，应该启动 `stageii-only torch backend` 这条线。
