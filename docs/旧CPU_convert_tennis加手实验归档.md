# 旧 CPU `convert_tennis.py` 加手实验归档

## 目的

归档 `/home/alpen/DEV/soma/convert_tennis.py` 这套旧 CPU SOMA + MoSh 配置，在 `haonan-73` 当前 `10m30s` 基准上复现与排障的全过程，重点记录：

- 旧 `convert_tennis.py` 的基线配置
- 为了让 `haonan-73` 可用而做的 **方案一** 修正
- `optimize_fingers=true` 加手版本的实际配置
- 各次失败的确定性原因
- 当前已确认的结论和后续使用建议

## 基线脚本配置

来源文件：

```text
/home/alpen/DEV/soma/convert_tennis.py
```

旧脚本里和这次复现直接相关的配置如下：

### SOMA 段

- `soma_expr_id = 'V48_02_SuperSet'`
- `soma_data_id = 'OC_05_G_03_real_000_synt_100'`
- `soma_cfg`：
  - `soma.batch_size = 256`
  - `dirs.support_base_dir = /home/user416/project/soma/support_files`
  - `mocap.unit = 'mm'`
  - `save_c3d = True`
  - `keep_nan_points = True`
  - `remove_zero_trajectories = True`

### MoSh 段

- `moshpp.verbosity = 1`
- `moshpp.stagei_frame_picker.type = 'random'`
- `dirs.support_base_dir = support_base_dir`

注意：

- 旧 `convert_tennis.py` **没有显式打开** `optimize_fingers`
- old MoSh 默认配置里 `optimize_fingers = false`
- 所以旧 CPU 基线更接近 **body-first / weak-hand** 路径，而不是“真的认真解手”

## 本次基准数据

主样本：

```text
/home/alpen/DEV/tmp/native_zh73_smoke20/input/wolf001/4090-haonan-73.mcp
```

本次 old CPU 对齐复现不是整段 `10m30s`，而是先裁前 `20s / 600` 帧做 apples-to-apples 对照。

切片工作目录：

```text
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1_fingers
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1_fingers_bg
```

## 为什么原始 old CPU 路线会坏掉

### 现象

直接沿用 old SOMA 标注输出时，最后 old MoSh 只拿到了 `4` 个可用 body labels，结果不成人形。

坏输入：

```text
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu/work_py37_mm/soma_labeled_mocap_tracklet/haonan73_20s/wolf001/4090-haonan-73_20s_trimmed.pkl
```

### 证据

坏输入 `labels` 的唯一标签只有 `5` 个，其中真正有意义的只有 `4` 个：

- `LCLAV`
- `LKNI`
- `RBUST`
- `RCLAV`
- `nan`

具体统计：

- `labels_count = 73`
- `unique = 5`
- `labels_perframe_total = 43800`
- `top_counts = [('nan', 41400), ('LCLAV', 600), ('LKNI', 600), ('RBUST', 600), ('RCLAV', 600)]`

这也是 old MoSh 日志里会出现“可用 marker 只有 4 个”的根因。

### 结论

这不是 old CPU MoSh 自身先天坏，而是 **old SOMA 在 `haonan-73` 上标签塌缩**，导致下游只剩 4 点。

## 方案一：保留 old CPU MoSh 配置，替换成可靠 73 点输入

### 核心思路

保留 `/home/alpen/DEV/soma/convert_tennis.py` 那套 old CPU MoSh 配置与 old `moshpp` 路线，但不再使用 old SOMA 崩掉的 4 点标签输入，而是换成可靠的 `73` 点 labeled mocap pkl 和配套 marker layout。

### 73 点可靠输入

修正版 labeled mocap pkl：

```text
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1/work_convertcfg73/soma_labeled_mocap_tracklet/haonan73_20s/wolf001/4090-haonan-73_20s_trimmed.pkl
```

配套 custom marker layout：

```text
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1/work_convertcfg73/haonan73_20s/haonan73_20s_73custom_smplx.json
```

### 标签组成

修正版 `73` 点标签是完整对应的：

- `labels_count = 73`
- `unique = 73`
- `labels_perframe_total = 43800`
- 所有标签每帧均有 `600` 次出现

完整标签集合包括：

```text
ARIEL, LFHD, LBHD, RFHD, RBHD, C7, T10, CLAV, STRN,
LFSH, LBSH, LUPA, LELB, LELBIN, LFRM, LIWR, LOWR,
LTHMB, LFIN, LTHM3, LTHM6, LIDX3, LIDX6, LMID0, LMID6,
LRNG3, LRNG6, LPNK3, LPNK6,
RFSH, RBSH, RUPA, RELB, RELBIN, RFRM, RIWR, ROWR,
RTHMB, RFIN, RTHM3, RTHM6, RIDX3, RIDX6, RMID0, RMID6,
RRNG3, RRNG6, RPNK3, RPNK6,
LFWT, MFWT, RFWT, LBWT, MBWT, RBWT,
LTHI, LKNE, LKNI, LSHN, LANK, LHEE, LMT5, LMT1, LTOE,
RTHI, RKNE, RKNI, RSHN, RANK, RHEE, RMT5, RMT1, RTOE
```

### old CPU no-fingers 基线结果

方案一 no-fingers 成功产物：

```text
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1/work_convertcfg73/haonan73_20s/wolf001/male_stagei.pkl
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1/work_convertcfg73/haonan73_20s/wolf001/4090-haonan-73_20s_trimmed_stageii.pkl
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1/work_convertcfg73/haonan73_20s/wolf001/4090-haonan-73_20s_trimmed_stageii_front.mp4
```

日志里的基线关键信息：

- `optimize_fingers: False`
- `Available marker types: {'body': 53, 'finger_left': 10, 'finger_right': 10}. Total: 73 markers.`
- `Estimating for #latent markers: 73`

实际耗时：

- `stagei = 0:24:06.124794`
- `stageii = 0:07:33.373214`

## 加手版本的实际配置

### 启动脚本

```text
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1_fingers_bg/start_oldcpu_fingers_bg.py
```

### 配置来源

加手版不是从零手写整套 old `OmegaConf`，而是：

1. 读取方案一 no-fingers 成功产物：

```text
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1/work_convertcfg73/haonan73_20s/wolf001/4090-haonan-73_20s_trimmed_stageii.pkl
```

2. 从其中的：

```python
stageii_debug_details["cfg"]
```

拷出 old CPU 成功配置

3. 只覆盖以下关键项：

- `cfg["moshpp"]["optimize_fingers"] = True`
- `cfg["moshpp"]["betas_fname"] = None`
- `cfg["moshpp"]["v_template_fname"] = None`
- `cfg["moshpp"]["stagei_frame_picker"]["stagei_mocap_fnames"] = None`
- `cfg["mocap"]["exclude_markers"] = None`
- `cfg["mocap"]["exclude_marker_types"] = None`
- `cfg["mocap"]["only_markers"] = None`
- `cfg["dirs"]["work_base_dir"] = /home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1_fingers_bg/work_convertcfg73`
- `cfg["dirs"]["stagei_fname"] = .../male_stagei.pkl`
- `cfg["dirs"]["stageii_fname"] = .../4090-haonan-73_20s_trimmed_stageii.pkl`
- `cfg["dirs"]["log_fname"] = .../4090-haonan-73_20s_trimmed.log`

### 运行时确认

加手版日志已经确认：

- `optimize_fingers: True`
- `optimizing for fingers. dof_per_hand = 24`
- `Available marker types: {'body': 53, 'finger_left': 10, 'finger_right': 10}. Total: 73 markers.`
- `Estimating for #latent markers: 73`

### 运行输出目录

```text
/home/alpen/DEV/tmp/convert_tennis_haonan73_20s_cpu_scheme1_fingers_bg/work_convertcfg73/haonan73_20s/wolf001
```

## 失败原因归档

### 失败 1：直接沿用 old SOMA 输出，标签塌缩成 4 点

#### 现象

渲染出来“不成人形”。

#### 根因

old SOMA 在 `haonan-73` 上只保住了 `LCLAV / LKNI / RBUST / RCLAV` 四个真实标签，其余 `69` 个槽位全部塌成 `nan`。

#### 后果

old MoSh 实际只在解 `4` 个 body 点，不可能正常还原人体。

### 失败 2：第一次 manual old CPU 复现时单位用错成 `m`

#### 现象

第一次正面渲染出来像一团，尺度明显错乱。

#### 根因

当时没有严格沿用 `/home/alpen/DEV/soma/convert_tennis.py` 里的：

```text
mocap.unit = mm
```

而是误用成了 `m`。

#### 证据

错误单位版 `trans` 范围：

```text
min = 46.82587707340981
max = 602.561099620465
```

修正成 `mm` 后的范围：

```text
min = -0.05214709803941058
max = 0.9233075752589441
```

#### 结论

这次是配置错误，不是算法问题。

### 失败 3：第一次 fingers 版交互运行没有跑完

#### 现象

任务推进到 `stagei Step 3/4`，但没有产出 `male_stagei.pkl`。

#### 根因

这是**执行/会话层中断**，不是 73 点或 finger 优化本身报错。该次中断后没有留下新的 Python traceback。

#### 结论

不能据此判断 old CPU 加手算法本身不工作，只能说明“前台交互式运行不稳定”。

### 失败 4：第一次伪后台 `nohup ... &` 方式并不可靠

#### 现象

日志只写到 very early startup，随后进程消失。

#### 根因

在当前 agent 执行层里，裸 `nohup ... &` 子进程会在外层命令返回后被回收，不是真正稳定的 detached 托管方式。

#### 结论

这个失败属于**启动方式问题**，不是 old CPU fingers 求解逻辑问题。

### 失败 5：第一次 `tmux` 托管启动缺少 old `soma` 环境

#### 现象

任务一启动就退出。

#### 根因

`tmux` 会话里没有补齐：

```text
cd /home/alpen/DEV/soma
PYTHONPATH=/home/alpen/DEV/soma
```

所以脚本直接报：

```text
ModuleNotFoundError: No module named 'moshpp'
```

#### 结论

这是运行环境问题，不是 fingers 数学问题。

### 失败 6：修正环境后的 `tmux` 版在 `stagei Step 4/4` 被 `earlyoom` 杀掉

#### 现象

任务已经推进到：

```text
Step 4/4 : Opt. wt_anneal_factor = 0.12, wt_data = 0.12, wt_poseB = 378.08, wt_poseH = 0.38
```

但 `male_stagei.pkl` 仍未落盘，任务随后消失。

#### 系统证据

系统日志原文：

```text
Apr 20 16:10:49 HKU-4090D earlyoom[1422]: mem avail:  2977 of 31098 MiB ( 9.57%), swap free:    0 of 2047 MiB ( 0.01%)
Apr 20 16:10:49 HKU-4090D earlyoom[1422]: low memory! at or below SIGTERM limits: mem 10.00%, swap 10.00%
Apr 20 16:10:49 HKU-4090D earlyoom[1422]: sending SIGTERM to process 3028373 uid 1000 "python3.7": badness 813, VmRSS 7280 MiB
Apr 20 16:10:49 HKU-4090D earlyoom[1422]: process exited after 0.1 seconds
```

#### 结论

这是**明确的系统内存压力 kill**：

- 不是代码 traceback
- 不是 marker layout 配错
- 不是 `optimize_fingers=true` 逻辑崩溃

而是 `stagei Step 4/4` 高内存阶段被 `earlyoom` 直接 `SIGTERM`

## 当前结论

截至当前归档时点：

- old CPU no-fingers 方案一已经成功跑通，可作为可靠基线
- old CPU fingers 版已经证明：
  - `73` 点对应关系没问题
  - `optimize_fingers=true` 会真实进入 old `stagei`
  - 任务可以推进到 `stagei Step 4/4`
- 但 **还没有成功产出** fingers 版的：
  - `male_stagei.pkl`
  - `stageii.pkl`

当前 fingers 版尚未成功的主因不是算法错误，而是：

1. 启动/托管方式踩坑
2. 最终稳定运行后，被系统 `earlyoom` 在高内存阶段杀掉

## 后续建议

如果还要继续把 old CPU fingers 版跑完，最优先的不是改算法，而是先解决资源问题：

1. 先停掉当前机器上最重的内存/GPU 进程
2. 确保 `swap` 不再满载
3. 再用 `tmux` 或其他真正 detached 的托管方式重跑

否则这条任务很可能再次在 `stagei Step 4/4` 附近被 `earlyoom` 杀掉。
