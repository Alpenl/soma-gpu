## SOMA: Solving Optical Marker-Based MoCap Automatically, ICCV'21

## Installation

**在2080ti上已经有安装好的conda环境： `/home/user416/miniconda3/envs/soma`**
 
<details open>
<summary>Click to expand 详细安装流程</summary>
SOMA is originally developed in Python 3.7, PyTorch 1.8.2 LTS, for Ubuntu 20.04.2 LTS. 
Below we prepare the python environment using [Anaconda](https://www.anaconda.com/products/individual), 
however, we opt for a simple pip package manager for installing dependencies.

````
sudo apt install libatlas-base-dev
sudo apt install libpython3.7
sudo apt install libtbb2

conda create -n soma python=3.7 
conda install -c conda-forge ezc3d

pip3 install torch==1.8.2+cu102 torchvision==0.9.2+cu102 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

````
ezc3d installation is currently not supported by pip.

Assuming that you have already cloned this repository to your local drive 
go to the root directory of SOMA code and run
````
pip install -r requirements.txt
python setup.py develop
````
Copy the precompiled 
[smpl-fast-derivatives](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=smpl-fast-derivatives.tar.bz2) 
into your python site-packages folder, i.e. ````anaconda3/envs/soma/lib/python3.7/site-packages````.
The final directory should look like ````anaconda3/envs/soma/lib/python3.7/site-packages/psbody/smpl````.

Install the psbody.mesh library following the instructions in [https://github.com/MPI-IS/mesh](https://github.com/MPI-IS/mesh).
Hint: clone the mesh repository and run the following from the anaconda environment:  ````python setup.py install ````.

~~To use the rendering capabilities first install an instance of Blender-2.83 LTS on your machine.
Afterward uncompress contents of the precompiled 
[bpy-2.83](https://download.is.tue.mpg.de/download.php?domain=soma&sfile=blender/bpy-2.83-20200908.tar.bz2) 
into your python site-packages folder, i.e. ````anaconda3/envs/soma/lib/python3.7/site-packages````.~~ 

Last but not least, the current SOMA code relies on [MoSh++](https://github.com/nghorbani/moshpp) mocap solver. 
Please install MoSh++ following the guidelines in its repository.

</details>

## Directory Structure
2080ti机器上， `ROOT`为`/home/user416/data/tennis_motion`

文件层次从上到下为 `每次mocap session` -> `每次mocap session的每个subject` -> `subject的多个动作序列`

注意，同一subject在不同mocap session的shape是不同的，因为身材会有变化。
````
ROOT
--mocap_raw/[session]/[subject]
----[seq].fbx
----[seq].npy
----[seq].c3d
----[seq]_racket.npy
--soma_labeled_mocap_tracklet
----soma中间结果，不太用管
--mosh_results_tracklet/[session]/[subject]
----[seq]_stageii.pkl
--processed
````

## Pipeline
1. Pre-process data using Maya  
用maya的pip安装numpy   
`C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe -m pip install numpy`  
打开Maya，右下角script窗口，将`fbx_convert.py`复制进去运行，需要更改中间的`data_dir`路径为你的`ROOT`路径（见上一节）
结果会保存在`ROOT/mocap_raw/[session]/[subject]/[seq].npy`和`[seq]_racket.npy`
2. 切回soma的环境，运行`npy2c3d.py`，将npy文件转换为c3d文件
3. 运行soma的`convert_tennis.py`，会自动运行soma和mosh++，得到最终的smplx结果，储存在`ROOT/mosh_results_tracklet/[session]/[subject]/[seq]_stageii.pkl`。默认会自动发现 `ROOT/mocap_raw/[session]/[subject]` 下的 `.c3d` 和 `.mcp` 输入；如需强制只跑某一种扩展，可显式传 `--mocap-ext .c3d` 或 `--mocap-ext .mcp`。若同一 `subject/sequence` 同时保留了 `.c3d` 和 `.mcp` 两份别名输入，脚本会直接报错，避免生成同名输出时互相覆盖。如需在主流程里直接继续导出 mesh 和 preview 视频，可追加 `--export-artifacts`：
````
python convert_tennis.py \
  --dataset [session] \
  --mocap-base-dir ROOT/mocap_raw \
  --soma-work-base-dir ROOT \
  --export-artifacts
````
关于pkl文件内容解析，可参考`save_smplx_verts.py`
4. 若不需要 SOMA，只想直接对 `ROOT/mocap_raw/[session]/[subject]` 下的 `.c3d` 或 `.mcp` 跑 MoSh++，并在主流程尾部顺带导出 OBJ/PC2/preview MP4，可运行：
````
python convert_mosh.py \
  --dataset [session] \
  --mocap-base-dir ROOT/mocap_raw \
  --work-base-dir ROOT \
  --export-artifacts
````
默认会同时发现 `.c3d` 和 `.mcp` 输入；如果同一 `subject/sequence` 同时存在这两种别名文件，脚本会直接报错，避免 direct MoSh 生成同名结果时互相覆盖。如 stageii pickle 里的模型路径来自旧机器，会优先回退到当前 `support_files/[surface_model.type]/[gender]/model.npz|model.pkl`。
5. 若只想对已有 `*_stageii.pkl` 单独补跑导出，也可运行：
````
python export_stageii_artifacts.py \
  --input-pkl ROOT/mosh_results_tracklet/[session]/[subject]/[seq]_stageii.pkl \
  --support-base-dir support_files
````
此时脚本会优先从 stageii pickle 内嵌的 `stageii_debug_details.cfg.surface_model.fname` 解析模型路径；若该路径来自旧机器，则会优先回退到当前 `support_files/[surface_model.type]/[gender]/model.npz|model.pkl`。如需强制指定单一模型，仍可显式传 `--model-path`。

若想递归扫描整个 session / 目录下的现有 stageii 结果并批量补导出，可运行：
````
python export_stageii_artifacts.py \
  --input-dir ROOT/mosh_results_tracklet/[session] \
  --support-base-dir support_files \
  --fname-filter swing
````
批量模式会递归发现匹配的 `*_stageii.pkl` 并逐个在原路径旁生成 OBJ/PC2/preview MP4；如果目录下没有任何匹配结果，脚本会直接报错而不是静默成功。

`convert_tennis.py --export-artifacts` 和 `convert_mosh.py --export-artifacts` 现在也采用同一套批量导出逻辑：若 `fname_filter`/dataset 最终没有匹配到任何 stageii 结果，会直接报错，避免“主流程成功但没有任何导出产物”的静默空跑。

若只需要 OBJ/PC2，可继续使用 `save_smplx_verts.py`：
````
python save_smplx_verts.py \
  --input-pkl ROOT/mosh_results_tracklet/[session]/[subject]/[seq]_stageii.pkl \
  --support-base-dir support_files
````
`save_smplx_verts.py` 与 `export_stageii_artifacts.py` 一样，默认会优先从 stageii pickle 解析模型路径，并在需要时回退到当前 `support_files/[surface_model.type]/[gender]/model.npz|model.pkl`；只有想强制指定单一模型时才需要显式传 `--model-path`。如需批量补导出 mesh 而不想渲染 mp4，可直接用 mesh-only 批量入口：
````
python save_smplx_verts.py \
  --input-dir ROOT/mosh_results_tracklet/[session] \
  --support-base-dir support_files \
  --output-dir ROOT/mesh_exports/[session] \
  --fname-filter swing
````
批量模式会递归发现匹配的 `*_stageii.pkl`，按相对目录结构镜像写出 OBJ/PC2；如果目录下没有任何匹配结果，脚本会直接报错而不是静默成功。`--fname-filter` 现在也只允许和 `--input-dir` 一起使用，避免在单文件 `--input-pkl` 模式下被静默忽略。若只需要 preview MP4，可使用 `render_video.py`。

若要在复核真实 `.mcp -> stageii -> mesh` 候选时，把单样本 `stageii` 质量摘要和 baseline 的 mesh-space 对比收敛到同一份 JSON，也可直接运行：
````
python benchmark_stageii_public.py \
  --input ROOT/mosh_results_tracklet/[session]/[subject]/candidate_stageii.pkl \
  --mesh-reference ROOT/mosh_results_tracklet/[session]/[subject]/baseline_stageii.pkl \
  --mesh-support-base-dir support_files \
  --output ROOT/benchmarks/candidate_vs_baseline.json
````
此时报告除了现有 `quality.marker_residual_l2` / `trans_frame_delta_l2` / `pose_frame_delta_l2` / `trans_jitter_l2` / `pose_jitter_l2` / `chunk_seam_*` 摘要外，还会在 `quality.mesh_compare` 下追加 baseline 与 candidate 的 `reference` / `candidate` / `frame_delta_l2` mesh-space 摘要；如果 `--mesh-reference` 本身也是 `stageii.pkl`，同一份 JSON 里还会额外写出 `quality.reference_stageii_quality` 与 `quality.reference_stageii_delta`，把 baseline 的 stageii 质量摘要和 candidate-reference 的关键统计量差值一起收进来。`reference_stageii_delta` 目前固定汇总 `mean/p90/max`，数值按 `candidate - reference` 计算，因此对现有 residual / frame-delta / jitter / seam 指标来说，负值表示 candidate 更低、更接近我们想要的方向。同一场景下，`speed` 节点现在也会并列给出 `reference_stageii_elapsed_s` 与 `reference_stageii_elapsed_delta_s`，把 baseline 的 solver 耗时和 `candidate - reference` 的 elapsed 差值一起收进这份 candidate JSON，方便 low/mid/high 候选直接在一份报告里做质量与速度取舍。另外 `quality.chunk_seam_transl_jump_over_trans_frame_delta_ratio` 与 `quality.chunk_seam_pose_jump_over_pose_frame_delta_ratio` 现在也会直接给出 seam 相对普通 frame-to-frame motion 的比值：大于 `1` 表示 seam jump 比对应统计量下的正常运动幅度更大，能更快识别“虽然绝对数值不大，但对这条序列来说 seam 仍偏突兀”的候选。对于 `stageii.pkl` 输入通常不需要显式传 `--mesh-chunk-size/--mesh-chunk-overlap`；只有直接比较裸 `pc2/pc16` 缓存时才需要覆盖，且 `--mesh-chunk-size` 必须是正整数、`--mesh-chunk-overlap` 必须是非负整数。
direct benchmark CLI 现在也会把 mesh compare 相关参数收紧到真正需要它们的场景：`--mesh-chunk-size`、`--mesh-chunk-overlap`、`--mesh-support-base-dir` 都要求同时存在 `--mesh-reference`；其中 `--mesh-chunk-overlap` 还必须与 `--mesh-chunk-size` 成对出现，并要求 `--mesh-chunk-size > 0`、`--mesh-chunk-overlap >= 0`。同时 benchmark run-count 也会在 CLI 边界直接校验：`--warmup-runs >= 0`、`--measured-runs > 0`。未提供 reference、只单独传 overlap，或传入非正/负值时，这些参数都会直接报错，而不会再静默忽略。若提供了 `--mesh-reference` 但没显式传 `--mesh-support-base-dir`，CLI 才会默认回退到 `support_files`。
若不显式传 `--output`，`benchmark_stageii_public.py` 现在也会默认把报告写到输入同目录下的 `*_benchmark.json`：例如 `candidate_stageii.pkl -> candidate_benchmark.json`。`--output` 仅用于覆盖这个默认落点；如果它最终指回了当前 `--input` 或 `--mesh-reference`，CLI 会直接报错，避免 benchmark JSON 覆盖 stageii/reference 资产。
如果当前主线只是想更快地复核 `.mcp -> mesh` 候选，而不想把时间花在 preview/mp4/artifact 这些非主线 speed 探针上，可额外加 `--lean-benchmark`。这个模式会跳过 `preview_vertex_decode_ms`、`mesh_export_ms`、`mp4_render_ms`、`artifact_bundle_export_ms` 四类可选 speed benchmark，只保留核心的 ingest latency / repeatability / quality / mesh compare 摘要；当前 real `.mcp` baseline/candidate 迭代更适合默认带上它。
如果 benchmark/mesh compare 参数本身不合法，例如只传了 `--mesh-chunk-overlap` 却没配 `--mesh-chunk-size`，或者把 `--mesh-reference` 指回了当前 `--input` / 当前输出的同一个 `stageii.pkl`，`benchmark_stageii_public.py`、`run_stageii_torch_official.py` 和 `run_stageii_torch_pair.py` 现在都会直接以 CLI error 退出，而不是把内部 `ValueError` 栈追踪直接打到终端或生成伪零差值报告。即使不是 CLI 参数，而是 `stageii.pkl` 自己嵌的 runtime chunk 元数据出现了 `sequence_chunk_size<=0` / `sequence_chunk_overlap<0` 这类坏值，benchmark / mesh compare 现在也会直接报错，而不会静默改写后继续出报告。

若想直接走官方 `run_moshpp_once(cfg)` 单序列入口，并在同一条命令里产出 `stageii.pkl`，再按需顺带导出 OBJ/PC2 与 benchmark JSON，可使用：
````
python run_stageii_torch_official.py \
  --mocap-fname ROOT/mocap_raw/[session]/[subject]/[seq].mcp \
  --support-base-dir support_files \
  --work-base-dir ROOT/work \
  --preset real-mcp-baseline \
  --output-suffix _baseline \
  --cfg surface_model.gender=male \
  --benchmark-output ROOT/benchmarks/[seq]_torch.json
````
该脚本只做薄编排：基础路径参数会直接落到 `MoSh.prepare_cfg(...)`，其余 candidate-specific 参数继续通过 repeatable `--cfg key=value` 透传；默认会在官方入口结束后立即对生成的 `stageii.pkl` 复用 `benchmark_stageii_public.py` 同一套质量/mesh 对比口径。如只想先产出 `stageii.pkl`、暂时不跑 benchmark，可加 `--skip-benchmark`。
如果当前是在 real `.mcp -> mesh` 主线上快速筛候选，通常也建议同时加 `--lean-benchmark`，这样 single runner 会继续产出同一份 quality / mesh compare JSON，但不再额外测 preview/mp4/artifact 速度。
如果 `--mocap-fname` 不是 `dataset/session/...` 这种默认目录布局，single runner 也仍可参与这套静态路径规划：显式追加 `--cfg mocap.ds_name=...` 与 `--cfg mocap.session_name=...` 即可；若你本来就想把输出收敛到更自定义的子目录，也可以直接给 `--cfg dirs.session_subject_subfolders=...`。这些 override 会直接影响默认 `stageii.pkl` / `*_benchmark.json` / `--mesh-reference-output-suffix` 的推导结果，因此 flat mocap 路径下也不需要退回手工拼全量输出路径。
若当前主线想直接闭环到 mesh，可在同一条命令上再加 `--export-mesh`。默认会复用 `save_smplx_verts.export_stageii_meshes(...)`，把 OBJ/PC2 写到生成的 `stageii.pkl` 同目录；如需集中导出到独立目录，可追加 `--mesh-output-dir ROOT/mesh_exports/[session]/[subject]`。`--mesh-output-dir` 现在必须和 `--export-mesh` 一起用，避免被静默忽略。`--mesh-support-base-dir` 现在同时服务于 mesh 导出和 benchmark 里的 mesh compare；若不显式传，则默认回退到 `--support-base-dir`。
若 mesh 导出阶段本身因为模型资产缺失、输出写盘失败，或 render/model 依赖链缺失而抛 `ImportError` / `ModuleNotFoundError`，single runner 现在也会把这些异常统一收口成 CLI error，而不是直接打 Python 栈。
若不显式传 `--benchmark-output`，runner 也会默认把报告写到同目录下的 `*_benchmark.json`：例如 `foo_stageii.pkl -> foo_benchmark.json`。`--benchmark-output` 现在只用于覆盖这个默认落点，而不是决定“是否写盘”。若它最终指回了当前这次输出的 `stageii.pkl`、benchmark 用到的 `--mesh-reference`，或本次 `--export-mesh` 计划写出的 OBJ/PC2，single runner 会直接报错；当这些路径能静态推导出来时，这个错误会在 `MoSh.prepare_cfg(...)` 之前触发。
对当前由官方入口产出的新格式 `stageii.pkl`，benchmark JSON 现在还会额外带上 `speed.stageii_elapsed_s`，直接复用 `stageii_debug_details.stageii_elapsed_time`。如果当前 benchmark 还同时拿另一个 `stageii.pkl` 当 `--mesh-reference`，同一份 candidate JSON 里还会额外补上 `speed.reference_stageii_elapsed_s` 与 `speed.reference_stageii_elapsed_delta_s`。这样 baseline/candidate 的 solver 实际耗时不需要再从终端日志或第二份 baseline JSON 里手抄。
若 benchmark 这段本身因为 mesh compare 参数不合法、reference 路径/模型解析失败等原因出错，single runner 现在也会把异常收口成 CLI error，避免直接抛 Python 栈。
若 official-run 段本身因为 `MoSh.prepare_cfg(...)` 配置错误、缺字段 `KeyError`、`run_moshpp_once(cfg)` 运行期文件错误、依赖链缺失而抛 `ImportError` / `ModuleNotFoundError`，或最终没有真正产出预期的 `stageii.pkl` 而失败，single runner 现在也会直接以 CLI error 退出，而不是把原始 Python 栈追踪打到终端。
当 single runner 被 pair runner 静默复用时，它现在除了校验 hidden 的 `--expected-stageii-path` 之外，也会校验 hidden 的 `--expected-benchmark-output`、`--expected-mesh-obj-path`、`--expected-mesh-pc2-path`。只要当前 `stageii.pkl` 落点能被静态推导出来，这些 internal contract 现在就会在 `MoSh.prepare_cfg(...)` / `run_moshpp_once(cfg)` 之前先生效；若当前输出路径只能到运行后才最终显式暴露，runner 仍会在真正写 benchmark JSON 或导出 OBJ/PC2 前再做一次同样的校验。这样无论是 internal planner 自己算错，还是底层实际落点和 pair runner 的 planned 路径漂开，命令都会在覆盖 baseline 产物之前直接报错。
这些 hidden internal args 现在也不再允许 silent no-op：`--expected-benchmark-output` 必须和 benchmark 一起启用，`--expected-mesh-obj-path` / `--expected-mesh-pc2-path` 必须和 `--export-mesh` 一起启用，且 OBJ/PC2 两条 mesh contract 必须成对提供。
single runner 现在还会继续校验 helper 返回 payload 本身：`export_stageii_meshes(...)` 返回的 `obj_path/pc2_path`，以及 `write_benchmark_report(...)` 返回的 `artifact.report_path`，都必须仍然等于它刚刚请求写盘的路径。这样下游拿到的 JSON/path payload 不会和实际写盘目标脱钩。
若显式传了 `--skip-benchmark`，single runner 现在也会同步拒绝所有 benchmark-only 参数：例如 `--benchmark-output`、`--warmup-runs`、`--measured-runs`、`--mesh-reference` / `--mesh-reference-output-suffix`、`--mesh-chunk-*`。这样不会再出现“命令看起来要求了 compare/report，但其实整段 benchmark 被跳过”的 silent no-op。
只要 benchmark 真正开启，single runner 现在也会在 CLI 边界直接校验 `--warmup-runs >= 0`、`--measured-runs > 0`，避免先白跑 official entry，再在 `run_public_stageii_benchmark(...)` 里才因为非法计数失败。
`--lean-benchmark` 也属于 benchmark-only 参数：只有 benchmark 真正开启时才允许传，用来显式跳过 preview/mp4/artifact speed 探针。
若 benchmark 仍开启，但你没有提供 `--mesh-reference` / `--mesh-reference-output-suffix`，single runner 现在也会拒绝纯 mesh-compare 参数：`--mesh-chunk-size`、`--mesh-chunk-overlap`，以及在既不导 mesh、也不做 mesh compare 时单独传入的 `--mesh-support-base-dir`。即使已经提供了 mesh reference，`--mesh-chunk-overlap` 现在也必须和 `--mesh-chunk-size` 成对出现，且 `--mesh-chunk-size > 0`、`--mesh-chunk-overlap >= 0`。这样不会再把 overlap-only 或非正/负值配置拖到更深层 mesh helper 才报错。

`--preset real-mcp-baseline` 会先注入当前已验证的 corrected real `.mcp` torch baseline 参数：
`moshpp.optimize_fingers=true`、`runtime.sequence_chunk_size=32`、`runtime.sequence_chunk_overlap=4`、`runtime.sequence_seed_refine_iters=5`、`runtime.refine_lr=0.05`、`runtime.sequence_lr=0.05`、`runtime.sequence_max_iters=30`。如果要在此基础上做单变量 sweep，继续追加 `--cfg key=value` 即可；`--cfg` 会覆盖同名 preset 项，因此不需要每次重打一整串 baseline override。
对于 direct single-run 的 `.mcp` 命令，如果你故意不传 `--preset`，现在也必须显式补齐这三条 corrected-baseline anchor：`--cfg moshpp.optimize_fingers=true --cfg runtime.refine_lr=0.05 --cfg runtime.sequence_lr=0.05`。否则 CLI 会直接拒绝运行，避免继续把已确认会掉进 bad replay cfg 的裸跑路径误当成官方 code-first 主线入口。

如果想直接复现一个更保守的 low-risk translation 候选，可把 preset 换成 `real-mcp-transvelo10-seedvelowindow`；它会在同一组 corrected baseline 参数上再叠加 `runtime.sequence_transl_velocity=10` 与 `runtime.sequence_boundary_transl_velocity_reference=true`，对应当前已确认可稳定复现、且五个主质量指标都仍朝正确方向轻微改善的低权重 candidate。

如果想直接复现一个介于 low-risk 与 high-weight 之间的中档 translation 候选，可把 preset 换成 `real-mcp-transvelo32-seedvelowindow`；它同样复用 corrected baseline，并固定 `runtime.sequence_transl_velocity=32` 与 `runtime.sequence_boundary_transl_velocity_reference=true`。当前 real 300f 复核显示：它相对 `transvelo10` 在 `pose_jitter`、`trans_jitter`、`chunk_seam_pose`、`chunk_seam_transl` 和 mesh-space seam/accel 上都更健康，而 `marker_residual` 只停留在近似持平的噪声量级，因此更适合作为一个显式的中档 preset，而不是继续靠手写 `--cfg` 保存。

如果想直接复现当前保留的高权重 translation-friendly 候选，可把 preset 换成 `real-mcp-transvelo100-seedvelowindow`；它会在同一组 corrected baseline 参数上再叠加 `runtime.sequence_transl_velocity=100` 与 `runtime.sequence_boundary_transl_velocity_reference=true`。这样就能配合现有 `--mesh-reference` / `--benchmark-output` 直接做 baseline vs candidate 的 stageii / mesh 对照，而不需要再手工维护第二串高权重 velocity overrides。

当 baseline 和 candidate 共用同一个 `--work-base-dir` 时，建议同时给 `--output-suffix`，例如 `_baseline` / `_seedvelowindow`。这个后缀会在 preset/`--cfg` 解析完成后追加到 `mocap.basename`，从而让默认生成的 `stageii/log` 文件名分开；如果你已经显式用 `--cfg mocap.basename=...` 固定了名字，suffix 会继续在那个 basename 后面追加。

如果当前是在跑 candidate，而 baseline 也正好是同一组 `mocap/support/work` 参数下、只差一个输出 suffix，那么不必再手工把 baseline `stageii.pkl` 路径抄进 `--mesh-reference`。可以直接在 candidate 命令上改用：
````
python run_stageii_torch_official.py \
  --mocap-fname ROOT/mocap_raw/[session]/[subject]/[seq].mcp \
  --support-base-dir support_files \
  --work-base-dir ROOT/work \
  --preset real-mcp-transvelo100-seedvelowindow \
  --output-suffix _seedvelowindow \
  --mesh-reference-output-suffix _baseline \
  --cfg surface_model.gender=male \
  --benchmark-output ROOT/benchmarks/[seq]_candidate_vs_baseline.json
````
`--mesh-reference-output-suffix` 会按同一套 `preset < --cfg < dedicated args` 解析逻辑静态推导 baseline 的默认 `stageii.pkl` 路径，不会再额外调用一次 `MoSh.prepare_cfg(...)`；因此适合“baseline/candidate 共用一个 work dir，只靠 basename suffix 分名”的主线复现，也能避免为参考路径规划重复触发 resolver。如果 baseline 根本不在同一命名规则下，再继续显式传 `--mesh-reference`。现在当你同时通过 `--cfg dirs.stageii_fname=...` 显式改了输出路径时，runner 会直接拒绝 `--mesh-reference-output-suffix`，避免再按错误的 basename 规则去推导 baseline 路径。并且无论你是显式传 `--mesh-reference`，还是让 `--mesh-reference-output-suffix` 去推导，只要它最终指回了当前这次输出的 `stageii.pkl`，runner 都会直接报错，避免 candidate benchmark 退化成“拿自己跟自己比”的伪零差值结果；当当前输出路径能静态推导出来时，这个错误现在会在 `MoSh.prepare_cfg(...)` / `run_moshpp_once(cfg)` 之前就触发，不再白跑一整次 official entry。
同样地，如果当前 candidate 命令的 `--mocap-fname` 只是一个扁平文件路径，而不是天然带 `dataset/session` 目录层级，只要显式补上 `mocap.ds_name` / `mocap.session_name`，或者直接给 `dirs.session_subject_subfolders`，`--mesh-reference-output-suffix` 仍会按这些 override 去规划 baseline reference，而不是退化成必须手工传完整 `--mesh-reference`。

如果当前主线就是“先跑 corrected baseline，再跑一个候选并立刻做 stageii / mesh 对照”，可以直接改用成对入口：
````
python run_stageii_torch_pair.py \
  --mocap-fname ROOT/mocap_raw/[session]/[subject]/[seq].mcp \
  --support-base-dir support_files \
  --work-base-dir ROOT/work \
  --cfg surface_model.gender=male \
  --lean-benchmark \
  --export-mesh \
  --mesh-output-dir ROOT/mesh_exports/[session]/[subject]
````
这个脚本会顺序调用现有 `run_stageii_torch_official.py` 两次：
- baseline 侧默认使用 `--preset real-mcp-baseline --output-suffix _baseline`，并默认只产 `stageii.pkl`，不重复跑 standalone benchmark
- candidate 侧默认使用 `--preset real-mcp-transvelo32-seedvelowindow --output-suffix _candidate`，并自动把 baseline 那次真实返回的 `stageii_path` 显式传给 `--mesh-reference`，因此输出的 candidate benchmark JSON 会直接带上 baseline 的 stageii / mesh 对照摘要；即使 baseline 侧额外用了 `--baseline-cfg mocap.basename=...` 这类只影响路径命名的覆盖，也不会再被 candidate 侧的配置重推导错。这样默认主线会直接落在当前 real 300f 复核里更平衡的中档 Pareto 点，而不是默认走高权重 tradeoff。
- 如果想在同一套 pair 入口下复核更激进的高权重版本，可显式加 `--candidate-preset real-mcp-transvelo100-seedvelowindow`；这样 baseline/candidate 的编排与 benchmark contract 不变，只把 candidate runtime 切到高权重 translation-friendly candidate。
- 如果想在同一套 pair 入口下复核低权重版本，可显式加 `--candidate-preset real-mcp-transvelo10-seedvelowindow`；这样 baseline/candidate 的编排与 benchmark contract 不变，只把 candidate runtime 切到 low-risk candidate。
- 若加了 `--export-mesh`，pair runner 会把同一套 mesh 导出参数同时透传给 baseline 和 candidate；由于两侧默认 `mocap.basename` suffix 不同，即使共用一个 `--mesh-output-dir`，OBJ/PC2 也会自动分名，不需要再手工分两个导出目录。若没开 `--export-mesh` 却传了 `--mesh-output-dir`，pair runner 现在会直接报错，而不是静默忽略。
- 若加了 `--lean-benchmark`，pair runner 会把该标志透传给所有真正开启 benchmark 的 underlying single run：candidate 默认 benchmark 会启用，baseline 只有在显式给了 `--baseline-benchmark-output` 时才启用。这样 baseline/candidate 对照仍会保留核心 quality / mesh compare JSON，但不会再为非主线 preview/mp4/artifact speed 探针额外花时间。
- pair runner 对 benchmark run-count 现在也会先做自己的 CLI 校验：`--warmup-runs >= 0`、`--measured-runs > 0`；非法值会在 baseline 启动前直接报错，而不是等 underlying runner 或 benchmark helper 更晚失败。
- pair runner 现在也会在自己的 CLI 边界收紧 mesh compare chunk 参数：`--mesh-chunk-overlap` 必须和 `--mesh-chunk-size` 成对出现，且 `--mesh-chunk-size > 0`、`--mesh-chunk-overlap >= 0`；否则命令会在 baseline 启动前直接报错，而不是等 candidate benchmark 深层调用 `utils.mesh_compare` 时才失败。
- pair runner 现在还会在真正调用 baseline / candidate 两次 single runner 之前，先按与 single runner 相同的 `preset < --cfg < dedicated args` 规则静态推导两侧默认 `stageii.pkl` 输出路径；如果 `--baseline-cfg` / `--candidate-cfg` 里的 `mocap.basename`、`dirs.stageii_fname` 或相关路径覆盖最终把两侧指到同一个 `stageii.pkl`，命令会直接报错，而不会先跑一轮再把 baseline 自己覆盖掉。
- 这层静态推导不要求 `--mocap-fname` 一定已经处在 `dataset/session/...` 目录里；如果输入只是一个 flat `.mcp/.c3d` 文件，也可以通过共享或侧向 `--cfg mocap.ds_name=... --cfg mocap.session_name=...`，或者直接给 `dirs.session_subject_subfolders=...`，把 baseline/candidate 的 hidden `--expected-stageii-path` / `--expected-benchmark-output` 规划成稳定路径。这样 pair runner 在非标准目录布局下也仍能保住同一套 output contract 护栏。
- 若开启了 `--export-mesh --mesh-output-dir`，pair runner 现在还会继续检查 baseline/candidate 推导出的 OBJ/PC2 落点是否会同名；即使两侧 `stageii.pkl` 在不同目录，只要 basename 一样、最终会把导出写进同一个 OBJ/PC2 路径，也会直接报错，避免 mesh 主线复核时把 baseline 导出物悄悄被 candidate 覆盖。
- 若 baseline 也开启了 standalone benchmark，pair runner 现在不只会比较 baseline/candidate 两侧的 benchmark JSON 落点；只要 `--baseline-benchmark-output` 最终指向 candidate 的显式或默认 `*_benchmark.json`、candidate 的计划 `stageii.pkl`，或开启 `--export-mesh` 后 candidate 计划写出的 OBJ/PC2，命令都会直接报错，避免 baseline benchmark 先把 candidate 资产占掉。
- 即使 baseline 不跑 standalone benchmark，candidate 这侧的 benchmark JSON 现在也不能反过来占用 baseline 的计划 `stageii.pkl` 或 baseline 计划写出的 OBJ/PC2；如果 `--candidate-benchmark-output` 显式或默认落点会撞到 baseline 资产，pair runner 会在 baseline 启动前直接报错。
- 即使静态预检没有命中，pair runner 现在也会在 baseline 真正跑完后，再把 baseline payload 里实际返回的 `stageii_path`、`mesh_export.obj/pc2`、以及 baseline benchmark `report_path` 与 candidate 的计划落点做一次二次比对；如果 underlying single runner 的真实落点和静态预判漂移到会覆盖 candidate 的 benchmark/stageii/mesh 产物，命令会在 candidate 启动前直接中止，而不是继续把 baseline 产物覆盖掉。
- 这层二次比对里，baseline 实际 `stageii_path` 现在不只会对 candidate 的 `stageii.pkl` / benchmark JSON 做比较；开启 `--export-mesh` 时，也会继续对 candidate 计划 OBJ/PC2 做比较，避免 baseline stageii 路径漂到 candidate mesh 落点后等第二次导出才把 baseline 文件覆写掉。
- 这层 baseline-actual 二次比对现在也会把 baseline 实际 `stageii_path` / `mesh_export.obj/pc2` 与 candidate 计划 benchmark JSON 一并比较；因此就算 underlying single runner 的真实返回路径漂到了 candidate benchmark 落点，也会在第二次调用前被挡住，不会先把 baseline 资产覆写成 report JSON。
- 除了上面的 baseline-actual 二次比对，pair runner 现在也会把 planned benchmark / mesh 输出路径连同 `stageii.pkl` 计划路径一起，作为 hidden internal arg 下推给 underlying single runner：
  - `--expected-stageii-path`
  - `--expected-benchmark-output`
  - `--expected-mesh-obj-path`
  - `--expected-mesh-pc2-path`
  这样若 single runner 在真正写 benchmark JSON / OBJ / PC2 前发现输出路径已经偏离 pair runner 的静态计划，就会在覆盖发生前直接失败，而不是等 pair runner 事后从 payload 才发现。
- pair runner 现在还会把请求过的产物当作强 contract：
  - 开了 `--export-mesh`，baseline/candidate 两侧都必须从 underlying single runner 返回 `mesh_export.obj_path` 与 `mesh_export.pc2_path`
  - baseline 显式开了 standalone benchmark 或 candidate 默认 benchmark 路径缺失 `benchmark.artifact.report_path` 时，也会直接报错
  - 这样不会再出现“命令成功退出，但 pair payload 其实没带齐 mesh/report 产物路径”的 silent success
- 若 baseline / candidate 任一侧的 single runner 运行失败、抛缺字段 `KeyError`，或者返回 payload 缺失 `stageii_path`，pair runner 现在也会统一以 CLI error 退出，而不是把 `KeyError` / `ValueError` / `FileNotFoundError` 栈直接打到终端。

若要在同一条命令里继续做 sweep，可用：
- `--candidate-cfg key=value`：只改 candidate
- `--baseline-cfg key=value`：只改 baseline
- `--cfg key=value`：两侧共享

如果确实也想落 baseline 的 standalone benchmark JSON，再额外传 `--baseline-benchmark-output ...`；否则 pair runner 会默认跳过 baseline benchmark，避免把同一轮 mesh compare 多算一遍。
