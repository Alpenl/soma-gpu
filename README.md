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
批量模式会递归发现匹配的 `*_stageii.pkl`，按相对目录结构镜像写出 OBJ/PC2；如果目录下没有任何匹配结果，脚本会直接报错而不是静默成功。若只需要 preview MP4，可使用 `render_video.py`。

若要在复核真实 `.mcp -> stageii -> mesh` 候选时，把单样本 `stageii` 质量摘要和 baseline 的 mesh-space 对比收敛到同一份 JSON，也可直接运行：
````
python benchmark_stageii_public.py \
  --input ROOT/mosh_results_tracklet/[session]/[subject]/candidate_stageii.pkl \
  --mesh-reference ROOT/mosh_results_tracklet/[session]/[subject]/baseline_stageii.pkl \
  --mesh-support-base-dir support_files \
  --output ROOT/benchmarks/candidate_vs_baseline.json
````
此时报告除了现有 `quality.marker_residual_l2` / `trans_jitter_l2` / `chunk_seam_*` 摘要外，还会在 `quality.mesh_compare` 下追加 baseline 与 candidate 的 `reference` / `candidate` / `frame_delta_l2` mesh-space 摘要。对于 `stageii.pkl` 输入通常不需要显式传 `--mesh-chunk-size/--mesh-chunk-overlap`；只有直接比较裸 `pc2/pc16` 缓存时才需要覆盖。

若想直接走官方 `run_moshpp_once(cfg)` 单序列入口，并在同一条命令里产出 `stageii.pkl + benchmark JSON`，可使用：
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

`--preset real-mcp-baseline` 会先注入当前已验证的 corrected real `.mcp` torch baseline 参数：
`moshpp.optimize_fingers=true`、`runtime.sequence_chunk_size=32`、`runtime.sequence_chunk_overlap=4`、`runtime.sequence_seed_refine_iters=5`、`runtime.refine_lr=0.05`、`runtime.sequence_lr=0.05`、`runtime.sequence_max_iters=30`。如果要在此基础上做单变量 sweep，继续追加 `--cfg key=value` 即可；`--cfg` 会覆盖同名 preset 项，因此不需要每次重打一整串 baseline override。

如果想直接复现当前保留的 translation-friendly 候选，可把 preset 换成 `real-mcp-transvelo100-seedvelowindow`；它会在同一组 corrected baseline 参数上再叠加 `runtime.sequence_transl_velocity=100` 与 `runtime.sequence_boundary_transl_velocity_reference=true`。这样就能配合现有 `--mesh-reference` / `--benchmark-output` 直接做 baseline vs candidate 的 stageii / mesh 对照，而不需要再手工维护第二串高权重 velocity overrides。

当 baseline 和 candidate 共用同一个 `--work-base-dir` 时，建议同时给 `--output-suffix`，例如 `_baseline` / `_seedvelowindow`。这个后缀会在 preset/`--cfg` 解析完成后追加到 `mocap.basename`，从而让默认生成的 `stageii/log` 文件名分开；如果你已经显式用 `--cfg mocap.basename=...` 固定了名字，suffix 会继续在那个 basename 后面追加。
