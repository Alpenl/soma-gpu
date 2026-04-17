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
3. 运行soma的`convert_tennis.py`，会自动运行soma和mosh++，得到最终的smplx结果，储存在`ROOT/mosh_results_tracklet/[session]/[subject]/[seq]_stageii.pkl`。如需在主流程里直接继续导出 mesh 和 preview 视频，可追加 `--export-artifacts`：
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
默认会同时发现 `.c3d` 和 `.mcp` 输入；如 stageii pickle 里的模型路径来自旧机器，会优先回退到当前 `support_files/[surface_model.type]/[gender]/model.npz|model.pkl`。
5. 若只想对已有 `*_stageii.pkl` 单独补跑导出，也可运行：
````
python export_stageii_artifacts.py \
  --input-pkl ROOT/mosh_results_tracklet/[session]/[subject]/[seq]_stageii.pkl \
  --model-path support_files/smplx/[gender]/model.npz
````
若只需要 OBJ/PC2，可继续使用 `save_smplx_verts.py`；若只需要 preview MP4，可使用 `render_video.py`。
