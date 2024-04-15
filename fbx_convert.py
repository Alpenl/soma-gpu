import os
import maya.cmds as cmds
import numpy as np
# import maya.standalone
# maya.standalone.initialize()
"""
need to install numpy by running:
C:/Program Files/Autodesk/Maya2025/bin/mayapy.exe -m pip install numpy
"""

def get_hierarchy(node):
    # 用于存储层级信息的字典
    hierarchy = {}

    # 检查节点是否存在
    if not cmds.objExists(node):
        print(f"Error: No object matches name: {node}")
        return hierarchy

    # 获取给定节点的子节点
    children = cmds.listRelatives(node, children=True, fullPath=True) or []

    # 遍历子节点
    for child in children:
        # 递归调用，构建层级结构
        hierarchy[child] = get_hierarchy(child)
    return hierarchy


def get_marker_names(scene_hierarchy, node_name):
    marker_names = []
    for k in scene_hierarchy[node_name].keys():
        if cmds.nodeType(k) == "transform" and k != node_name + '|Position':
            marker_names.append(k)
    return marker_names


def get_end_frame():
    animCurves = cmds.ls(type='animCurve')

    # 初始化起始和结束时间
    startTime = None
    endTime = None

    # 遍历所有的动画曲线
    for curve in animCurves:
        # 获取当前曲线的时间范围
        curveStartTime = cmds.findKeyframe(curve, which='first')
        curveEndTime = cmds.findKeyframe(curve, which='last')

        # 更新总的起始和结束时间
        if startTime is None or curveStartTime < startTime:
            startTime = curveStartTime
        if endTime is None or curveEndTime > endTime:
            endTime = curveEndTime
    return int(np.ceil(endTime))

data_dir = "C:/data/tennis_motion"
# data_dir = "/home/user416/data/tennis_motion/mocap_raw/20240205/linxu"
fnames = [k for k in os.listdir(data_dir) if k.endswith('.fbx')]

for fname in fnames:
    save_path = os.path.join(data_dir, fname.replace('.fbx', '.npy'))
    racket_save_path = os.path.join(data_dir, fname.replace('.fbx', '_racket.npy'))
    if os.path.exists(os.path.join(data_dir, save_path)) and os.path.exists(os.path.join(data_dir, racket_save_path)):
        continue

    cmds.file(new=True, force=True)
    cmds.file(os.path.join(data_dir, fname), i=True)

    # 设置播放速率为固定的30fps
    cmds.playbackOptions(playbackSpeed=1, maxPlaybackSpeed=1)
    # 设置时间轴的实际帧率为30fps
    cmds.currentUnit(time='ntsc')

    root_nodes = cmds.ls(assemblies=True, long=True)
    scene_hierarchy = {}
    for node in root_nodes:
        scene_hierarchy[node] = get_hierarchy(node)

    human_names = ["|LinxuTest", "|WenjiaTest", "|ZhouTest"]
    human_name = None
    for name in human_names:
        if name in scene_hierarchy:
            human_name = name
    racket_name = "|tennis_racket"

    # get marker object
    human_marker_names = get_marker_names(scene_hierarchy, human_name)
    racket_marker_names = get_marker_names(scene_hierarchy, racket_name)

    start_frame = 0
    end_frame = get_end_frame()
    cmds.playbackOptions(animationStartTime=0, animationEndTime=end_frame)

    human_res, racket_res = [], []
    for frame in range(int(start_frame), int(end_frame) + 1):
    # for frame in range(int(start_frame), 100):
        cmds.currentTime(frame,edit=True)

        human_res_frame, racket_res_frame = [], []
        for o in human_marker_names:
            J_posX = cmds.getAttr(o + ".translateX")
            J_posY = cmds.getAttr(o + ".translateY")
            J_posZ = cmds.getAttr(o + ".translateZ")
            human_res_frame.append([J_posX/100, J_posY/100, J_posZ/100])
        for o in racket_marker_names:
            J_posX = cmds.getAttr(o + ".translateX")
            J_posY = cmds.getAttr(o + ".translateY")
            J_posZ = cmds.getAttr(o + ".translateZ")
            racket_res_frame.append([J_posX/100, J_posY/100, J_posZ/100])
        human_res_frame = np.array(human_res_frame, dtype=np.float32)
        racket_res_frame = np.array(racket_res_frame, dtype=np.float32)
        human_res.append(human_res_frame)
        racket_res.append(racket_res_frame)
    human_res = np.array(human_res)
    racket_res = np.array(racket_res)

    np.save(save_path, human_res)
    np.save(racket_save_path, racket_res)