import os.path

import smplx
from utils.mesh_io import load_obj_mesh, save_obj_mesh, writePC2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def smooth_iter(x, alpha=0.1):
    # padding
    x2 = np.concatenate([x[0:1], x, x[-2:-1]], 0)
    dst = (x2[:-2] + x2[2:]) / 2
    x = (dst - x) * alpha + x
    return x


def find_all_children(parents, idx):
    children = [idx]
    for i in range(len(parents)):
        if parents[i] == idx:
            children.extend(find_all_children(parents, i))
    return children


if __name__ == '__main__':
    model = smplx.SMPLH("/home/user416/data/smpl_models/smplh/SMPLH_NEUTRAL.pkl", use_pca=False)
    data_dir = "/home/user416/data/tennis_motion/20240415_demo"

    v0 = model.v_template.detach().numpy()
    rl_joints = find_all_children(model.parents, 2)  # right_leg
    la_joints = find_all_children(model.parents, 16)  # left_arm

    sw = model.lbs_weights.numpy()

    rl_vertices = np.sum(sw[:, rl_joints], 1) > 0.7
    la_vertices = np.sum(sw[:, la_joints], 1) > 0.7

    rl_color = np.ones([rl_vertices.shape[0], 3]) * 0.7
    la_color = np.ones([la_vertices.shape[0], 3]) * 0.7

    rl_color[rl_vertices] = [0.7, 0, 0]
    la_color[la_vertices] = [0.7, 0, 0]

    import trimesh
    rl_mesh = trimesh.Trimesh(vertices=v0, faces=model.faces, vertex_colors=rl_color)
    la_mesh = trimesh.Trimesh(vertices=v0, faces=model.faces, vertex_colors=la_color)
    rl_mesh.export(os.path.join(data_dir, "rl_mesh.ply"))
    la_mesh.export(os.path.join(data_dir, "la_mesh.ply"))
    exit()

    data_path = os.path.join(data_dir, "results.npz")
    c = np.load(data_path)

    body_pose = R.from_matrix(c['poses_body'][:, :21].reshape([-1, 3, 3])).as_rotvec().reshape([-1, 21*3])

    for _ in range(15):
        body_pose = smooth_iter(body_pose)

    body_pose = torch.from_numpy(body_pose).float()
    # global_orient = torch.zeros_like(body_pose[:, :3])
    left_hand_pose = torch.zeros_like(body_pose[:, :45])
    right_hand_pose = torch.zeros_like(body_pose[:, :45])

    global_orient = R.from_matrix(c['poses_root_world']).as_rotvec()
    global_orient = torch.from_numpy(global_orient).float()

    transl = torch.from_numpy(c['trans_world']).float()

    betas = np.tile(np.mean(c['betas'], 0, keepdims=True), (body_pose.shape[0], 1))
    betas = torch.from_numpy(betas).float()

    ret = model(body_pose=body_pose, global_orient=global_orient, transl=transl,
                left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                betas=betas*0)
    vertices = ret.vertices.detach().numpy()
    vertices[:, :, 1] -= np.min(vertices[:, :, 1], axis=1, keepdims=True)
    writePC2(os.path.join(data_dir, "smplh.pc2"), vertices)

    # save_obj_mesh("smplh.obj", ret.vertices[0].detach().numpy(), model.faces)