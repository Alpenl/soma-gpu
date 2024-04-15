import os
import cv2
import smplx
import pickle
import numpy as np
from tqdm import tqdm

import torch
from utils.mesh_io import writePC2, save_obj_mesh
from scipy.spatial.transform import Rotation as R


def rotate_global_orient(global_orient, trans, root_joint):
    orig_shape = global_orient.shape
    global_orient = global_orient.reshape(-1, 3)
    rot_mat = R.from_rotvec(global_orient).as_matrix()
    trans_mat = R.from_rotvec(np.array([-np.pi/2, 0, 0])).as_matrix()
    new_rot_mat = np.einsum("ab,nbc->nac", trans_mat, rot_mat)
    new_rot_vec = R.from_matrix(new_rot_mat).as_rotvec()

    new_trans = np.einsum("ab,nb->na", trans_mat, trans + root_joint[None]) - root_joint[None]

    return new_rot_vec.reshape(orig_shape), new_trans


def load_smpl(pkl_path, start, end, num_transition):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    shape_components = torch.from_numpy(data['betas']).float()[None]
    betas = shape_components[:, :10]
    expression = shape_components[:, 300:310]

    canonical_joints = model(betas=betas)['joints'].detach().cpu().numpy()[0]
    root_joint = canonical_joints[0]

    data['fullpose'] = data['fullpose'][start:end]
    data['trans'] = data['trans'][start:end]
    data['fullpose'][:, :3], data['trans'] = rotate_global_orient(data['fullpose'][:, :3], data['trans'], root_joint)


    # add transition
    transition_poses = np.linspace(np.zeros_like(data['fullpose'][0]), data['fullpose'][0], num_transition+1)
    data['fullpose'] = np.concatenate([transition_poses[:-1], data['fullpose']], axis=0)
    start_trans = np.array([0, -np.min(canonical_joints[:, 1]) + 0.05, 0])
    transition_trans = np.linspace(start_trans, data['trans'][0], num_transition + 1)
    data['trans'] = np.concatenate([transition_trans[:-1], data['trans']], axis=0)

    num_frames = data['fullpose'].shape[0]
    fullpose = torch.from_numpy(data['fullpose']).float()

    trans = torch.from_numpy(data['trans']).float()



    global_orient = fullpose[:, :3]
    body_pose = fullpose[:, 3:66]   # 21 joints
    jaw_pose = fullpose[:, 66:69]
    leye_pose = fullpose[:, 69:72]
    reye_pose = fullpose[:, 72:75]
    left_hand_pose = fullpose[:, 75:120]
    right_hand_pose = fullpose[:, 120:165]

    ret = model(global_orient=global_orient, transl=trans, body_pose=body_pose,
                betas=betas, expression=expression.repeat(num_frames, 1),
                jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose,
                left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)
    vertices = ret.vertices.detach().cpu().numpy()
    return vertices


if __name__ == '__main__':
    start = 6337
    end = 7150
    num_transition = 30

    model = smplx.SMPLX("/home/user416/data/smpl_models/smplx/SMPLX_NEUTRAL.npz", use_pca=False)
    data_dir = f"/home/user416/data/tennis_motion/mosh_results_tracklet/20240205/linxu"
    racket_dir = "/home/user416/data/tennis_motion/mocap_raw/20240205/linxu"

    faces = model.faces
    fname = "1"
    smpl_fname = f"{fname}_stageii.pkl"
    racket_path = os.path.join(racket_dir, f"{fname}_racket.npy")

    racket_data = np.load(racket_path)
    racket_data = racket_data[start:end]
    racket_data = np.concatenate([np.tile(racket_data[:1], (num_transition, 1, 1)), racket_data], axis=0)

    racket_obj_path = os.path.join(racket_dir, f"{fname}_racket.obj")
    racket_pc2_path = os.path.join(racket_dir, f"{fname}_racket_{start}-{end}.pc2")
    racket_faces = np.array([[0, 1, 2], [1, 2, 3]])
    save_obj_mesh(racket_obj_path, racket_data[0], faces)
    writePC2(racket_pc2_path, racket_data)

    data_path = os.path.join(data_dir, smpl_fname)
    save_path = os.path.join(data_dir, smpl_fname.replace('_stageii.pkl', f'_{start}-{end}.pc2'))

    print("Start processing", data_path)
    vertices = load_smpl(data_path, start, end, num_transition=num_transition)
    save_obj_mesh(save_path.replace('.pc2', '.obj'), vertices[0], faces)
    writePC2(save_path, vertices)


