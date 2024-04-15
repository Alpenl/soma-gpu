import smplx
import pickle

import torch
from utils.mesh_io import writePC2, save_obj_mesh


def load_smpl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    num_frames = data['fullpose'].shape[0]
    fullpose = torch.from_numpy(data['fullpose']).float()
    shape_components = torch.from_numpy(data['betas']).float()[None]
    trans = torch.from_numpy(data['trans']).float()

    betas = shape_components[:, :10]
    expression = shape_components[:, 300:310]

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
    model = smplx.SMPLX("/home/user416/data/smpl_models/smplx/SMPLX_NEUTRAL.npz", use_pca=False)
    fname = 'out_test'
    data_path = f"/home/user416/data/tennis_motion/mosh_results_tracklet/20240205/linxu/{fname}_stageii.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    num_frames = data['fullpose'].shape[0]
    fullpose = torch.from_numpy(data['fullpose']).float()
    shape_components = torch.from_numpy(data['betas']).float()[None]
    trans = torch.from_numpy(data['trans']).float()

    betas = shape_components[:, :10]
    expression = shape_components[:, 300:310]

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
    faces = model.faces

    save_obj_mesh(f"/home/user416/data/tennis_motion/mosh_results_tracklet/20240205/linxu/{fname}_stageii.obj", vertices[0], faces)
    writePC2(f"/home/user416/data/tennis_motion/mosh_results_tracklet/20240205/linxu/{fname}_stageii.pc2", vertices)