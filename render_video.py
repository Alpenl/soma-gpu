import os
import cv2
import smplx
import pickle
import taichi as ti
import numpy as np
from tqdm import tqdm

import torch
from utils.mesh_io import writePC2, save_obj_mesh

ti.init(arch=ti.cuda, debug=False)
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
    ti.init(arch=ti.cuda, debug=False)
    model = smplx.SMPLX("/home/user416/data/smpl_models/smplx/SMPLX_NEUTRAL.npz", use_pca=False)
    data_dir = f"/home/user416/data/tennis_motion/mosh_results_tracklet/20240205/wenjia"

    window = ti.ui.Window('Window Title', (512, 512), show_window=False)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()


    # Particle state
    particle_vertices = ti.Vector.field(3, dtype=ti.f32, shape=model.v_template.shape[0])

    faces = model.faces
    faces_ti = ti.field(dtype=ti.i32, shape=faces.shape)
    faces_ti = ti.field(dtype=ti.i32, shape=faces.shape)
    faces_ti.from_numpy(faces)
    indices = ti.field(int, shape=3 * faces.shape[0])
    for i in range(faces.shape[0]):
        indices[3 * i] = faces_ti[i, 0]
        indices[3 * i + 1] = faces_ti[i, 1]
        indices[3 * i + 2] = faces_ti[i, 2]

    fnames = [k for k in os.listdir(data_dir) if k.endswith('_stageii.pkl')]
    for fname in fnames:
        data_path = os.path.join(data_dir, fname)
        video_path = os.path.join(data_dir, fname.replace('_stageii.pkl', '_stageii.mp4'))
        if os.path.exists(video_path):
            continue

        print("Start processing", data_path)
        vertices = load_smpl(data_path)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (512, 512))

        for v in tqdm(vertices):
            particle_vertices.from_numpy(v)

            camera.position(0.0, -3, 1)
            camera.lookat(0.0, 0.0, 1)
            camera.up(0, 0, 1)
            scene.set_camera(camera)

            scene.point_light(pos=(0, -3, 0), color=(1, 1, 1))
            scene.ambient_light((0.5, 0.5, 0.5))

            # scene.lines(frame_vertices, color=(1, 0, 0), width=1)
            scene.mesh(particle_vertices,
                       indices=indices,
                       two_sided=True,
                       show_wireframe=False)
            # scene.particles(particle_vertices, radius=0.01, color=(0, 0, 1))

            canvas.scene(scene)

            img = window.get_image_buffer_as_numpy()

            img = cv2.cvtColor((img[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img = np.transpose(img, (1, 0, 2))
            img = np.flip(img, axis=0)
            video_writer.write(img)

        video_writer.release()
