import argparse
import os.path as osp
import pickle

from utils.script_utils import codec_for_video_path, list_stageii_pickles


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render every *_stageii.pkl in a directory into videos."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing stageii pickle files.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the SMPL-X model npz file.",
    )
    parser.add_argument(
        "--input-suffix",
        default="_stageii.pkl",
        help="Suffix used to discover stageii pickle files.",
    )
    parser.add_argument(
        "--video-suffix",
        default="_stageii.mp4",
        help="Suffix used for the rendered videos.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output videos.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output video width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output video height.",
    )
    parser.add_argument(
        "--arch",
        default="gpu",
        help="Taichi backend to use, for example gpu or cuda.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render videos even if the target file already exists.",
    )
    return parser


def load_vertices(pkl_path, model):
    import torch

    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)

    num_frames = data["fullpose"].shape[0]
    fullpose = torch.from_numpy(data["fullpose"]).float()
    shape_components = torch.from_numpy(data["betas"]).float()[None]
    trans = torch.from_numpy(data["trans"]).float()

    ret = model(
        global_orient=fullpose[:, :3],
        transl=trans,
        body_pose=fullpose[:, 3:66],
        betas=shape_components[:, :10],
        expression=shape_components[:, 300:310].repeat(num_frames, 1),
        jaw_pose=fullpose[:, 66:69],
        leye_pose=fullpose[:, 69:72],
        reye_pose=fullpose[:, 72:75],
        left_hand_pose=fullpose[:, 75:120],
        right_hand_pose=fullpose[:, 120:165],
    )
    return ret.vertices.detach().cpu().numpy()


def build_video_path(pkl_path, input_suffix, video_suffix):
    if pkl_path.endswith(input_suffix):
        return pkl_path[: -len(input_suffix)] + video_suffix
    return osp.splitext(pkl_path)[0] + video_suffix


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    import cv2
    import numpy as np
    import smplx
    import taichi as ti
    from tqdm import tqdm

    if not hasattr(ti, args.arch):
        parser.error("Unsupported Taichi arch: {}".format(args.arch))

    ti.init(arch=getattr(ti, args.arch), debug=False)

    model = smplx.SMPLX(args.model_path, use_pca=False)
    window = ti.ui.Window("SOMA Renderer", (args.width, args.height), show_window=False)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    particle_vertices = ti.Vector.field(
        3, dtype=ti.f32, shape=model.v_template.shape[0]
    )
    faces = model.faces
    faces_ti = ti.field(dtype=ti.i32, shape=faces.shape)
    faces_ti.from_numpy(faces)
    indices = ti.field(dtype=ti.i32, shape=3 * faces.shape[0])
    for index in range(faces.shape[0]):
        indices[3 * index] = faces_ti[index, 0]
        indices[3 * index + 1] = faces_ti[index, 1]
        indices[3 * index + 2] = faces_ti[index, 2]

    for pkl_path in list_stageii_pickles(args.input_dir, args.input_suffix):
        video_path = build_video_path(pkl_path, args.input_suffix, args.video_suffix)
        if osp.exists(video_path) and not args.force:
            continue

        vertices = load_vertices(pkl_path, model)
        video_writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*codec_for_video_path(video_path)),
            args.fps,
            (args.width, args.height),
        )

        for frame_vertices in tqdm(vertices, desc=osp.basename(pkl_path)):
            particle_vertices.from_numpy(frame_vertices)

            camera.position(0.0, -3.0, 1.0)
            camera.lookat(0.0, 0.0, 1.0)
            camera.up(0.0, 0.0, 1.0)
            scene.set_camera(camera)
            scene.point_light(pos=(0.0, -3.0, 0.0), color=(1.0, 1.0, 1.0))
            scene.ambient_light((0.5, 0.5, 0.5))
            scene.mesh(
                particle_vertices,
                indices=indices,
                two_sided=True,
                show_wireframe=False,
            )

            canvas.scene(scene)
            image = window.get_image_buffer_as_numpy()
            image = cv2.cvtColor(
                (image[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            image = np.transpose(image, (1, 0, 2))
            image = np.flip(image, axis=0)
            video_writer.write(image)

        video_writer.release()


if __name__ == "__main__":
    main()
