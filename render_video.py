import argparse
import inspect
import os.path as osp
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.script_utils import codec_for_video_path, list_stageii_pickles

CAMERA_PRESETS = {
    "frontal": {
        "camera_x": 0.0,
        "camera_y": -3.0,
        "camera_z": 1.0,
        "lookat_x": 0.0,
        "lookat_y": 0.0,
        "lookat_z": 1.0,
        "up_x": 0.0,
        "up_y": 0.0,
        "up_z": 1.0,
    }
}
CAMERA_FIELDS = tuple(CAMERA_PRESETS["frontal"].keys())


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Render stageii pickles into preview mp4 videos without using the legacy "
            "Blender mesh-export pipeline."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dir",
        help="Directory containing stageii pickle files.",
    )
    input_group.add_argument(
        "--input-path",
        help="Single stageii pickle file to render.",
    )
    parser.add_argument(
        "--output-path",
        help="Explicit video output path used only with --input-path.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the SMPL-X model .npz or .pkl file.",
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
        "--camera-preset",
        default="frontal",
        choices=sorted(CAMERA_PRESETS),
        help="Stable preview camera preset. Manual camera args override preset fields.",
    )
    parser.add_argument("--camera-x", type=float, default=None, help="Camera position X.")
    parser.add_argument("--camera-y", type=float, default=None, help="Camera position Y.")
    parser.add_argument("--camera-z", type=float, default=None, help="Camera position Z.")
    parser.add_argument("--lookat-x", type=float, default=None, help="Camera look-at X.")
    parser.add_argument("--lookat-y", type=float, default=None, help="Camera look-at Y.")
    parser.add_argument("--lookat-z", type=float, default=None, help="Camera look-at Z.")
    parser.add_argument("--up-x", type=float, default=None, help="Camera up-vector X.")
    parser.add_argument("--up-y", type=float, default=None, help="Camera up-vector Y.")
    parser.add_argument("--up-z", type=float, default=None, help="Camera up-vector Z.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render videos even if the target file already exists.",
    )
    return parser


def _load_pickle_compat(path):
    _ensure_legacy_pickle_compat()
    with open(path, "rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _coerce_frame_matrix(values, *, name):
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array
    raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(array.shape)}")


def _coerce_optional_expression(values):
    if values is None:
        return None
    return _coerce_frame_matrix(values, name="expression")


def _broadcast_frame_rows(values, num_frames, *, name):
    if values.shape[0] == num_frames:
        return values
    if values.shape[0] == 1:
        return values.repeat(num_frames, axis=0)
    raise ValueError(f"{name} row count {values.shape[0]} does not match num_frames {num_frames}")


def _modern_surface_model_type(stageii_data):
    cfg = stageii_data.get("stageii_debug_details", {}).get("cfg")
    if cfg is None:
        return stageii_data.get("surface_model_type", "smplx")
    if isinstance(cfg, dict):
        return cfg.get("surface_model", {}).get("type", "smplx")
    try:
        return cfg["surface_model"]["type"]
    except Exception:
        return "smplx"


def _coerce_fullpose_to_smplx_layout(fullpose, surface_model_type):
    import numpy as np

    fullpose = np.asarray(fullpose, dtype=np.float32)
    pose_dim = fullpose.shape[1]
    if pose_dim == 165:
        return fullpose
    if surface_model_type == "smplh" and pose_dim == 156:
        zeros = np.zeros((fullpose.shape[0], 9), dtype=fullpose.dtype)
        return np.concatenate([fullpose[:, :66], zeros, fullpose[:, 66:]], axis=1)
    raise ValueError(
        f"Unsupported fullpose shape {tuple(fullpose.shape)} for surface_model_type={surface_model_type}"
    )


def _load_stageii_render_inputs(pkl_path):
    data = _load_pickle_compat(pkl_path)

    if "fullpose" in data and "trans" in data:
        surface_model_type = _modern_surface_model_type(data)
        fullpose = _coerce_frame_matrix(data["fullpose"], name="fullpose")
        betas = _coerce_frame_matrix(data["betas"], name="betas")
        trans = _coerce_frame_matrix(data["trans"], name="trans")
        expression = _coerce_optional_expression(data.get("expression"))
    else:
        surface_model_type = str(data["ps"]["fitting_model"])
        fullpose = _coerce_frame_matrix(data["pose_est_fullposes"], name="pose_est_fullposes")
        betas = _coerce_frame_matrix(data["shape_est_betas"], name="shape_est_betas")
        trans = _coerce_frame_matrix(data["pose_est_trans"], name="pose_est_trans")
        expression = _coerce_optional_expression(data.get("pose_est_exprs"))

    fullpose = _coerce_fullpose_to_smplx_layout(fullpose, surface_model_type)
    num_frames = fullpose.shape[0]
    betas = _broadcast_frame_rows(betas, num_frames, name="betas")
    trans = _broadcast_frame_rows(trans, num_frames, name="trans")
    if expression is not None:
        expression = _broadcast_frame_rows(expression, num_frames, name="expression")

    return {
        "fullpose": fullpose,
        "betas": betas,
        "trans": trans,
        "expression": expression,
    }


def load_vertices(pkl_path, model):
    import torch

    stageii_inputs = _load_stageii_render_inputs(pkl_path)
    fullpose = torch.from_numpy(stageii_inputs["fullpose"]).float()
    shape_components = torch.from_numpy(stageii_inputs["betas"]).float()
    trans = torch.from_numpy(stageii_inputs["trans"]).float()
    num_frames = fullpose.shape[0]
    if stageii_inputs["expression"] is not None:
        expression = torch.from_numpy(stageii_inputs["expression"]).float()
    else:
        expression = (
            shape_components[:, 300:310]
            if shape_components.shape[1] >= 310
            else torch.zeros(num_frames, 10, dtype=shape_components.dtype)
        )

    ret = model(
        global_orient=fullpose[:, :3],
        transl=trans,
        body_pose=fullpose[:, 3:66],
        betas=shape_components[:, :10],
        expression=expression,
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


def _ensure_legacy_pickle_compat():
    legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in legacy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


class HumanBodyPriorRenderModel:
    def __init__(self, body_model, *, num_betas, num_expressions):
        import torch

        self.body_model = body_model
        self.faces = body_model.f.detach().cpu().numpy() if hasattr(body_model.f, "detach") else body_model.f

        with torch.no_grad():
            template = body_model(
                root_orient=torch.zeros(1, 3),
                pose_body=torch.zeros(1, 63),
                pose_hand=torch.zeros(1, 90),
                pose_jaw=torch.zeros(1, 3),
                pose_eye=torch.zeros(1, 6),
                betas=torch.zeros(1, num_betas),
                trans=torch.zeros(1, 3),
                expression=torch.zeros(1, num_expressions),
            )
        self.v_template = template.v[0].detach().cpu().numpy()

    def __call__(self, **kwargs):
        import torch

        pose_hand = torch.cat((kwargs["left_hand_pose"], kwargs["right_hand_pose"]), dim=1)
        pose_eye = torch.cat((kwargs["leye_pose"], kwargs["reye_pose"]), dim=1)
        output = self.body_model(
            root_orient=kwargs["global_orient"],
            pose_body=kwargs["body_pose"],
            pose_hand=pose_hand,
            pose_jaw=kwargs["jaw_pose"],
            pose_eye=pose_eye,
            betas=kwargs["betas"],
            trans=kwargs["transl"],
            expression=kwargs["expression"],
        )
        return SimpleNamespace(vertices=output.v, joints=output.Jtr)


def load_render_model(model_path):
    model_path = Path(model_path)
    if model_path.suffix == ".pkl":
        npz_path = model_path.with_suffix(".npz")
        if npz_path.exists():
            model_path = npz_path

    if model_path.suffix == ".npz":
        from human_body_prior.body_model.body_model import BodyModel

        body_model = BodyModel(
            bm_fname=str(model_path),
            num_betas=10,
            num_expressions=10,
        )
        return HumanBodyPriorRenderModel(body_model, num_betas=10, num_expressions=10)

    import smplx

    ext = model_path.suffix.lstrip(".")
    if ext == "pkl":
        _ensure_legacy_pickle_compat()
    return smplx.SMPLX(str(model_path), use_pca=False, ext=ext or "npz")


def build_preview_jobs(args):
    input_path = getattr(args, "input_path", None)
    output_path = getattr(args, "output_path", None)

    if input_path:
        input_path = str(input_path)
        resolved_output_path = output_path
        if resolved_output_path is None:
            resolved_output_path = build_video_path(input_path, args.input_suffix, args.video_suffix)
        return [(input_path, str(resolved_output_path))]

    if output_path:
        raise ValueError("--output-path requires --input-path for the preview renderer")

    return [
        (pkl_path, build_video_path(pkl_path, args.input_suffix, args.video_suffix))
        for pkl_path in list_stageii_pickles(args.input_dir, args.input_suffix)
    ]


def resolve_camera_config(args):
    camera_values = CAMERA_PRESETS[args.camera_preset].copy()
    for field in CAMERA_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            camera_values[field] = value
    return SimpleNamespace(**camera_values)


def _build_mesh_indices(ti, faces):
    faces_ti = ti.field(dtype=ti.i32, shape=faces.shape)
    faces_ti.from_numpy(faces)
    indices = ti.field(dtype=ti.i32, shape=3 * faces.shape[0])
    for index in range(faces.shape[0]):
        indices[3 * index] = faces_ti[index, 0]
        indices[3 * index + 1] = faces_ti[index, 1]
        indices[3 * index + 2] = faces_ti[index, 2]
    return indices


def render_preview_jobs(args):
    import cv2
    import numpy as np
    import taichi as ti
    from tqdm import tqdm

    if not hasattr(ti, args.arch):
        raise ValueError("Unsupported Taichi arch: {}".format(args.arch))

    ti.init(arch=getattr(ti, args.arch), debug=False)

    model = load_render_model(args.model_path)
    camera_cfg = resolve_camera_config(args)

    window = ti.ui.Window("SOMA Preview Renderer", (args.width, args.height), show_window=False)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    particle_vertices = ti.Vector.field(3, dtype=ti.f32, shape=model.v_template.shape[0])
    indices = _build_mesh_indices(ti, model.faces)

    for pkl_path, video_path in build_preview_jobs(args):
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

            camera.position(camera_cfg.camera_x, camera_cfg.camera_y, camera_cfg.camera_z)
            camera.lookat(camera_cfg.lookat_x, camera_cfg.lookat_y, camera_cfg.lookat_z)
            camera.up(camera_cfg.up_x, camera_cfg.up_y, camera_cfg.up_z)
            scene.set_camera(camera)
            scene.point_light(
                pos=(camera_cfg.camera_x, camera_cfg.camera_y, 0.0),
                color=(1.0, 1.0, 1.0),
            )
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


def render_stageii_preview(
    *,
    input_path,
    output_path=None,
    model_path,
    fps=30,
    width=512,
    height=512,
    arch="gpu",
    camera_preset="frontal",
    input_suffix="_stageii.pkl",
    video_suffix="_stageii.mp4",
    force=False,
    **camera_overrides,
):
    args_dict = {field: None for field in CAMERA_FIELDS}
    args_dict.update(camera_overrides)
    resolved_output_path = (
        str(output_path)
        if output_path is not None
        else build_video_path(str(input_path), input_suffix, video_suffix)
    )
    args = SimpleNamespace(
        input_dir=None,
        input_path=str(input_path),
        output_path=resolved_output_path,
        model_path=str(model_path),
        input_suffix=input_suffix,
        video_suffix=video_suffix,
        fps=fps,
        width=width,
        height=height,
        arch=arch,
        camera_preset=camera_preset,
        force=force,
        **args_dict,
    )
    render_preview_jobs(args)
    return resolved_output_path


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        render_preview_jobs(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
