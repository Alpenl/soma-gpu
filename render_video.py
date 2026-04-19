import argparse
import os
import inspect
import os.path as osp
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.script_utils import codec_for_video_path, list_stageii_pickles

WORLD_FRONTAL_CAMERA_PRESET = "frontal"
SUBJECT_FRONTAL_CAMERA_PRESET = "subject-frontal"
DEFAULT_CAMERA_PRESET = SUBJECT_FRONTAL_CAMERA_PRESET
SUBJECT_CAMERA_DISTANCE = 3.0
SUBJECT_CAMERA_LOOKAT_HEIGHT = 0.15

CAMERA_PRESETS = {
    WORLD_FRONTAL_CAMERA_PRESET: {
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
CAMERA_FIELDS = tuple(CAMERA_PRESETS[WORLD_FRONTAL_CAMERA_PRESET].keys())
CAMERA_CHOICES = sorted(tuple(CAMERA_PRESETS) + (SUBJECT_FRONTAL_CAMERA_PRESET,))


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
        "--supersample",
        type=int,
        default=1,
        help="Render internally at N times the output resolution and downsample before encoding. Use 2 for final videos.",
    )
    parser.add_argument(
        "--ffmpeg-crf",
        type=int,
        default=None,
        help="Use ffmpeg/libx264 CRF encoding instead of cv2 VideoWriter. Try 16 for high-quality MP4.",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="medium",
        help="ffmpeg/libx264 preset used when --ffmpeg-crf is set, for example medium or slow.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to the ffmpeg executable used when --ffmpeg-crf is set.",
    )
    parser.add_argument(
        "--neutral-face",
        action="store_true",
        help="Render with neutral face by zeroing jaw/eye pose and expression.",
    )
    parser.add_argument(
        "--zero-jaw",
        action="store_true",
        help="Zero the SMPL-X jaw and eye pose channels before rendering.",
    )
    parser.add_argument(
        "--zero-expression",
        action="store_true",
        help="Zero expression coefficients before rendering.",
    )
    parser.add_argument(
        "--arch",
        default="gpu",
        help="Taichi backend to use, for example gpu or cuda.",
    )
    parser.add_argument(
        "--camera-preset",
        default=DEFAULT_CAMERA_PRESET,
        choices=CAMERA_CHOICES,
        help="Preview camera preset. subject-frontal follows the actor's facing direction and manual camera args override preset fields.",
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


def load_stageii_render_inputs(pkl_path):
    return _load_stageii_render_inputs(pkl_path)


def load_vertices(
    pkl_path,
    model,
    *,
    stageii_inputs=None,
    neutral_face=False,
    zero_jaw=False,
    zero_expression=False,
):
    import torch

    stageii_inputs = stageii_inputs or _load_stageii_render_inputs(pkl_path)
    fullpose = torch.from_numpy(stageii_inputs["fullpose"]).float()
    shape_components = torch.from_numpy(stageii_inputs["betas"]).float()
    trans = torch.from_numpy(stageii_inputs["trans"]).float()
    num_frames = fullpose.shape[0]
    if stageii_inputs["expression"] is not None:
        expression = torch.from_numpy(stageii_inputs["expression"]).float()
    else:
        expression = torch.zeros(num_frames, 10, dtype=shape_components.dtype)

    if neutral_face:
        zero_jaw = True
        zero_expression = True
    if zero_jaw:
        fullpose = fullpose.clone()
        fullpose[:, 66:75] = 0.0
    if zero_expression:
        expression = torch.zeros_like(expression)

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


def _apply_axis_angle(axis_angle, vector):
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-8:
        return vector.copy()
    axis = axis_angle / theta
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    return (
        vector * cos_theta
        + np.cross(axis, vector) * sin_theta
        + axis * np.dot(axis, vector) * (1.0 - cos_theta)
    )


def _estimate_subject_front_vector(fullpose):
    fullpose = np.asarray(fullpose, dtype=np.float32)
    local_front = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    front_xy_vectors = []
    for axis_angle in fullpose[:, :3]:
        world_front = _apply_axis_angle(axis_angle, local_front)
        world_front[2] = 0.0
        norm = float(np.linalg.norm(world_front))
        if norm > 1e-8:
            front_xy_vectors.append(world_front / norm)

    if not front_xy_vectors:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)

    mean_front = np.mean(np.asarray(front_xy_vectors, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(mean_front))
    if norm <= 1e-8:
        return np.array([0.0, -1.0, 0.0], dtype=np.float32)
    return mean_front / norm


def _subject_frontal_camera_values(stageii_inputs):
    trans = np.asarray(stageii_inputs["trans"], dtype=np.float32)
    fullpose = np.asarray(stageii_inputs["fullpose"], dtype=np.float32)
    lookat_xy = np.median(trans[:, :2], axis=0)
    lookat_z = max(
        1.0,
        float(np.median(trans[:, 2]) + SUBJECT_CAMERA_LOOKAT_HEIGHT),
    )
    front = _estimate_subject_front_vector(fullpose)
    camera_xy = lookat_xy + front[:2] * SUBJECT_CAMERA_DISTANCE
    return {
        "camera_x": float(camera_xy[0]),
        "camera_y": float(camera_xy[1]),
        "camera_z": lookat_z,
        "lookat_x": float(lookat_xy[0]),
        "lookat_y": float(lookat_xy[1]),
        "lookat_z": lookat_z,
        "up_x": 0.0,
        "up_y": 0.0,
        "up_z": 1.0,
    }


def resolve_camera_config(args, *, stageii_inputs=None):
    if args.camera_preset == SUBJECT_FRONTAL_CAMERA_PRESET:
        if stageii_inputs is None:
            raise ValueError(
                "subject-frontal camera preset requires stageii pose/trans inputs"
            )
        camera_values = _subject_frontal_camera_values(stageii_inputs)
    else:
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


def _configure_taichi_env():
    os.environ.setdefault("TI_ENABLE_PYBUF", "0")


def _create_preview_renderer(ti, *, num_vertices, faces, width, height):
    window = ti.ui.Window("SOMA Preview Renderer", (width, height), show_window=False)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    particle_vertices = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)
    indices = _build_mesh_indices(ti, np.asarray(faces, dtype=np.int32))
    return SimpleNamespace(
        window=window,
        canvas=canvas,
        scene=scene,
        camera=camera,
        particle_vertices=particle_vertices,
        indices=indices,
    )


class _OpenCvVideoSink:
    def __init__(self, cv2, output_path, *, fps, width, height):
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec_for_video_path(str(output_path))),
            fps,
            (width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {output_path}")

    def write(self, image):
        self.writer.write(image)

    def release(self):
        self.writer.release()


class _FfmpegVideoSink:
    def __init__(self, output_path, *, fps, width, height, crf, preset, ffmpeg_path):
        cmd = [
            ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", str(preset),
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, image):
        if image.dtype != np.uint8:
            raise TypeError(f"ffmpeg video frames must be uint8, got {image.dtype}")
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def release(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        ret = self.process.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with status {ret}")


def _make_video_sink(cv2, output_path, *, fps, width, height, ffmpeg_crf=None, ffmpeg_preset="medium", ffmpeg_path="ffmpeg"):
    if ffmpeg_crf is None:
        return _OpenCvVideoSink(cv2, output_path, fps=fps, width=width, height=height)
    return _FfmpegVideoSink(
        output_path,
        fps=fps,
        width=width,
        height=height,
        crf=ffmpeg_crf,
        preset=ffmpeg_preset,
        ffmpeg_path=ffmpeg_path,
    )


def _write_vertices_video(
    *,
    cv2,
    renderer,
    vertices,
    output_path,
    fps,
    width,
    height,
    camera_cfg,
    progress_label,
    show_progress,
    render_width=None,
    render_height=None,
    ffmpeg_crf=None,
    ffmpeg_preset="medium",
    ffmpeg_path="ffmpeg",
):
    from tqdm import tqdm

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(width)
    height = int(height)
    render_width = int(render_width or width)
    render_height = int(render_height or height)
    video_writer = _make_video_sink(
        cv2,
        output_path,
        fps=fps,
        width=width,
        height=height,
        ffmpeg_crf=ffmpeg_crf,
        ffmpeg_preset=ffmpeg_preset,
        ffmpeg_path=ffmpeg_path,
    )

    try:
        for frame_vertices in tqdm(vertices, desc=progress_label, disable=not show_progress):
            renderer.particle_vertices.from_numpy(np.asarray(frame_vertices, dtype=np.float32))

            renderer.camera.position(camera_cfg.camera_x, camera_cfg.camera_y, camera_cfg.camera_z)
            renderer.camera.lookat(camera_cfg.lookat_x, camera_cfg.lookat_y, camera_cfg.lookat_z)
            renderer.camera.up(camera_cfg.up_x, camera_cfg.up_y, camera_cfg.up_z)
            renderer.scene.set_camera(renderer.camera)
            renderer.scene.point_light(
                pos=(camera_cfg.camera_x, camera_cfg.camera_y, 0.0),
                color=(1.0, 1.0, 1.0),
            )
            renderer.scene.ambient_light((0.5, 0.5, 0.5))
            renderer.scene.mesh(
                renderer.particle_vertices,
                indices=renderer.indices,
                two_sided=True,
                show_wireframe=False,
            )

            renderer.canvas.scene(renderer.scene)
            image = renderer.window.get_image_buffer_as_numpy()
            image = cv2.cvtColor((image[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            image = np.transpose(image, (1, 0, 2))
            image = np.flip(image, axis=0)
            if render_width != width or render_height != height:
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            video_writer.write(image)
    finally:
        video_writer.release()


def render_vertices_to_video(
    *,
    vertices,
    faces,
    output_path,
    fps=30,
    width=512,
    height=512,
    arch="gpu",
    camera_preset=WORLD_FRONTAL_CAMERA_PRESET,
    show_progress=True,
    supersample=1,
    ffmpeg_crf=None,
    ffmpeg_preset="medium",
    ffmpeg_path="ffmpeg",
    **camera_overrides,
):
    import cv2

    _configure_taichi_env()
    import taichi as ti

    if not hasattr(ti, arch):
        raise ValueError("Unsupported Taichi arch: {}".format(arch))

    if hasattr(ti, "reset"):
        ti.reset()
    ti.init(arch=getattr(ti, arch), debug=False, log_level=ti.ERROR)

    supersample = max(int(supersample), 1)
    render_width = int(width) * supersample
    render_height = int(height) * supersample

    vertices = np.asarray(vertices, dtype=np.float32)
    camera_args = SimpleNamespace(camera_preset=camera_preset, **{field: None for field in CAMERA_FIELDS})
    for field in CAMERA_FIELDS:
        if field in camera_overrides:
            setattr(camera_args, field, camera_overrides[field])
    camera_cfg = resolve_camera_config(camera_args)
    renderer = _create_preview_renderer(
        ti,
        num_vertices=vertices.shape[1],
        faces=faces,
        width=render_width,
        height=render_height,
    )

    try:
        _write_vertices_video(
            cv2=cv2,
            renderer=renderer,
            vertices=vertices,
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            camera_cfg=camera_cfg,
            progress_label=Path(output_path).name,
            show_progress=show_progress,
            render_width=render_width,
            render_height=render_height,
            ffmpeg_crf=ffmpeg_crf,
            ffmpeg_preset=ffmpeg_preset,
            ffmpeg_path=ffmpeg_path,
        )
    finally:
        if hasattr(renderer.window, "destroy"):
            renderer.window.destroy()

    return str(output_path)


def render_preview_jobs(args):
    import cv2

    _configure_taichi_env()
    import taichi as ti

    if not hasattr(ti, args.arch):
        raise ValueError("Unsupported Taichi arch: {}".format(args.arch))

    if hasattr(ti, "reset"):
        ti.reset()
    ti.init(arch=getattr(ti, args.arch), debug=False, log_level=ti.ERROR)

    model = getattr(args, "model", None) or load_render_model(args.model_path)
    supersample = max(int(getattr(args, "supersample", 1)), 1)
    render_width = int(args.width) * supersample
    render_height = int(args.height) * supersample
    renderer = _create_preview_renderer(
        ti,
        num_vertices=model.v_template.shape[0],
        faces=model.faces,
        width=render_width,
        height=render_height,
    )
    show_progress = getattr(args, "show_progress", True)

    try:
        for pkl_path, video_path in build_preview_jobs(args):
            if osp.exists(video_path) and not args.force:
                continue

            stageii_inputs = load_stageii_render_inputs(pkl_path)
            camera_cfg = resolve_camera_config(args, stageii_inputs=stageii_inputs)
            vertices = load_vertices(
                pkl_path,
                model,
                stageii_inputs=stageii_inputs,
                neutral_face=getattr(args, "neutral_face", False),
                zero_jaw=getattr(args, "zero_jaw", False),
                zero_expression=getattr(args, "zero_expression", False),
            )
            _write_vertices_video(
                cv2=cv2,
                renderer=renderer,
                vertices=vertices,
                output_path=video_path,
                fps=args.fps,
                width=args.width,
                height=args.height,
                camera_cfg=camera_cfg,
                progress_label=osp.basename(pkl_path),
                show_progress=show_progress,
                render_width=render_width,
                render_height=render_height,
                ffmpeg_crf=getattr(args, "ffmpeg_crf", None),
                ffmpeg_preset=getattr(args, "ffmpeg_preset", "medium"),
                ffmpeg_path=getattr(args, "ffmpeg_path", "ffmpeg"),
            )
    finally:
        if hasattr(renderer.window, "destroy"):
            renderer.window.destroy()


def render_stageii_preview(
    *,
    input_path,
    output_path=None,
    model_path,
    fps=30,
    width=512,
    height=512,
    arch="gpu",
    camera_preset=DEFAULT_CAMERA_PRESET,
    input_suffix="_stageii.pkl",
    video_suffix="_stageii.mp4",
    force=False,
    supersample=1,
    ffmpeg_crf=None,
    ffmpeg_preset="medium",
    ffmpeg_path="ffmpeg",
    neutral_face=False,
    zero_jaw=False,
    zero_expression=False,
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
        supersample=supersample,
        ffmpeg_crf=ffmpeg_crf,
        ffmpeg_preset=ffmpeg_preset,
        ffmpeg_path=ffmpeg_path,
        neutral_face=neutral_face,
        zero_jaw=zero_jaw,
        zero_expression=zero_expression,
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
