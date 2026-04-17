import argparse
from pathlib import Path

import render_video
import save_smplx_verts
from utils.script_utils import default_stageii_artifact_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export OBJ, PC2, and preview MP4 artifacts from a stageii pickle."
    )
    parser.add_argument(
        "--input-pkl",
        required=True,
        help="Path to the *_stageii.pkl file.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the SMPL-X model .npz or .pkl file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory used for default OBJ/PC2/MP4 outputs.",
    )
    parser.add_argument(
        "--obj-out",
        default=None,
        help="Optional path for the OBJ output. Defaults to the input stem.",
    )
    parser.add_argument(
        "--pc2-out",
        default=None,
        help="Optional path for the PC2 output. Defaults to the input stem.",
    )
    parser.add_argument(
        "--video-out",
        default=None,
        help="Optional path for the preview MP4 output. Defaults to the input stem.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the preview video.",
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
        choices=sorted(render_video.CAMERA_PRESETS),
        help="Stable preview camera preset. Manual camera args override preset fields.",
    )
    for field in render_video.CAMERA_FIELDS:
        parser.add_argument(
            "--" + field.replace("_", "-"),
            type=float,
            default=None,
            help="Override camera field {}.".format(field),
        )
    return parser


def _resolve_artifact_paths(input_pkl, *, output_dir=None, obj_out=None, pc2_out=None, video_out=None, video_suffix="_stageii.mp4"):
    default_obj_out, default_pc2_out, default_video_out = default_stageii_artifact_paths(
        str(input_pkl),
        video_suffix=video_suffix,
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        default_obj_out = str(output_dir / Path(default_obj_out).name)
        default_pc2_out = str(output_dir / Path(default_pc2_out).name)
        default_video_out = str(output_dir / Path(default_video_out).name)

    return (
        str(obj_out or default_obj_out),
        str(pc2_out or default_pc2_out),
        str(video_out or default_video_out),
    )


def export_stageii_artifacts(
    input_pkl,
    model_path=None,
    *,
    model=None,
    vertices=None,
    output_dir=None,
    obj_out=None,
    pc2_out=None,
    video_out=None,
    fps=30,
    width=512,
    height=512,
    arch="gpu",
    camera_preset="frontal",
    video_suffix="_stageii.mp4",
    show_progress=True,
    **camera_overrides,
):
    if model is None:
        if model_path is None:
            raise ValueError("model_path is required when model is not provided")
        model = render_video.load_render_model(model_path)
    if vertices is None:
        vertices = render_video.load_vertices(input_pkl, model)

    obj_path, pc2_path, resolved_video_out = _resolve_artifact_paths(
        input_pkl,
        output_dir=output_dir,
        obj_out=obj_out,
        pc2_out=pc2_out,
        video_out=video_out,
        video_suffix=video_suffix,
    )
    obj_path, pc2_path = save_smplx_verts.export_stageii_meshes(
        input_pkl=input_pkl,
        model=model,
        vertices=vertices,
        obj_out=obj_path,
        pc2_out=pc2_path,
    )
    video_path = render_video.render_vertices_to_video(
        vertices=vertices,
        faces=model.faces,
        output_path=resolved_video_out,
        fps=fps,
        width=width,
        height=height,
        arch=arch,
        camera_preset=camera_preset,
        show_progress=show_progress,
        **camera_overrides,
    )
    return {
        "obj_path": obj_path,
        "pc2_path": pc2_path,
        "video_path": video_path,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    camera_overrides = {
        field: getattr(args, field)
        for field in render_video.CAMERA_FIELDS
        if getattr(args, field) is not None
    }
    export_stageii_artifacts(
        input_pkl=args.input_pkl,
        model_path=args.model_path,
        output_dir=args.output_dir,
        obj_out=args.obj_out,
        pc2_out=args.pc2_out,
        video_out=args.video_out,
        fps=args.fps,
        width=args.width,
        height=args.height,
        arch=args.arch,
        camera_preset=args.camera_preset,
        **camera_overrides,
    )


if __name__ == "__main__":
    main()
