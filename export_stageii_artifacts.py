import argparse
from pathlib import Path
from types import SimpleNamespace

import render_video
import save_smplx_verts
from utils.script_utils import (
    batch_output_dir_for_input,
    default_stageii_artifact_paths,
    discover_stageii_pickles_in_dir,
    format_stageii_match_error,
    resolve_stageii_model_path,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export OBJ, PC2, and preview MP4 artifacts from a stageii pickle."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-pkl",
        help="Path to the *_stageii.pkl file.",
    )
    input_group.add_argument(
        "--input-dir",
        default=None,
        help="Recursive root used to discover *_stageii.pkl files for batch export.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to the SMPL-X model .npz or .pkl file. Auto-resolved when omitted.",
    )
    parser.add_argument(
        "--support-base-dir",
        default=None,
        help="Optional support_files root used to relocate stageii model paths.",
    )
    parser.add_argument(
        "--fname-filter",
        nargs="*",
        default=[],
        help="Optional list of substrings used to filter discovered *_stageii.pkl paths.",
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
        "--supersample",
        type=int,
        default=1,
        help="Render preview MP4 internally at N times the output resolution and downsample. Use 2 for final videos.",
    )
    parser.add_argument(
        "--ffmpeg-crf",
        type=int,
        default=None,
        help="Use ffmpeg/libx264 CRF encoding for preview MP4. Try 16 for high-quality MP4.",
    )
    parser.add_argument(
        "--ffmpeg-preset",
        default="medium",
        help="ffmpeg/libx264 preset used when --ffmpeg-crf is set.",
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
        default=render_video.DEFAULT_CAMERA_PRESET,
        choices=render_video.CAMERA_CHOICES,
        help="Preview camera preset. subject-frontal follows the actor's facing direction and manual camera args override preset fields.",
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


def _load_render_model_with_context(model_path, *, input_pkl):
    try:
        return render_video.load_render_model(model_path)
    except (FileNotFoundError, ImportError, ModuleNotFoundError, ValueError) as exc:
        raise ValueError(
            f"{input_pkl}: failed to load render model from {model_path}: {exc}"
        ) from exc


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
    camera_preset=render_video.DEFAULT_CAMERA_PRESET,
    video_suffix="_stageii.mp4",
    show_progress=True,
    supersample=1,
    ffmpeg_crf=None,
    ffmpeg_preset="medium",
    ffmpeg_path="ffmpeg",
    neutral_face=False,
    zero_jaw=False,
    zero_expression=False,
    **camera_overrides,
):
    stageii_inputs = None
    if model is None:
        if model_path is None:
            raise ValueError("model_path is required when model is not provided")
        model = _load_render_model_with_context(model_path, input_pkl=input_pkl)
    if vertices is None:
        vertices = render_video.load_vertices(
            input_pkl,
            model,
            neutral_face=neutral_face,
            zero_jaw=zero_jaw,
            zero_expression=zero_expression,
        )
    if stageii_inputs is None and camera_preset == render_video.SUBJECT_FRONTAL_CAMERA_PRESET:
        try:
            stageii_inputs = render_video.load_stageii_render_inputs(input_pkl)
        except Exception:
            camera_preset = render_video.WORLD_FRONTAL_CAMERA_PRESET

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
    camera_args = SimpleNamespace(
        camera_preset=camera_preset,
        **{field: camera_overrides.get(field) for field in render_video.CAMERA_FIELDS},
    )
    camera_cfg = render_video.resolve_camera_config(
        camera_args,
        stageii_inputs=stageii_inputs,
    )
    resolved_camera_overrides = {
        field: getattr(camera_cfg, field) for field in render_video.CAMERA_FIELDS
    }
    video_path = render_video.render_vertices_to_video(
        vertices=vertices,
        faces=model.faces,
        output_path=resolved_video_out,
        fps=fps,
        width=width,
        height=height,
        arch=arch,
        camera_preset=render_video.WORLD_FRONTAL_CAMERA_PRESET,
        show_progress=show_progress,
        supersample=supersample,
        ffmpeg_crf=ffmpeg_crf,
        ffmpeg_preset=ffmpeg_preset,
        ffmpeg_path=ffmpeg_path,
        **resolved_camera_overrides,
    )
    return {
        "obj_path": obj_path,
        "pc2_path": pc2_path,
        "video_path": video_path,
    }


def export_stageii_artifacts_batch(
    *,
    input_pkls,
    support_base_dir=None,
    output_dir=None,
    input_root=None,
    fps=30,
    width=512,
    height=512,
    arch="gpu",
    camera_preset=render_video.DEFAULT_CAMERA_PRESET,
    show_progress=True,
    supersample=1,
    ffmpeg_crf=None,
    ffmpeg_preset="medium",
    ffmpeg_path="ffmpeg",
    neutral_face=False,
    zero_jaw=False,
    zero_expression=False,
    **camera_overrides,
):
    input_pkls = [str(Path(input_pkl)) for input_pkl in input_pkls]
    if not input_pkls:
        raise ValueError("No *_stageii.pkl files matched for artifact export.")

    results = []
    model_cache = {}
    for input_pkl in input_pkls:
        try:
            resolved_model_path = resolve_stageii_model_path(
                input_pkl, support_base_dir=support_base_dir
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{input_pkl}: {exc}") from exc
        model = model_cache.get(resolved_model_path)
        if model is None:
            model = _load_render_model_with_context(
                resolved_model_path,
                input_pkl=input_pkl,
            )
            model_cache[resolved_model_path] = model
        results.append(
            export_stageii_artifacts(
                input_pkl=input_pkl,
                model_path=resolved_model_path,
                model=model,
                output_dir=batch_output_dir_for_input(
                    input_pkl,
                    output_dir=output_dir,
                    input_root=input_root,
                ),
                fps=fps,
                width=width,
                height=height,
                arch=arch,
                camera_preset=camera_preset,
                show_progress=show_progress,
                supersample=supersample,
                ffmpeg_crf=ffmpeg_crf,
                ffmpeg_preset=ffmpeg_preset,
                ffmpeg_path=ffmpeg_path,
                neutral_face=neutral_face,
                zero_jaw=zero_jaw,
                zero_expression=zero_expression,
                **camera_overrides,
            )
        )
    return results


def _batch_only_args_error(args):
    unsupported_args = []
    if args.model_path is not None:
        unsupported_args.append("--model-path")
    if args.obj_out is not None:
        unsupported_args.append("--obj-out")
    if args.pc2_out is not None:
        unsupported_args.append("--pc2-out")
    if args.video_out is not None:
        unsupported_args.append("--video-out")
    if unsupported_args:
        return "{} only support --input-pkl.".format(", ".join(unsupported_args))
    return None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    camera_overrides = {
        field: getattr(args, field)
        for field in render_video.CAMERA_FIELDS
        if getattr(args, field) is not None
    }
    if args.input_dir:
        input_dir_error = _batch_only_args_error(args)
        if input_dir_error is not None:
            parser.error(input_dir_error)
        input_pkls = discover_stageii_pickles_in_dir(
            args.input_dir,
            fname_filter=args.fname_filter,
        )
        if not input_pkls:
            parser.error(
                format_stageii_match_error(
                    args.input_dir,
                    fname_filter=args.fname_filter,
                )
            )
        try:
            export_stageii_artifacts_batch(
                input_pkls=input_pkls,
                support_base_dir=args.support_base_dir,
                output_dir=args.output_dir,
                input_root=args.input_dir,
                fps=args.fps,
                width=args.width,
                height=args.height,
                arch=args.arch,
                camera_preset=args.camera_preset,
                supersample=args.supersample,
                ffmpeg_crf=args.ffmpeg_crf,
                ffmpeg_preset=args.ffmpeg_preset,
                ffmpeg_path=args.ffmpeg_path,
                neutral_face=args.neutral_face,
                zero_jaw=args.zero_jaw,
                zero_expression=args.zero_expression,
                **camera_overrides,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return

    try:
        resolved_model_path = args.model_path or resolve_stageii_model_path(
            args.input_pkl,
            support_base_dir=args.support_base_dir,
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    try:
        export_stageii_artifacts(
            input_pkl=args.input_pkl,
            model_path=resolved_model_path,
            output_dir=args.output_dir,
            obj_out=args.obj_out,
            pc2_out=args.pc2_out,
            video_out=args.video_out,
            fps=args.fps,
            width=args.width,
            height=args.height,
            arch=args.arch,
            camera_preset=args.camera_preset,
            supersample=args.supersample,
            ffmpeg_crf=args.ffmpeg_crf,
            ffmpeg_preset=args.ffmpeg_preset,
            ffmpeg_path=args.ffmpeg_path,
            neutral_face=args.neutral_face,
            zero_jaw=args.zero_jaw,
            zero_expression=args.zero_expression,
            **camera_overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
