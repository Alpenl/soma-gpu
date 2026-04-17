import argparse
from pathlib import Path

import render_video
from utils.mesh_io import save_obj_mesh, writePC2
from utils.script_utils import (
    batch_output_dir_for_input,
    default_stageii_output_paths,
    discover_stageii_pickles_in_dir,
    format_stageii_match_error,
    resolve_stageii_model_path,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export the first-frame OBJ and full-sequence PC2 from a stageii pickle."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-pkl",
        help="Path to the *_stageii.pkl file.",
    )
    input_group.add_argument(
        "--input-dir",
        default=None,
        help="Recursive root used to discover *_stageii.pkl files for batch mesh export.",
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
        default=None,
        help="Optional list of substrings used to filter discovered *_stageii.pkl paths in --input-dir batch mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory used for default OBJ/PC2 outputs.",
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
    return parser


def load_smpl_vertices(pkl_path, model):
    return render_video.load_vertices(pkl_path, model)


def _load_render_model_with_context(model_path, *, input_pkl):
    try:
        return render_video.load_render_model(model_path)
    except (FileNotFoundError, ImportError, ModuleNotFoundError, ValueError) as exc:
        raise ValueError(
            f"{input_pkl}: failed to load render model from {model_path}: {exc}"
        ) from exc


def _resolve_mesh_output_paths(input_pkl, *, output_dir=None, obj_out=None, pc2_out=None):
    default_obj_out, default_pc2_out = default_stageii_output_paths(str(input_pkl))
    if output_dir is not None:
        output_dir = Path(output_dir)
        default_obj_out = str(output_dir / Path(default_obj_out).name)
        default_pc2_out = str(output_dir / Path(default_pc2_out).name)
    return str(obj_out or default_obj_out), str(pc2_out or default_pc2_out)


def export_stageii_meshes(
    input_pkl,
    model_path=None,
    *,
    model=None,
    vertices=None,
    obj_out=None,
    pc2_out=None,
):
    if model is None:
        if model_path is None:
            raise ValueError("model_path is required when model is not provided")
        model = _load_render_model_with_context(model_path, input_pkl=input_pkl)
    if vertices is None:
        vertices = load_smpl_vertices(input_pkl, model)
    obj_path, pc2_path = _resolve_mesh_output_paths(input_pkl, obj_out=obj_out, pc2_out=pc2_out)
    Path(obj_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pc2_path).parent.mkdir(parents=True, exist_ok=True)

    save_obj_mesh(obj_path, vertices[0], model.faces)
    writePC2(pc2_path, vertices)
    return obj_path, pc2_path


def export_stageii_meshes_batch(*, input_pkls, support_base_dir=None, output_dir=None, input_root=None):
    input_pkls = [str(Path(input_pkl)) for input_pkl in input_pkls]
    if not input_pkls:
        raise ValueError("No *_stageii.pkl files matched for mesh export.")

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
        obj_out, pc2_out = _resolve_mesh_output_paths(
            input_pkl,
            output_dir=batch_output_dir_for_input(
                input_pkl,
                output_dir=output_dir,
                input_root=input_root,
            ),
        )
        obj_path, pc2_path = export_stageii_meshes(
            input_pkl=input_pkl,
            model_path=resolved_model_path,
            model=model,
            obj_out=obj_out,
            pc2_out=pc2_out,
        )
        results.append(
            {
                "obj_path": obj_path,
                "pc2_path": pc2_path,
            }
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
    if unsupported_args:
        return "{} only support --input-pkl.".format(", ".join(unsupported_args))
    return None


def _single_input_only_args_error(args):
    unsupported_args = []
    if args.fname_filter is not None:
        unsupported_args.append("--fname-filter")
    if unsupported_args:
        return "{} requires --input-dir.".format(", ".join(unsupported_args))
    return None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_dir:
        input_dir_error = _batch_only_args_error(args)
        if input_dir_error is not None:
            parser.error(input_dir_error)
        fname_filter = args.fname_filter or []
        input_pkls = discover_stageii_pickles_in_dir(
            args.input_dir,
            fname_filter=fname_filter,
        )
        if not input_pkls:
            parser.error(
                format_stageii_match_error(
                    args.input_dir,
                    fname_filter=fname_filter,
                )
            )
        try:
            export_stageii_meshes_batch(
                input_pkls=input_pkls,
                support_base_dir=args.support_base_dir,
                output_dir=args.output_dir,
                input_root=args.input_dir,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return
    single_input_error = _single_input_only_args_error(args)
    if single_input_error is not None:
        parser.error(single_input_error)
    try:
        resolved_model_path = args.model_path or resolve_stageii_model_path(
            args.input_pkl,
            support_base_dir=args.support_base_dir,
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    try:
        obj_out, pc2_out = _resolve_mesh_output_paths(
            args.input_pkl,
            output_dir=args.output_dir,
            obj_out=args.obj_out,
            pc2_out=args.pc2_out,
        )
        export_stageii_meshes(
            input_pkl=args.input_pkl,
            model_path=resolved_model_path,
            obj_out=obj_out,
            pc2_out=pc2_out,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
