import argparse
from pathlib import Path

import render_video
from utils.mesh_io import save_obj_mesh, writePC2
from utils.script_utils import default_stageii_output_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export the first-frame OBJ and full-sequence PC2 from a stageii pickle."
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


def export_stageii_meshes(input_pkl, model_path=None, *, model=None, vertices=None, obj_out=None, pc2_out=None):
    if model is None:
        if model_path is None:
            raise ValueError("model_path is required when model is not provided")
        model = render_video.load_render_model(model_path)
    if vertices is None:
        vertices = load_smpl_vertices(input_pkl, model)
    default_obj_out, default_pc2_out = default_stageii_output_paths(str(input_pkl))

    obj_path = str(obj_out or default_obj_out)
    pc2_path = str(pc2_out or default_pc2_out)
    Path(obj_path).parent.mkdir(parents=True, exist_ok=True)
    Path(pc2_path).parent.mkdir(parents=True, exist_ok=True)

    save_obj_mesh(obj_path, vertices[0], model.faces)
    writePC2(pc2_path, vertices)
    return obj_path, pc2_path


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    export_stageii_meshes(
        input_pkl=args.input_pkl,
        model_path=args.model_path,
        obj_out=args.obj_out,
        pc2_out=args.pc2_out,
    )


if __name__ == "__main__":
    main()
