import argparse
import pickle

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
        help="Path to the SMPL-X model npz file.",
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
    import torch

    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)

    num_frames = data["fullpose"].shape[0]
    fullpose = torch.from_numpy(data["fullpose"]).float()
    shape_components = torch.from_numpy(data["betas"]).float()[None]
    trans = torch.from_numpy(data["trans"]).float()

    betas = shape_components[:, :10]
    expression = shape_components[:, 300:310]

    ret = model(
        global_orient=fullpose[:, :3],
        transl=trans,
        body_pose=fullpose[:, 3:66],
        betas=betas,
        expression=expression.repeat(num_frames, 1),
        jaw_pose=fullpose[:, 66:69],
        leye_pose=fullpose[:, 69:72],
        reye_pose=fullpose[:, 72:75],
        left_hand_pose=fullpose[:, 75:120],
        right_hand_pose=fullpose[:, 120:165],
    )
    return ret.vertices.detach().cpu().numpy()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    import smplx

    model = smplx.SMPLX(args.model_path, use_pca=False)
    vertices = load_smpl_vertices(args.input_pkl, model)
    obj_out, pc2_out = default_stageii_output_paths(args.input_pkl)

    save_obj_mesh(args.obj_out or obj_out, vertices[0], model.faces)
    writePC2(args.pc2_out or pc2_out, vertices)


if __name__ == "__main__":
    main()
