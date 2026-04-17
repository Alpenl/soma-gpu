import argparse
import os.path as osp
from glob import glob
from pathlib import Path

from utils.script_utils import (
    discover_stageii_pickles,
    resolve_stageii_model_path,
    resolve_support_base_dir,
)

SUPPORTED_MOCAP_SUFFIXES = (".c3d", ".mcp")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run MoSh++ directly on a dataset without the SOMA pass."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset/session name under the mocap base directory.",
    )
    parser.add_argument(
        "--mocap-base-dir",
        required=True,
        help="Base directory containing <dataset>/<subject>/*.c3d.",
    )
    parser.add_argument(
        "--work-base-dir",
        default=osp.dirname(osp.abspath(__file__)),
        help="Root working directory used by this repository.",
    )
    parser.add_argument(
        "--support-base-dir",
        default=None,
        help="Optional override for the support files directory.",
    )
    parser.add_argument(
        "--mosh-output-base-dir",
        default=None,
        help="Optional override for the MoSh++ output directory.",
    )
    parser.add_argument(
        "--fname-filter",
        nargs="*",
        default=[],
        help="Optional list of substrings used to filter mocap filenames.",
    )
    parser.add_argument(
        "--mocap-unit",
        default=None,
        help="Optional mocap unit override, for example mm.",
    )
    parser.add_argument(
        "--export-artifacts",
        action="store_true",
        help="After MoSh++, export OBJ, PC2, and preview MP4 for discovered stageii pickles.",
    )
    parser.add_argument(
        "--export-fps",
        type=int,
        default=30,
        help="Frames per second used by stageii artifact preview rendering.",
    )
    parser.add_argument(
        "--export-width",
        type=int,
        default=512,
        help="Output width used by stageii artifact preview rendering.",
    )
    parser.add_argument(
        "--export-height",
        type=int,
        default=512,
        help="Output height used by stageii artifact preview rendering.",
    )
    parser.add_argument(
        "--export-arch",
        default="gpu",
        help="Taichi backend used by stageii artifact preview rendering.",
    )
    parser.add_argument(
        "--export-camera-preset",
        default="frontal",
        help="Camera preset forwarded to export_stageii_artifacts.py.",
    )
    return parser


def collect_mocap_fnames(mocap_base_dir, dataset_name, fname_filter):
    mocap_fnames = []
    for suffix in SUPPORTED_MOCAP_SUFFIXES:
        pattern = osp.join(mocap_base_dir, dataset_name, "*", "*" + suffix)
        mocap_fnames.extend(glob(pattern))
    mocap_fnames = sorted(mocap_fnames)
    if fname_filter:
        mocap_fnames = [
            mocap_fname
            for mocap_fname in mocap_fnames
            if any(token in mocap_fname for token in fname_filter)
        ]
    return mocap_fnames


def find_duplicate_mocap_aliases(mocap_fnames, *, dataset_root):
    seen = {}
    duplicates = set()
    dataset_root = Path(dataset_root)
    for mocap_fname in mocap_fnames:
        relative_stem = str(Path(mocap_fname).relative_to(dataset_root).with_suffix(""))
        current_suffix = Path(mocap_fname).suffix.lower()
        previous_suffix = seen.setdefault(relative_stem, current_suffix)
        if previous_suffix != current_suffix:
            duplicates.add(relative_stem)
    return sorted(duplicates)


def export_stageii_artifacts_for_dataset(
    *,
    work_base_dir,
    dataset,
    support_base_dir=None,
    fname_filter=None,
    fps=30,
    width=512,
    height=512,
    arch="gpu",
    camera_preset="frontal",
):
    import export_stageii_artifacts

    results = []
    for stageii_pkl in discover_stageii_pickles(
        work_base_dir, dataset, fname_filter=fname_filter
    ):
        result = export_stageii_artifacts.export_stageii_artifacts(
            input_pkl=stageii_pkl,
            model_path=resolve_stageii_model_path(
                stageii_pkl, support_base_dir=support_base_dir
            ),
            fps=fps,
            width=width,
            height=height,
            arch=arch,
            camera_preset=camera_preset,
        )
        results.append(result)
    return results


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    support_base_dir = resolve_support_base_dir(
        args.work_base_dir, args.support_base_dir
    )
    mosh_output_base_dir = args.mosh_output_base_dir or osp.join(
        args.work_base_dir, "mosh_results"
    )
    mocap_fnames = collect_mocap_fnames(
        args.mocap_base_dir, args.dataset, args.fname_filter
    )
    if not mocap_fnames:
        parser.error(
            "No .c3d or .mcp files matched under {}/{}/*/".format(
                args.mocap_base_dir, args.dataset
            )
        )
    duplicate_aliases = find_duplicate_mocap_aliases(
        mocap_fnames,
        dataset_root=Path(args.mocap_base_dir) / args.dataset,
    )
    if duplicate_aliases:
        parser.error(
            "Found duplicate .c3d/.mcp aliases for the same sequence: {}. "
            "Remove one source file.".format(", ".join(duplicate_aliases))
        )

    from soma.amass.mosh_manual import mosh_manual

    mosh_cfg = {
        "dirs.support_base_dir": support_base_dir,
        "dirs.work_base_dir": mosh_output_base_dir,
        "moshpp.verbosity": 1,
    }
    if args.mocap_unit:
        mosh_cfg["mocap.unit"] = args.mocap_unit

    parallel_cfg = {"randomly_run_jobs": True}

    mosh_manual(
        mocap_fnames=mocap_fnames,
        mosh_cfg=mosh_cfg,
        render_cfg={},
        parallel_cfg=parallel_cfg,
        run_tasks=["mosh"],
    )

    if args.export_artifacts:
        export_stageii_artifacts_for_dataset(
            work_base_dir=mosh_output_base_dir,
            dataset=args.dataset,
            support_base_dir=support_base_dir,
            fname_filter=args.fname_filter,
            fps=args.export_fps,
            width=args.export_width,
            height=args.export_height,
            arch=args.export_arch,
            camera_preset=args.export_camera_preset,
        )


if __name__ == "__main__":
    main()
