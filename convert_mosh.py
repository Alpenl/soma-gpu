import argparse
import os.path as osp
from glob import glob

from utils.script_utils import resolve_support_base_dir


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
    return parser


def collect_mocap_fnames(mocap_base_dir, dataset_name, fname_filter):
    pattern = osp.join(mocap_base_dir, dataset_name, "*", "*.c3d")
    mocap_fnames = sorted(glob(pattern))
    if fname_filter:
        mocap_fnames = [
            mocap_fname
            for mocap_fname in mocap_fnames
            if any(token in mocap_fname for token in fname_filter)
        ]
    return mocap_fnames


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    from soma.amass.mosh_manual import mosh_manual

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
            "No .c3d files matched {}/{}/*/*.c3d".format(
                args.mocap_base_dir, args.dataset
            )
        )

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


if __name__ == "__main__":
    main()
