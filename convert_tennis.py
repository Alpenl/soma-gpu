import argparse
import os.path as osp
import pickle
from pathlib import Path

from utils.script_utils import resolve_support_base_dir

DEFAULT_EXPR_ID = "V48_02_SuperSet"
DEFAULT_DATA_ID = "OC_05_G_03_real_000_synt_100"
DEFAULT_MOCAP_EXT = ".c3d"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SOMA followed by MoSh++ for a tennis-style mocap dataset."
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
        "--soma-work-base-dir",
        default=osp.dirname(osp.abspath(__file__)),
        help="Root working directory used by SOMA.",
    )
    parser.add_argument(
        "--support-base-dir",
        default=None,
        help="Optional override for the support files directory.",
    )
    parser.add_argument(
        "--expr-id",
        default=DEFAULT_EXPR_ID,
        help="SOMA checkpoint expression id.",
    )
    parser.add_argument(
        "--data-id",
        default=DEFAULT_DATA_ID,
        help="SOMA dataset id for the selected checkpoint.",
    )
    parser.add_argument(
        "--fname-filter",
        nargs="*",
        default=[],
        help="Optional list of substrings used to filter mocap filenames.",
    )
    parser.add_argument(
        "--mocap-unit",
        default="mm",
        help="Unit stored in the mocap files.",
    )
    parser.add_argument(
        "--mocap-ext",
        default=DEFAULT_MOCAP_EXT,
        help="Extension of the input mocap files.",
    )
    parser.add_argument(
        "--soma-batch-size",
        type=int,
        default=256,
        help="Batch size passed to SOMA inference.",
    )
    parser.add_argument(
        "--skip-soma",
        action="store_true",
        help="Skip the SOMA labeling pass.",
    )
    parser.add_argument(
        "--skip-mosh",
        action="store_true",
        help="Skip the MoSh++ fitting pass.",
    )
    parser.add_argument(
        "--export-artifacts",
        action="store_true",
        help="After SOMA/MoSh++, export OBJ, PC2, and preview MP4 for discovered stageii pickles.",
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


def _load_pickle_compat(path):
    with Path(path).open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _matches_fname_filter(path, fname_filter):
    if not fname_filter:
        return True
    path = str(path)
    return any(token in path for token in fname_filter)


def _discover_stageii_pickles(work_base_dir, dataset, *, fname_filter=None):
    dataset_root = Path(work_base_dir) / dataset
    if not dataset_root.exists():
        return []
    return [
        str(path)
        for path in sorted(dataset_root.rglob("*_stageii.pkl"))
        if _matches_fname_filter(path, fname_filter)
    ]


def _stageii_model_path(stageii_pkl):
    stageii_data = _load_pickle_compat(stageii_pkl)
    cfg = stageii_data.get("stageii_debug_details", {}).get("cfg")
    if cfg is None:
        raise KeyError(f"{stageii_pkl} does not include stageii_debug_details.cfg")
    try:
        model_path = cfg["surface_model"]["fname"]
    except Exception as exc:
        raise KeyError(f"{stageii_pkl} does not include cfg.surface_model.fname") from exc
    if not model_path:
        raise ValueError(f"{stageii_pkl} resolved an empty cfg.surface_model.fname")
    return str(model_path)


def export_stageii_artifacts_for_dataset(
    *,
    work_base_dir,
    dataset,
    fname_filter=None,
    fps=30,
    width=512,
    height=512,
    arch="gpu",
    camera_preset="frontal",
):
    import export_stageii_artifacts

    results = []
    for stageii_pkl in _discover_stageii_pickles(work_base_dir, dataset, fname_filter=fname_filter):
        result = export_stageii_artifacts.export_stageii_artifacts(
            input_pkl=stageii_pkl,
            model_path=_stageii_model_path(stageii_pkl),
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

    if args.skip_soma and args.skip_mosh:
        parser.error("At least one of SOMA or MoSh++ must remain enabled.")

    from soma.tools.run_soma_multiple import run_soma_on_multiple_settings

    support_base_dir = resolve_support_base_dir(
        args.soma_work_base_dir, args.support_base_dir
    )
    parallel_cfg = {"randomly_run_jobs": True}

    if not args.skip_soma:
        run_soma_on_multiple_settings(
            soma_expr_ids=[args.expr_id],
            soma_mocap_target_ds_names=[args.dataset],
            soma_data_ids=[args.data_id],
            soma_cfg={
                "soma.batch_size": args.soma_batch_size,
                "dirs.support_base_dir": support_base_dir,
                "mocap.unit": args.mocap_unit,
                "save_c3d": True,
                "keep_nan_points": True,
                "remove_zero_trajectories": True,
            },
            parallel_cfg=parallel_cfg,
            run_tasks=["soma"],
            mocap_base_dir=args.mocap_base_dir,
            soma_work_base_dir=args.soma_work_base_dir,
            mocap_ext=args.mocap_ext,
            fname_filter=args.fname_filter,
        )

    if not args.skip_mosh:
        run_soma_on_multiple_settings(
            soma_expr_ids=[args.expr_id],
            soma_mocap_target_ds_names=[args.dataset],
            soma_data_ids=[args.data_id],
            mosh_cfg={
                "moshpp.verbosity": 1,
                "moshpp.stagei_frame_picker.type": "random",
                "dirs.support_base_dir": support_base_dir,
            },
            mocap_base_dir=args.mocap_base_dir,
            run_tasks=["mosh"],
            fname_filter=args.fname_filter,
            mocap_ext=args.mocap_ext,
            soma_work_base_dir=args.soma_work_base_dir,
            parallel_cfg=parallel_cfg,
        )

    if args.export_artifacts:
        export_stageii_artifacts_for_dataset(
            work_base_dir=args.soma_work_base_dir,
            dataset=args.dataset,
            fname_filter=args.fname_filter,
            fps=args.export_fps,
            width=args.export_width,
            height=args.export_height,
            arch=args.export_arch,
            camera_preset=args.export_camera_preset,
        )


if __name__ == "__main__":
    main()
