import argparse
import json
from pathlib import Path

from moshpp.mosh_head import MoSh, run_moshpp_once
from utils.stageii_benchmark import run_public_stageii_benchmark, write_benchmark_report


REAL_MCP_BASELINE_PRESET = {
    "moshpp.optimize_fingers": "true",
    "runtime.sequence_chunk_size": "32",
    "runtime.sequence_chunk_overlap": "4",
    "runtime.sequence_seed_refine_iters": "5",
    "runtime.refine_lr": "0.05",
    "runtime.sequence_lr": "0.05",
    "runtime.sequence_max_iters": "30",
}

OFFICIAL_PRESETS = {
    "real-mcp-baseline": REAL_MCP_BASELINE_PRESET,
    "real-mcp-transvelo100-seedvelowindow": {
        **REAL_MCP_BASELINE_PRESET,
        "runtime.sequence_transl_velocity": "100",
        "runtime.sequence_boundary_transl_velocity_reference": "true",
    },
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the official single-sequence MoSh entrypoint with repeatable dotlist "
            "overrides, then optionally benchmark the produced stageii.pkl."
        )
    )
    parser.add_argument("--mocap-fname", required=True, help="Input .c3d or .mcp sequence path.")
    parser.add_argument("--support-base-dir", required=True, help="support_files root for model/layout assets.")
    parser.add_argument("--work-base-dir", required=True, help="Work directory where stagei/stageii outputs are written.")
    parser.add_argument(
        "--backend",
        default="torch",
        help="runtime.backend passed into MoSh.prepare_cfg(...). Defaults to torch.",
    )
    parser.add_argument(
        "--cfg",
        action="append",
        default=[],
        help="Repeatable dotlist override such as surface_model.gender=male or runtime.sequence_lr=0.05.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help=(
            "Optional suffix appended to mocap.basename after preset/--cfg resolution so repeated "
            "runs can keep distinct stageii/log outputs under one work directory."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(OFFICIAL_PRESETS),
        default=None,
        help=(
            "Optional named override pack applied before --cfg. "
            "Use real-mcp-baseline for the corrected real .mcp torch baseline or "
            "real-mcp-transvelo100-seedvelowindow for the translation-friendly candidate."
        ),
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only run the official entrypoint and report the produced stageii path.",
    )
    parser.add_argument(
        "--benchmark-output",
        default=None,
        help="Optional JSON output path for the benchmark report.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs forwarded to the stageii benchmark.")
    parser.add_argument("--measured-runs", type=int, default=5, help="Measured runs forwarded to the stageii benchmark.")
    parser.add_argument(
        "--mesh-reference",
        default=None,
        help="Optional baseline stageii.pkl or .pc2/.pc16 passed through to benchmark_stageii_public.",
    )
    parser.add_argument(
        "--mesh-support-base-dir",
        default=None,
        help="Optional support_files root used by mesh comparison; defaults to --support-base-dir.",
    )
    parser.add_argument("--mesh-chunk-size", type=int, default=None, help="Optional mesh comparison chunk-size override.")
    parser.add_argument(
        "--mesh-chunk-overlap",
        type=int,
        default=None,
        help="Optional mesh comparison chunk-overlap override.",
    )
    return parser


def _cfg_overrides(parser, args):
    overrides = dict(OFFICIAL_PRESETS.get(args.preset, {}))
    for entry in args.cfg:
        if "=" not in entry:
            parser.error(f"--cfg entries must be KEY=VALUE, got: {entry}")
        key, value = entry.split("=", 1)
        if not key:
            parser.error(f"--cfg entries must include a non-empty key, got: {entry}")
        overrides[key] = value

    if args.output_suffix:
        basename = overrides.get("mocap.basename")
        if basename is None:
            basename = Path(args.mocap_fname).stem
        overrides["mocap.basename"] = f"{basename}{args.output_suffix}"

    overrides["mocap.fname"] = args.mocap_fname
    overrides["dirs.support_base_dir"] = args.support_base_dir
    overrides["dirs.work_base_dir"] = args.work_base_dir
    overrides["runtime.backend"] = args.backend
    return overrides


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = MoSh.prepare_cfg(**_cfg_overrides(parser, args))
    run_moshpp_once(cfg)

    stageii_path = Path(str(cfg.dirs.stageii_fname))
    if not stageii_path.exists():
        raise FileNotFoundError(f"run_moshpp_once did not produce the expected stageii file: {stageii_path}")

    payload = {
        "stageii_path": str(stageii_path),
        "benchmark": None,
    }

    if not args.skip_benchmark:
        report = run_public_stageii_benchmark(
            str(stageii_path),
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            mesh_reference_path=args.mesh_reference,
            mesh_support_base_dir=args.mesh_support_base_dir or args.support_base_dir,
            mesh_chunk_size=args.mesh_chunk_size,
            mesh_chunk_overlap=args.mesh_chunk_overlap,
        )
        if args.benchmark_output:
            report = write_benchmark_report(report, str(args.benchmark_output))
        payload["benchmark"] = report

    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
