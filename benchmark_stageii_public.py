import argparse
import json

from utils.stageii_benchmark import (
    default_benchmark_output_path,
    run_public_stageii_benchmark,
    validate_benchmark_output_path,
    write_benchmark_report,
)

BENCHMARK_CLI_ERROR_TYPES = (KeyError, ValueError, OSError, ImportError, ModuleNotFoundError)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a stageii pickle ingest workload, plus optional preview/export "
            "stages when the local environment supports them, and emit a JSON report."
        )
    )
    parser.add_argument(
        "--input",
        default="support_data/tests/mosh_stageii.pkl",
        help="Path to the stageii pickle sample. Defaults to the shipped public sample.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional path for the JSON report. Defaults to a *_benchmark.json file "
            "next to --input."
        ),
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup parses to discard before measuring.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=5,
        help="Number of measured parses to record.",
    )
    parser.add_argument(
        "--mesh-reference",
        default=None,
        help="Optional baseline stageii.pkl or .pc2/.pc16 path used to append mesh-space comparison summaries.",
    )
    parser.add_argument(
        "--mesh-support-base-dir",
        default=None,
        help="support_files root used when --mesh-reference stageii/pc2 inputs need relocated model paths.",
    )
    parser.add_argument(
        "--mesh-chunk-size",
        type=int,
        default=None,
        help="Optional chunk size override for mesh comparison, mainly for raw PC2 inputs.",
    )
    parser.add_argument(
        "--mesh-chunk-overlap",
        type=int,
        default=None,
        help="Optional chunk overlap override paired with --mesh-chunk-size for mesh comparison.",
    )
    parser.add_argument(
        "--lean-benchmark",
        action="store_true",
        help=(
            "Skip optional preview vertex decode, mesh export, mp4 render, and artifact bundle "
            "speed probes; keep the core ingest latency, repeatability, and quality summaries."
        ),
    )
    return parser


def _validate_mesh_cli_args(parser, args):
    if args.mesh_chunk_size is not None:
        if args.mesh_reference is None:
            parser.error("--mesh-chunk-size requires --mesh-reference")
        if args.mesh_chunk_size <= 0:
            parser.error("--mesh-chunk-size must be > 0")
    if args.mesh_chunk_overlap is not None:
        if args.mesh_reference is None:
            parser.error("--mesh-chunk-overlap requires --mesh-reference")
        if args.mesh_chunk_size is None:
            parser.error("--mesh-chunk-overlap requires --mesh-chunk-size")
        if args.mesh_chunk_overlap < 0:
            parser.error("--mesh-chunk-overlap must be >= 0")
    if args.mesh_support_base_dir is not None and args.mesh_reference is None:
        parser.error("--mesh-support-base-dir requires --mesh-reference")


def _mesh_support_base_dir(args):
    if args.mesh_reference is None:
        return None
    return args.mesh_support_base_dir or "support_files"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_mesh_cli_args(parser, args)
    output_path = args.output or default_benchmark_output_path(args.input)
    try:
        validate_benchmark_output_path(
            output_path,
            protected_paths=(
                ("benchmark input", args.input),
                ("mesh reference", args.mesh_reference),
            ),
        )
        report = run_public_stageii_benchmark(
            args.input,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            mesh_reference_path=args.mesh_reference,
            mesh_support_base_dir=_mesh_support_base_dir(args),
            mesh_chunk_size=args.mesh_chunk_size,
            mesh_chunk_overlap=args.mesh_chunk_overlap,
            lean_benchmark=args.lean_benchmark,
        )
        payload = write_benchmark_report(report, output_path)
    except BENCHMARK_CLI_ERROR_TYPES as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
