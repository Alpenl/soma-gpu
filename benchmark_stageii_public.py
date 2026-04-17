import argparse
import json

from utils.stageii_benchmark import run_public_stageii_benchmark, write_benchmark_report


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
        help="Optional path for the JSON report. Prints to stdout when omitted.",
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
        default="support_files",
        help="support_files root used when --mesh-reference or --input stageii pickles need relocated model paths.",
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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_public_stageii_benchmark(
        args.input,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
        mesh_reference_path=args.mesh_reference,
        mesh_support_base_dir=args.mesh_support_base_dir,
        mesh_chunk_size=args.mesh_chunk_size,
        mesh_chunk_overlap=args.mesh_chunk_overlap,
    )

    if args.output:
        payload = write_benchmark_report(report, args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
