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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run_public_stageii_benchmark(
        args.input,
        warmup_runs=args.warmup_runs,
        measured_runs=args.measured_runs,
    )

    if args.output:
        payload = write_benchmark_report(report, args.output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
