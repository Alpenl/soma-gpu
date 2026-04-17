import argparse
import json

import run_stageii_torch_official


DEFAULT_BASELINE_PRESET = "real-mcp-baseline"
DEFAULT_CANDIDATE_PRESET = "real-mcp-transvelo100-seedvelowindow"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a baseline/candidate pair through the official single-sequence MoSh "
            "entrypoint, then benchmark the candidate against the baseline mesh in one command."
        )
    )
    parser.add_argument("--mocap-fname", required=True, help="Input .c3d or .mcp sequence path.")
    parser.add_argument("--support-base-dir", required=True, help="support_files root for model/layout assets.")
    parser.add_argument("--work-base-dir", required=True, help="Work directory where stagei/stageii outputs are written.")
    parser.add_argument(
        "--backend",
        default="torch",
        help="runtime.backend forwarded to the underlying official runner. Defaults to torch.",
    )
    parser.add_argument(
        "--cfg",
        action="append",
        default=[],
        help="Repeatable shared dotlist override passed to both baseline and candidate runs.",
    )
    parser.add_argument(
        "--baseline-cfg",
        action="append",
        default=[],
        help="Repeatable dotlist override only applied to the baseline run.",
    )
    parser.add_argument(
        "--candidate-cfg",
        action="append",
        default=[],
        help="Repeatable dotlist override only applied to the candidate run.",
    )
    parser.add_argument(
        "--baseline-preset",
        choices=sorted(run_stageii_torch_official.OFFICIAL_PRESETS),
        default=DEFAULT_BASELINE_PRESET,
        help="Preset used for the baseline run. Defaults to the corrected real .mcp baseline.",
    )
    parser.add_argument(
        "--candidate-preset",
        choices=sorted(run_stageii_torch_official.OFFICIAL_PRESETS),
        default=DEFAULT_CANDIDATE_PRESET,
        help="Preset used for the candidate run. Defaults to the retained translation-friendly candidate.",
    )
    parser.add_argument(
        "--baseline-output-suffix",
        default="_baseline",
        help="Suffix appended to the baseline mocap.basename so its outputs do not collide with the candidate.",
    )
    parser.add_argument(
        "--candidate-output-suffix",
        default="_candidate",
        help="Suffix appended to the candidate mocap.basename so its outputs do not collide with the baseline.",
    )
    parser.add_argument(
        "--export-mesh",
        action="store_true",
        help="Forward mesh export to both baseline and candidate runs.",
    )
    parser.add_argument(
        "--mesh-output-dir",
        default=None,
        help="Optional shared output directory for baseline/candidate OBJ/PC2 exports.",
    )
    parser.add_argument(
        "--baseline-benchmark-output",
        default=None,
        help="Optional JSON output path for a standalone baseline benchmark. By default the baseline run skips benchmarking.",
    )
    parser.add_argument(
        "--candidate-benchmark-output",
        default=None,
        help="Optional JSON output path for the candidate benchmark report.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs forwarded to benchmarked runs when overriding the default.",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=5,
        help="Measured runs forwarded to benchmarked runs when overriding the default.",
    )
    parser.add_argument(
        "--mesh-support-base-dir",
        default=None,
        help="Optional support_files root used by mesh export and candidate mesh comparison; defaults to --support-base-dir.",
    )
    parser.add_argument("--mesh-chunk-size", type=int, default=None, help="Optional mesh comparison chunk-size override.")
    parser.add_argument(
        "--mesh-chunk-overlap",
        type=int,
        default=None,
        help="Optional mesh comparison chunk-overlap override.",
    )
    return parser


def _base_runner_args(args):
    runner_args = [
        "--mocap-fname",
        args.mocap_fname,
        "--support-base-dir",
        args.support_base_dir,
        "--work-base-dir",
        args.work_base_dir,
    ]
    if args.backend != "torch":
        runner_args.extend(["--backend", args.backend])
    return runner_args


def _append_cfg_args(runner_args, cfg_entries):
    for entry in cfg_entries:
        runner_args.extend(["--cfg", entry])


def _append_benchmark_args(runner_args, args, *, benchmark_output=None, include_mesh_reference=False):
    if benchmark_output is not None:
        runner_args.extend(["--benchmark-output", benchmark_output])
    if args.warmup_runs != 1:
        runner_args.extend(["--warmup-runs", str(args.warmup_runs)])
    if args.measured_runs != 5:
        runner_args.extend(["--measured-runs", str(args.measured_runs)])


def _append_mesh_args(runner_args, args, *, include_mesh_reference=False):
    if args.export_mesh:
        runner_args.append("--export-mesh")
        if args.mesh_output_dir is not None:
            runner_args.extend(["--mesh-output-dir", args.mesh_output_dir])
    if args.mesh_support_base_dir is not None and (args.export_mesh or include_mesh_reference):
        runner_args.extend(["--mesh-support-base-dir", args.mesh_support_base_dir])
    if not include_mesh_reference:
        return
    if args.mesh_chunk_size is not None:
        runner_args.extend(["--mesh-chunk-size", str(args.mesh_chunk_size)])
    if args.mesh_chunk_overlap is not None:
        runner_args.extend(["--mesh-chunk-overlap", str(args.mesh_chunk_overlap)])


def _build_baseline_runner_args(args):
    runner_args = _base_runner_args(args)
    runner_args.extend(["--preset", args.baseline_preset])
    runner_args.extend(["--output-suffix", args.baseline_output_suffix])
    _append_cfg_args(runner_args, args.cfg)
    _append_cfg_args(runner_args, args.baseline_cfg)
    _append_mesh_args(runner_args, args, include_mesh_reference=False)
    if args.baseline_benchmark_output is None:
        runner_args.append("--skip-benchmark")
    else:
        _append_benchmark_args(
            runner_args,
            args,
            benchmark_output=args.baseline_benchmark_output,
            include_mesh_reference=False,
        )
    return runner_args


def _build_candidate_runner_args(args, *, mesh_reference_path=None):
    runner_args = _base_runner_args(args)
    runner_args.extend(["--preset", args.candidate_preset])
    runner_args.extend(["--output-suffix", args.candidate_output_suffix])
    _append_cfg_args(runner_args, args.cfg)
    _append_cfg_args(runner_args, args.candidate_cfg)
    _append_mesh_args(runner_args, args, include_mesh_reference=True)
    _append_benchmark_args(
        runner_args,
        args,
        benchmark_output=args.candidate_benchmark_output,
        include_mesh_reference=True,
    )
    if mesh_reference_path is not None:
        runner_args.extend(["--mesh-reference", mesh_reference_path])
    else:
        runner_args.extend(["--mesh-reference-output-suffix", args.baseline_output_suffix])
    return runner_args


def _validate_mesh_cli_args(parser, args):
    if args.mesh_output_dir is not None and not args.export_mesh:
        parser.error("--mesh-output-dir requires --export-mesh")


def run(argv=None, *, emit_json=True):
    parser = build_parser()
    args = parser.parse_args(argv)

    _validate_mesh_cli_args(parser, args)
    if args.baseline_output_suffix == args.candidate_output_suffix:
        parser.error("--baseline-output-suffix and --candidate-output-suffix must be different")

    baseline_payload = run_stageii_torch_official.run(
        _build_baseline_runner_args(args),
        emit_json=False,
    )
    baseline_stageii_path = baseline_payload.get("stageii_path")
    if not baseline_stageii_path:
        raise ValueError("baseline runner did not return stageii_path")
    candidate_payload = run_stageii_torch_official.run(
        _build_candidate_runner_args(args, mesh_reference_path=baseline_stageii_path),
        emit_json=False,
    )

    payload = {
        "baseline": baseline_payload,
        "candidate": candidate_payload,
    }
    if emit_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv=None):
    return run(argv, emit_json=True)


if __name__ == "__main__":
    main()
