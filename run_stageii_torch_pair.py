import argparse
import json
from pathlib import Path

import run_stageii_torch_official
from utils.script_utils import planned_stageii_output_path_from_overrides


DEFAULT_BASELINE_PRESET = "real-mcp-baseline"
DEFAULT_CANDIDATE_PRESET = "real-mcp-transvelo100-seedvelowindow"
PAIR_RUNNER_CLI_ERROR_TYPES = (KeyError, ValueError, OSError, ImportError, ModuleNotFoundError)


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


def _single_runner_cfg_namespace(args, *, preset, cfg_entries, output_suffix):
    return argparse.Namespace(
        preset=preset,
        cfg=list(cfg_entries),
        output_suffix=output_suffix,
        mocap_fname=args.mocap_fname,
        support_base_dir=args.support_base_dir,
        work_base_dir=args.work_base_dir,
        backend=args.backend,
    )


def _planned_stageii_output_overrides(parser, args, *, preset, side_cfg_entries, output_suffix):
    runner_namespace = _single_runner_cfg_namespace(
        args,
        preset=preset,
        cfg_entries=[*args.cfg, *side_cfg_entries],
        output_suffix=output_suffix,
    )
    return run_stageii_torch_official._cfg_overrides(parser, runner_namespace)


def _normalized_path(path):
    return Path(path).expanduser().resolve(strict=False)


def _planned_stageii_output_paths(parser, args):
    baseline_stageii_path = planned_stageii_output_path_from_overrides(
        _planned_stageii_output_overrides(
            parser,
            args,
            preset=args.baseline_preset,
            side_cfg_entries=args.baseline_cfg,
            output_suffix=args.baseline_output_suffix,
        )
    )
    candidate_stageii_path = planned_stageii_output_path_from_overrides(
        _planned_stageii_output_overrides(
            parser,
            args,
            preset=args.candidate_preset,
            side_cfg_entries=args.candidate_cfg,
            output_suffix=args.candidate_output_suffix,
        )
    )
    return baseline_stageii_path, candidate_stageii_path


def _validate_distinct_stageii_output_paths(parser, *, baseline_stageii_path, candidate_stageii_path):
    if _normalized_path(baseline_stageii_path) == _normalized_path(candidate_stageii_path):
        parser.error(
            "baseline and candidate resolve to the same stageii output path; "
            "adjust output suffixes or explicit basename/stageii path overrides"
        )


def _validate_distinct_mesh_output_paths(parser, args, *, baseline_stageii_path, candidate_stageii_path):
    if not args.export_mesh or args.mesh_output_dir is None:
        return
    baseline_obj_path, baseline_pc2_path = _planned_mesh_output_paths(
        args,
        stageii_path=baseline_stageii_path,
    )
    candidate_obj_path, candidate_pc2_path = _planned_mesh_output_paths(
        args,
        stageii_path=candidate_stageii_path,
    )
    if (
        _normalized_path(baseline_obj_path) == _normalized_path(candidate_obj_path)
        or _normalized_path(baseline_pc2_path) == _normalized_path(candidate_pc2_path)
    ):
        parser.error(
            "baseline and candidate resolve to the same mesh export output path under --mesh-output-dir; "
            "adjust stageii basenames/paths or choose separate export directories"
        )


def _planned_mesh_output_paths(args, *, stageii_path):
    return run_stageii_torch_official._resolve_mesh_export_paths(
        stageii_path,
        output_dir=args.mesh_output_dir,
    )


def _planned_candidate_benchmark_output_path(args, *, candidate_stageii_path):
    return Path(
        args.candidate_benchmark_output
        or run_stageii_torch_official.default_benchmark_output_path(candidate_stageii_path)
    )


def _validate_distinct_benchmark_output_paths(parser, args, *, candidate_stageii_path):
    if args.baseline_benchmark_output is None:
        return
    baseline_benchmark_output = Path(args.baseline_benchmark_output)
    candidate_benchmark_output = _planned_candidate_benchmark_output_path(
        args,
        candidate_stageii_path=candidate_stageii_path,
    )
    if _normalized_path(baseline_benchmark_output) == _normalized_path(candidate_benchmark_output):
        parser.error(
            "baseline and candidate resolve to the same benchmark output path; "
            "adjust benchmark-output paths or candidate stageii basename/path"
        )


def _require_stageii_path(payload, *, label):
    stageii_path = payload.get("stageii_path")
    if not stageii_path:
        raise ValueError(f"{label} runner did not return stageii_path")
    return stageii_path


def _benchmark_report_path(payload):
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        return None
    artifact = benchmark.get("artifact")
    if not isinstance(artifact, dict):
        return None
    report_path = artifact.get("report_path")
    if not report_path:
        return None
    return Path(report_path)


def _require_benchmark_report_path(payload, *, label):
    report_path = _benchmark_report_path(payload)
    if report_path is None:
        raise ValueError(f"{label} runner did not return benchmark.artifact.report_path")
    return report_path


def _require_mesh_export_paths(payload, *, label):
    mesh_export = payload.get("mesh_export")
    if not isinstance(mesh_export, dict):
        raise ValueError(f"{label} runner did not return mesh_export.obj_path/pc2_path")
    obj_path = mesh_export.get("obj_path")
    pc2_path = mesh_export.get("pc2_path")
    if not obj_path or not pc2_path:
        raise ValueError(f"{label} runner did not return mesh_export.obj_path/pc2_path")
    return obj_path, pc2_path


def _validate_baseline_actual_outputs_against_candidate_plan(
    parser,
    args,
    *,
    baseline_payload,
    candidate_stageii_path,
):
    baseline_stageii_path = _require_stageii_path(baseline_payload, label="baseline")
    if _normalized_path(baseline_stageii_path) == _normalized_path(candidate_stageii_path):
        parser.error(
            "baseline actual stageii output path collides with candidate plan; "
            "adjust explicit stageii/basename overrides or investigate underlying runner path drift"
        )

    if args.export_mesh:
        candidate_obj_path, candidate_pc2_path = _planned_mesh_output_paths(
            args,
            stageii_path=candidate_stageii_path,
        )
        baseline_obj_path, baseline_pc2_path = _require_mesh_export_paths(
            baseline_payload,
            label="baseline",
        )
        if (
            _normalized_path(baseline_obj_path) == _normalized_path(candidate_obj_path)
        ) or (
            _normalized_path(baseline_pc2_path) == _normalized_path(candidate_pc2_path)
        ):
            parser.error(
                "baseline actual mesh export output collides with candidate plan; "
                "adjust stageii basenames/paths, export directories, or investigate underlying runner path drift"
            )

    if args.baseline_benchmark_output is not None:
        baseline_benchmark_output = _require_benchmark_report_path(
            baseline_payload,
            label="baseline",
        )
        candidate_benchmark_output = _planned_candidate_benchmark_output_path(
            args,
            candidate_stageii_path=candidate_stageii_path,
        )
        if _normalized_path(baseline_benchmark_output) == _normalized_path(candidate_benchmark_output):
            parser.error(
                "baseline actual benchmark output path collides with candidate plan; "
                "adjust benchmark outputs, candidate stageii basename/path, or investigate underlying runner path drift"
            )

    return baseline_stageii_path


def run(argv=None, *, emit_json=True):
    parser = build_parser()
    args = parser.parse_args(argv)

    _validate_mesh_cli_args(parser, args)
    if args.baseline_output_suffix == args.candidate_output_suffix:
        parser.error("--baseline-output-suffix and --candidate-output-suffix must be different")

    try:
        baseline_stageii_path, candidate_stageii_path = _planned_stageii_output_paths(parser, args)
        _validate_distinct_stageii_output_paths(
            parser,
            baseline_stageii_path=baseline_stageii_path,
            candidate_stageii_path=candidate_stageii_path,
        )
        _validate_distinct_mesh_output_paths(
            parser,
            args,
            baseline_stageii_path=baseline_stageii_path,
            candidate_stageii_path=candidate_stageii_path,
        )
        _validate_distinct_benchmark_output_paths(
            parser,
            args,
            candidate_stageii_path=candidate_stageii_path,
        )
        baseline_payload = run_stageii_torch_official.run(
            _build_baseline_runner_args(args),
            emit_json=False,
        )
        baseline_stageii_path = _validate_baseline_actual_outputs_against_candidate_plan(
            parser,
            args,
            baseline_payload=baseline_payload,
            candidate_stageii_path=candidate_stageii_path,
        )
        candidate_payload = run_stageii_torch_official.run(
            _build_candidate_runner_args(args, mesh_reference_path=baseline_stageii_path),
            emit_json=False,
        )
        _require_stageii_path(candidate_payload, label="candidate")
        if args.export_mesh:
            _require_mesh_export_paths(candidate_payload, label="candidate")
        _require_benchmark_report_path(candidate_payload, label="candidate")
    except PAIR_RUNNER_CLI_ERROR_TYPES as exc:
        parser.error(str(exc))

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
