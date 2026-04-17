import argparse
import json
from pathlib import Path

from moshpp.mosh_head import MoSh, run_moshpp_once
from save_smplx_verts import export_stageii_meshes
from utils.stageii_benchmark import (
    default_benchmark_output_path,
    run_public_stageii_benchmark,
    write_benchmark_report,
)
from utils.script_utils import (
    default_stageii_output_paths,
    planned_stageii_output_path_from_overrides,
    resolve_stageii_model_path,
)


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

BENCHMARK_CLI_ERROR_TYPES = (KeyError, ValueError, OSError, ImportError, ModuleNotFoundError)
OFFICIAL_RUN_CLI_ERROR_TYPES = (ValueError, OSError)
MESH_EXPORT_CLI_ERROR_TYPES = (KeyError, ValueError, OSError)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the official single-sequence MoSh entrypoint with repeatable dotlist "
            "overrides, then optionally export mesh outputs and benchmark the produced stageii.pkl."
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
        "--export-mesh",
        action="store_true",
        help="Export OBJ/PC2 from the produced stageii.pkl after the official run finishes.",
    )
    parser.add_argument(
        "--mesh-output-dir",
        default=None,
        help="Optional directory used for exported OBJ/PC2 when --export-mesh is enabled.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only run the official entrypoint and report the produced stageii path.",
    )
    parser.add_argument(
        "--benchmark-output",
        default=None,
        help="Optional JSON output path for the benchmark report. Defaults to a *_benchmark.json file next to the produced stageii.pkl.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs forwarded to the stageii benchmark.")
    parser.add_argument("--measured-runs", type=int, default=5, help="Measured runs forwarded to the stageii benchmark.")
    mesh_reference_group = parser.add_mutually_exclusive_group()
    mesh_reference_group.add_argument(
        "--mesh-reference",
        default=None,
        help="Optional baseline stageii.pkl or .pc2/.pc16 passed through to benchmark_stageii_public.",
    )
    mesh_reference_group.add_argument(
        "--mesh-reference-output-suffix",
        default=None,
        help=(
            "Optional suffix used to resolve a baseline stageii path under the same work dir and mocap "
            "basename logic as the current run, e.g. _baseline while the candidate uses _candidate."
        ),
    )
    parser.add_argument(
        "--mesh-support-base-dir",
        default=None,
        help="Optional support_files root used by mesh export/comparison; defaults to --support-base-dir.",
    )
    parser.add_argument("--mesh-chunk-size", type=int, default=None, help="Optional mesh comparison chunk-size override.")
    parser.add_argument(
        "--mesh-chunk-overlap",
        type=int,
        default=None,
        help="Optional mesh comparison chunk-overlap override paired with --mesh-chunk-size.",
    )
    parser.add_argument(
        "--expected-stageii-path",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-benchmark-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-mesh-obj-path",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-mesh-pc2-path",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _cfg_overrides(parser, args, *, output_suffix=None):
    overrides = _collect_cfg_overrides(parser, args)
    suffix = args.output_suffix if output_suffix is None else output_suffix
    if suffix:
        basename = overrides.get("mocap.basename")
        if basename is None:
            basename = Path(args.mocap_fname).stem
        overrides["mocap.basename"] = f"{basename}{suffix}"

    overrides["mocap.fname"] = args.mocap_fname
    overrides["dirs.support_base_dir"] = args.support_base_dir
    overrides["dirs.work_base_dir"] = args.work_base_dir
    overrides["runtime.backend"] = args.backend
    return overrides


def _collect_cfg_overrides(parser, args):
    overrides = dict(OFFICIAL_PRESETS.get(args.preset, {}))
    for entry in args.cfg:
        if "=" not in entry:
            parser.error(f"--cfg entries must be KEY=VALUE, got: {entry}")
        key, value = entry.split("=", 1)
        if not key:
            parser.error(f"--cfg entries must include a non-empty key, got: {entry}")
        overrides[key] = value
    return overrides


def _validate_mesh_reference_output_suffix(parser, args):
    if args.mesh_reference_output_suffix is None:
        return
    if "dirs.stageii_fname" in _collect_cfg_overrides(parser, args):
        parser.error(
            "--mesh-reference-output-suffix cannot be used when --cfg/preset overrides dirs.stageii_fname; "
            "pass --mesh-reference explicitly instead"
        )


def _validate_mesh_cli_args(parser, args):
    if args.mesh_output_dir is not None and not args.export_mesh:
        parser.error("--mesh-output-dir requires --export-mesh")
    if args.benchmark_output is not None and args.skip_benchmark:
        parser.error("--benchmark-output requires benchmark to be enabled")
    if args.warmup_runs != 1 and args.skip_benchmark:
        parser.error("--warmup-runs requires benchmark to be enabled")
    if args.measured_runs != 5 and args.skip_benchmark:
        parser.error("--measured-runs requires benchmark to be enabled")
    if args.mesh_reference is not None and args.skip_benchmark:
        parser.error("--mesh-reference requires benchmark to be enabled")
    if args.mesh_reference_output_suffix is not None and args.skip_benchmark:
        parser.error("--mesh-reference-output-suffix requires benchmark to be enabled")

    has_mesh_reference = args.mesh_reference is not None or args.mesh_reference_output_suffix is not None
    if args.mesh_chunk_size is not None:
        if args.skip_benchmark:
            parser.error("--mesh-chunk-size requires benchmark to be enabled")
        if not has_mesh_reference:
            parser.error("--mesh-chunk-size requires --mesh-reference or --mesh-reference-output-suffix")
    if args.mesh_chunk_overlap is not None:
        if args.skip_benchmark:
            parser.error("--mesh-chunk-overlap requires benchmark to be enabled")
        if not has_mesh_reference:
            parser.error("--mesh-chunk-overlap requires --mesh-reference or --mesh-reference-output-suffix")
        if args.mesh_chunk_size is None:
            parser.error("--mesh-chunk-overlap requires --mesh-chunk-size")
    if args.mesh_support_base_dir is not None and not args.export_mesh and not has_mesh_reference:
        parser.error("--mesh-support-base-dir requires --export-mesh or --mesh-reference/--mesh-reference-output-suffix")


def _validate_internal_contract_args(parser, args):
    if args.expected_benchmark_output is not None and args.skip_benchmark:
        parser.error("--expected-benchmark-output requires benchmark to be enabled")

    has_expected_mesh_obj_path = args.expected_mesh_obj_path is not None
    has_expected_mesh_pc2_path = args.expected_mesh_pc2_path is not None
    has_expected_mesh_paths = has_expected_mesh_obj_path or has_expected_mesh_pc2_path
    if has_expected_mesh_paths and not args.export_mesh:
        parser.error("--expected-mesh-obj-path/--expected-mesh-pc2-path require --export-mesh")
    if has_expected_mesh_obj_path != has_expected_mesh_pc2_path:
        parser.error("--expected-mesh-obj-path and --expected-mesh-pc2-path must be provided together")


def _resolve_mesh_reference_path(parser, args):
    if args.mesh_reference is not None:
        return args.mesh_reference
    if args.mesh_reference_output_suffix is None:
        return None

    reference_overrides = _cfg_overrides(
        parser,
        args,
        output_suffix=args.mesh_reference_output_suffix,
    )
    return str(planned_stageii_output_path_from_overrides(reference_overrides))


def _planned_stageii_output_path(parser, args):
    if args.expected_stageii_path is not None:
        return Path(str(args.expected_stageii_path))
    try:
        return planned_stageii_output_path_from_overrides(_cfg_overrides(parser, args))
    except ValueError:
        return None


def _normalized_path(path):
    return Path(path).expanduser().resolve(strict=False)


def _validate_mesh_reference_path(parser, *, stageii_path, mesh_reference_path):
    if mesh_reference_path is None:
        return
    if _normalized_path(mesh_reference_path) == _normalized_path(stageii_path):
        parser.error(
            "mesh reference resolves to the current stageii output; "
            "pass a distinct baseline stageii.pkl or .pc2/.pc16 reference"
        )


def _preflight_planned_output_contracts(parser, args):
    planned_stageii_path = _planned_stageii_output_path(parser, args)
    mesh_reference_path = None

    if not args.skip_benchmark:
        mesh_reference_path = _resolve_mesh_reference_path(parser, args)
    if planned_stageii_path is None:
        return mesh_reference_path

    if not args.skip_benchmark:
        benchmark_output_path = _resolve_benchmark_output_path(
            planned_stageii_path,
            output_path=args.benchmark_output,
        )
        _validate_expected_output_path(
            benchmark_output_path,
            expected_path=args.expected_benchmark_output,
            label="benchmark output",
        )
    if args.export_mesh:
        obj_out, pc2_out = _resolve_mesh_export_paths(
            planned_stageii_path,
            output_dir=args.mesh_output_dir,
        )
        _validate_expected_output_path(
            obj_out,
            expected_path=args.expected_mesh_obj_path,
            label="mesh export obj",
        )
        _validate_expected_output_path(
            pc2_out,
            expected_path=args.expected_mesh_pc2_path,
            label="mesh export pc2",
        )

    if mesh_reference_path is None:
        return None

    if planned_stageii_path is not None:
        _validate_mesh_reference_path(
            parser,
            stageii_path=planned_stageii_path,
            mesh_reference_path=mesh_reference_path,
        )
    return mesh_reference_path


def _mesh_support_base_dir(args):
    return args.mesh_support_base_dir or args.support_base_dir


def _require_stageii_output(stageii_path):
    if not stageii_path.exists():
        raise FileNotFoundError(f"run_moshpp_once did not produce the expected stageii file: {stageii_path}")


def _validate_expected_stageii_path(stageii_path, *, expected_stageii_path=None):
    if expected_stageii_path is None:
        return
    if _normalized_path(stageii_path) != _normalized_path(expected_stageii_path):
        raise ValueError(
            "prepared stageii output path drifted from expected plan: "
            f"expected {expected_stageii_path} but got {stageii_path}"
        )


def _validate_expected_output_path(path, *, expected_path=None, label):
    if expected_path is None:
        return
    if _normalized_path(path) != _normalized_path(expected_path):
        raise ValueError(
            f"resolved {label} path drifted from expected plan: "
            f"expected {expected_path} but got {path}"
        )


def _validate_returned_output_path(path, *, requested_path, label):
    if not path:
        raise ValueError(f"{label} is missing from helper return payload")
    if _normalized_path(path) != _normalized_path(requested_path):
        raise ValueError(
            f"{label} drifted from requested output path: "
            f"expected {requested_path} but got {path}"
        )


def _resolve_mesh_export_paths(stageii_path, *, output_dir=None):
    obj_out, pc2_out = default_stageii_output_paths(str(stageii_path))
    if output_dir is None:
        return obj_out, pc2_out

    output_dir = Path(output_dir)
    return str(output_dir / Path(obj_out).name), str(output_dir / Path(pc2_out).name)


def _resolve_benchmark_output_path(stageii_path, *, output_path=None):
    if output_path is not None:
        return Path(str(output_path))
    return Path(default_benchmark_output_path(stageii_path))


def _export_meshes(stageii_path, args):
    model_path = resolve_stageii_model_path(
        str(stageii_path),
        support_base_dir=_mesh_support_base_dir(args),
    )
    obj_out, pc2_out = _resolve_mesh_export_paths(stageii_path, output_dir=args.mesh_output_dir)
    _validate_expected_output_path(
        obj_out,
        expected_path=args.expected_mesh_obj_path,
        label="mesh export obj",
    )
    _validate_expected_output_path(
        pc2_out,
        expected_path=args.expected_mesh_pc2_path,
        label="mesh export pc2",
    )
    obj_path, pc2_path = export_stageii_meshes(
        str(stageii_path),
        model_path=model_path,
        obj_out=obj_out,
        pc2_out=pc2_out,
    )
    _validate_returned_output_path(
        obj_path,
        requested_path=obj_out,
        label="mesh export payload obj_path",
    )
    _validate_returned_output_path(
        pc2_path,
        requested_path=pc2_out,
        label="mesh export payload pc2_path",
    )
    return {
        "obj_path": obj_path,
        "pc2_path": pc2_path,
    }


def run(argv=None, *, emit_json=True):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_mesh_cli_args(parser, args)
    _validate_internal_contract_args(parser, args)
    _validate_mesh_reference_output_suffix(parser, args)
    try:
        mesh_reference_path = _preflight_planned_output_contracts(parser, args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        cfg = MoSh.prepare_cfg(**_cfg_overrides(parser, args))
        stageii_path = Path(str(cfg.dirs.stageii_fname))
        _validate_expected_stageii_path(
            stageii_path,
            expected_stageii_path=args.expected_stageii_path,
        )
        run_moshpp_once(cfg)
        _require_stageii_output(stageii_path)
    except OFFICIAL_RUN_CLI_ERROR_TYPES as exc:
        parser.error(str(exc))

    payload = {
        "stageii_path": str(stageii_path),
        "benchmark": None,
    }

    if args.export_mesh:
        try:
            payload["mesh_export"] = _export_meshes(stageii_path, args)
        except MESH_EXPORT_CLI_ERROR_TYPES as exc:
            parser.error(str(exc))

    if not args.skip_benchmark:
        try:
            _validate_mesh_reference_path(
                parser,
                stageii_path=stageii_path,
                mesh_reference_path=mesh_reference_path,
            )
            report = run_public_stageii_benchmark(
                str(stageii_path),
                warmup_runs=args.warmup_runs,
                measured_runs=args.measured_runs,
                mesh_reference_path=mesh_reference_path,
                mesh_support_base_dir=_mesh_support_base_dir(args),
                mesh_chunk_size=args.mesh_chunk_size,
                mesh_chunk_overlap=args.mesh_chunk_overlap,
            )
            benchmark_output_path = _resolve_benchmark_output_path(
                stageii_path,
                output_path=args.benchmark_output,
            )
            _validate_expected_output_path(
                benchmark_output_path,
                expected_path=args.expected_benchmark_output,
                label="benchmark output",
            )
            report = write_benchmark_report(report, str(benchmark_output_path))
            report_path = None
            if isinstance(report, dict):
                artifact = report.get("artifact")
                if isinstance(artifact, dict):
                    report_path = artifact.get("report_path")
            _validate_returned_output_path(
                report_path,
                requested_path=benchmark_output_path,
                label="benchmark payload report_path",
            )
            payload["benchmark"] = report
        except BENCHMARK_CLI_ERROR_TYPES as exc:
            parser.error(str(exc))

    if emit_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv=None):
    return run(argv, emit_json=True)


if __name__ == "__main__":
    main()
