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
from utils.script_utils import default_stageii_output_paths, resolve_stageii_model_path


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
        help="Optional mesh comparison chunk-overlap override.",
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


def _resolve_mesh_reference_path(parser, args):
    if args.mesh_reference is not None:
        return args.mesh_reference
    if args.mesh_reference_output_suffix is None:
        return None

    reference_cfg = MoSh.prepare_cfg(
        **_cfg_overrides(
            parser,
            args,
            output_suffix=args.mesh_reference_output_suffix,
        )
    )
    return str(reference_cfg.dirs.stageii_fname)


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


def _mesh_support_base_dir(args):
    return args.mesh_support_base_dir or args.support_base_dir


def _resolve_mesh_export_paths(stageii_path, *, output_dir=None):
    obj_out, pc2_out = default_stageii_output_paths(str(stageii_path))
    if output_dir is None:
        return obj_out, pc2_out

    output_dir = Path(output_dir)
    return str(output_dir / Path(obj_out).name), str(output_dir / Path(pc2_out).name)


def _export_meshes(stageii_path, args):
    model_path = resolve_stageii_model_path(
        str(stageii_path),
        support_base_dir=_mesh_support_base_dir(args),
    )
    obj_out, pc2_out = _resolve_mesh_export_paths(stageii_path, output_dir=args.mesh_output_dir)
    obj_path, pc2_path = export_stageii_meshes(
        str(stageii_path),
        model_path=model_path,
        obj_out=obj_out,
        pc2_out=pc2_out,
    )
    return {
        "obj_path": obj_path,
        "pc2_path": pc2_path,
    }


def run(argv=None, *, emit_json=True):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_mesh_reference_output_suffix(parser, args)

    cfg = MoSh.prepare_cfg(**_cfg_overrides(parser, args))
    run_moshpp_once(cfg)

    stageii_path = Path(str(cfg.dirs.stageii_fname))
    if not stageii_path.exists():
        raise FileNotFoundError(f"run_moshpp_once did not produce the expected stageii file: {stageii_path}")

    payload = {
        "stageii_path": str(stageii_path),
        "benchmark": None,
    }

    if args.export_mesh:
        try:
            payload["mesh_export"] = _export_meshes(stageii_path, args)
        except (KeyError, ValueError) as exc:
            parser.error(str(exc))

    if not args.skip_benchmark:
        mesh_reference_path = _resolve_mesh_reference_path(parser, args)
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
        benchmark_output_path = args.benchmark_output or default_benchmark_output_path(stageii_path)
        report = write_benchmark_report(report, str(benchmark_output_path))
        payload["benchmark"] = report

    if emit_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv=None):
    return run(argv, emit_json=True)


if __name__ == "__main__":
    main()
