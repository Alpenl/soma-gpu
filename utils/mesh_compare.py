import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import render_video
from utils.mesh_io import readPC2
from utils.script_utils import load_pickle_compat, resolve_stageii_model_path
from utils.stageii_benchmark import (
    _chunk_seam_jump_l2_samples,
    _chunk_seam_plans,
    _local_transition_diagnostics_at,
    _sequence_chunk_config,
    _sequence_chunk_keep_starts,
    _summarize_numeric_samples,
    _temporal_accel_l2_samples,
    normalize_stageii_sample,
)


@dataclass
class LoadedMeshSequence:
    input_path: Path
    source_format: str
    vertices: np.ndarray
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    chunk_keep_starts: list[int] | None = None


def _normalized_path(path):
    return Path(path).expanduser().resolve(strict=False)


def _explicit_chunk_config(chunk_size=None, chunk_overlap=None):
    if chunk_size is None:
        if chunk_overlap is not None:
            raise ValueError("chunk_overlap requires chunk_size")
        return None
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunk_overlap = int(chunk_overlap or 0)
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    return chunk_size, chunk_overlap


def _fallback_stageii_model_path(sample_path, *, support_base_dir=None):
    if not support_base_dir:
        return None
    baseline = normalize_stageii_sample(sample_path)
    candidate_types = []
    for model_type in (baseline.surface_model_type, "smplx"):
        if model_type and model_type != "unknown" and model_type not in candidate_types:
            candidate_types.append(model_type)
    candidate_genders = []
    for gender in (baseline.gender, "neutral"):
        if gender and gender != "unknown" and gender not in candidate_genders:
            candidate_genders.append(gender)

    fallback_candidates = []
    for model_type in candidate_types:
        for gender in candidate_genders:
            base_dir = Path(support_base_dir) / model_type / gender
            for suffix in (".npz", ".pkl"):
                candidate = base_dir / f"model{suffix}"
                if candidate.exists():
                    return str(candidate)
                fallback_candidates.append(str(candidate))
    if fallback_candidates:
        return fallback_candidates[0]
    return None


def _stageii_model_path(sample_path, *, support_base_dir=None):
    resolved_path = None
    resolve_error = None
    try:
        resolved_path = resolve_stageii_model_path(str(sample_path), support_base_dir=support_base_dir)
    except (KeyError, ValueError) as exc:
        resolve_error = exc

    if resolved_path is not None and Path(resolved_path).exists():
        return resolved_path

    fallback = _fallback_stageii_model_path(sample_path, support_base_dir=support_base_dir)
    if fallback is not None:
        return fallback
    if resolved_path is not None:
        return resolved_path
    raise resolve_error


def _loaded_chunk_config(input_path, *, explicit_chunk_config=None):
    if explicit_chunk_config is not None:
        return explicit_chunk_config
    if Path(input_path).suffix.lower() != ".pkl":
        return None
    return _sequence_chunk_config(load_pickle_compat(input_path))


def _loaded_chunk_keep_starts(stageii_data, *, chunk_config, explicit_chunk_config=None):
    if explicit_chunk_config is not None or chunk_config is None:
        return None
    total_frames = None
    for key in ("trans", "fullpose"):
        values = stageii_data.get(key)
        if values is not None:
            total_frames = int(np.asarray(values).shape[0])
            break
    if total_frames is None:
        raise ValueError("stageii sample must contain trans or fullpose to resolve chunk keep_starts")
    chunk_size, chunk_overlap = chunk_config
    return _sequence_chunk_keep_starts(
        stageii_data,
        total_frames=total_frames,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )


def load_mesh_sequence(input_path, *, support_base_dir=None, chunk_size=None, chunk_overlap=None):
    input_path = Path(input_path)
    explicit_chunk_config = _explicit_chunk_config(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    suffix = input_path.suffix.lower()
    stageii_data = None

    if suffix in {".pc2", ".pc16"}:
        mesh_cache = readPC2(str(input_path), float16=suffix == ".pc16")
        vertices = np.asarray(mesh_cache["V"], dtype=np.float32)
        source_format = suffix[1:]
    elif suffix == ".pkl":
        stageii_data = load_pickle_compat(input_path)
        model_path = _stageii_model_path(input_path, support_base_dir=support_base_dir)
        model = render_video.load_render_model(model_path)
        vertices = np.asarray(render_video.load_vertices(str(input_path), model), dtype=np.float32)
        source_format = "stageii_pkl"
    else:
        raise ValueError(f"Unsupported mesh input format: {input_path}")

    chunk_config = (
        _loaded_chunk_config(input_path, explicit_chunk_config=explicit_chunk_config)
        if stageii_data is None
        else (
            explicit_chunk_config
            if explicit_chunk_config is not None
            else _sequence_chunk_config(stageii_data)
        )
    )
    if chunk_config is None:
        resolved_chunk_size = None
        resolved_chunk_overlap = None
        resolved_chunk_keep_starts = None
    else:
        resolved_chunk_size, resolved_chunk_overlap = chunk_config
        resolved_chunk_keep_starts = (
            None
            if stageii_data is None
            else _loaded_chunk_keep_starts(
                stageii_data,
                chunk_config=chunk_config,
                explicit_chunk_config=explicit_chunk_config,
            )
        )

    return LoadedMeshSequence(
        input_path=input_path,
        source_format=source_format,
        vertices=vertices,
        chunk_size=resolved_chunk_size,
        chunk_overlap=resolved_chunk_overlap,
        chunk_keep_starts=resolved_chunk_keep_starts,
    )


def _frame_delta_l2_samples(reference_vertices, candidate_vertices):
    return [float(value) for value in _frame_delta_l2_array(reference_vertices, candidate_vertices).tolist()]


def _frame_delta_l2_array(reference_vertices, candidate_vertices):
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    candidate_vertices = np.asarray(candidate_vertices, dtype=np.float64)
    if reference_vertices.shape != candidate_vertices.shape:
        raise ValueError(
            "reference and candidate meshes must have the same shape, got "
            f"{reference_vertices.shape} != {candidate_vertices.shape}"
        )

    delta = np.nan_to_num(candidate_vertices - reference_vertices, copy=False)
    delta = delta.reshape(delta.shape[0], -1)
    return np.linalg.norm(delta, axis=1)


def _summarize_frame_delta_window(frame_delta_l2, *, seam_index, window_size):
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("frame_delta_window must be > 0")

    frame_delta_l2 = np.asarray(frame_delta_l2, dtype=np.float64)
    if seam_index < 0 or seam_index >= frame_delta_l2.shape[0]:
        raise ValueError(f"seam_index {seam_index} is out of bounds for frame_delta_l2 with {frame_delta_l2.shape[0]} frames")

    window_end = min(int(seam_index) + window_size, frame_delta_l2.shape[0])
    window_values = frame_delta_l2[int(seam_index):window_end]
    summary = _summarize_numeric_samples([float(value) for value in window_values.tolist()], include_samples=False)
    if summary is None:
        return None
    peak_offset = int(np.argmax(window_values))
    summary.update(
        {
            "window_start": int(seam_index),
            "window_end": int(window_end),
            "seam_frame_delta_l2": float(window_values[0]),
            "last_frame_delta_l2": float(window_values[-1]),
            "peak_frame_index": int(seam_index) + peak_offset,
            "peak_offset": peak_offset,
        }
    )
    return summary


def summarize_mesh_chunk_seam_diagnostics(sequence):
    if sequence.chunk_size is None:
        return None

    rows = []
    for seam_plan in _chunk_seam_plans(
        int(sequence.vertices.shape[0]),
        chunk_size=sequence.chunk_size,
        overlap=sequence.chunk_overlap or 0,
        keep_starts=sequence.chunk_keep_starts,
    ):
        rows.append(
            {
                **seam_plan,
                **_local_transition_diagnostics_at(sequence.vertices, int(seam_plan["seam_index"])),
            }
        )
    return {
        "chunk_size": sequence.chunk_size,
        "chunk_overlap": sequence.chunk_overlap,
        "chunk_keep_starts": sequence.chunk_keep_starts,
        "rows": rows,
    }


def compare_mesh_chunk_seam_diagnostics(
    reference_path,
    candidate_path,
    *,
    support_base_dir=None,
    chunk_size=None,
    chunk_overlap=None,
    frame_delta_window=None,
):
    if _normalized_path(reference_path) == _normalized_path(candidate_path):
        raise ValueError(f"candidate_path resolves to reference_path: {reference_path}")

    reference = load_mesh_sequence(
        reference_path,
        support_base_dir=support_base_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    candidate = load_mesh_sequence(
        candidate_path,
        support_base_dir=support_base_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    frame_delta_l2 = _frame_delta_l2_array(reference.vertices, candidate.vertices)
    candidate_diagnostics = summarize_mesh_chunk_seam_diagnostics(candidate)
    if candidate_diagnostics is None:
        return None

    resolved_frame_delta_window = (
        int(candidate.chunk_overlap or 0) if frame_delta_window is None else int(frame_delta_window)
    )
    if resolved_frame_delta_window < 0:
        raise ValueError("frame_delta_window must be >= 0")

    rows = []
    for row in candidate_diagnostics["rows"]:
        seam_index = int(row["seam_index"])
        reference_metrics = _local_transition_diagnostics_at(reference.vertices, seam_index)
        delta_metrics = {}
        for key, candidate_value in row.items():
            if key in {"chunk_index", "row_start", "row_end", "keep_start", "trim_count", "seam_index"}:
                continue
            reference_value = reference_metrics.get(key)
            if candidate_value is None or reference_value is None:
                delta_metrics[key] = None
            else:
                delta_metrics[key] = float(candidate_value) - float(reference_value)
        compared_row = {
            "chunk_index": row["chunk_index"],
            "row_start": row["row_start"],
            "row_end": row["row_end"],
            "keep_start": row["keep_start"],
            "trim_count": row["trim_count"],
            "seam_index": seam_index,
            "candidate": {
                key: row[key]
                for key in (
                    "prev_frame_delta_l2",
                    "seam_jump_l2",
                    "next_frame_delta_l2",
                    "pre_accel_l2",
                    "post_accel_l2",
                )
            },
            "reference": reference_metrics,
            "delta": delta_metrics,
            "frame_delta_window": None,
        }
        if resolved_frame_delta_window > 0:
            compared_row["frame_delta_window"] = _summarize_frame_delta_window(
                frame_delta_l2,
                seam_index=seam_index,
                window_size=resolved_frame_delta_window,
            )
        rows.append(compared_row)

    return {
        "chunk_size": candidate_diagnostics["chunk_size"],
        "chunk_overlap": candidate_diagnostics["chunk_overlap"],
        "chunk_keep_starts": candidate_diagnostics["chunk_keep_starts"],
        "frame_delta_window": resolved_frame_delta_window,
        "rows": rows,
    }


def summarize_mesh_sequence(sequence):
    summary = {
        "input_path": str(sequence.input_path),
        "source_format": sequence.source_format,
        "frames": int(sequence.vertices.shape[0]),
        "vertices_per_frame": int(sequence.vertices.shape[1]),
        "chunk_size": sequence.chunk_size,
        "chunk_overlap": sequence.chunk_overlap,
        "mesh_accel_l2": _summarize_numeric_samples(
            _temporal_accel_l2_samples(sequence.vertices),
            include_samples=False,
        ),
        "mesh_seam_jump_l2": None,
    }
    if sequence.chunk_size is not None:
        summary["mesh_seam_jump_l2"] = _summarize_numeric_samples(
            _chunk_seam_jump_l2_samples(
                sequence.vertices,
                chunk_size=sequence.chunk_size,
                overlap=sequence.chunk_overlap or 0,
                keep_starts=sequence.chunk_keep_starts,
            ),
            include_samples=False,
        )
    return summary


def compare_mesh_sequences(
    reference_path,
    candidate_path,
    *,
    support_base_dir=None,
    chunk_size=None,
    chunk_overlap=None,
):
    if _normalized_path(reference_path) == _normalized_path(candidate_path):
        raise ValueError(f"candidate_path resolves to reference_path: {reference_path}")

    reference = load_mesh_sequence(
        reference_path,
        support_base_dir=support_base_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    candidate = load_mesh_sequence(
        candidate_path,
        support_base_dir=support_base_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return {
        "reference": summarize_mesh_sequence(reference),
        "candidate": summarize_mesh_sequence(candidate),
        "frame_delta_l2": _summarize_numeric_samples(
            _frame_delta_l2_samples(reference.vertices, candidate.vertices),
            include_samples=False,
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare mesh-space temporal summaries between two stageii.pkl or PC2 sequences."
        )
    )
    parser.add_argument("--reference", required=True, help="Reference stageii.pkl or .pc2/.pc16 path.")
    parser.add_argument("--candidate", required=True, help="Candidate stageii.pkl or .pc2/.pc16 path.")
    parser.add_argument(
        "--support-base-dir",
        help="Optional support_files root used to relocate stageii surface-model paths.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Optional chunk size override, mainly for raw PC2 inputs without runtime metadata.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Optional chunk overlap override paired with --chunk-size.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. When omitted, the report is printed to stdout.",
    )
    return parser


def _validate_cli_args(parser, args):
    if args.chunk_size is not None and args.chunk_size <= 0:
        parser.error("--chunk-size must be > 0")
    if args.chunk_overlap is not None:
        if args.chunk_size is None:
            parser.error("--chunk-overlap requires --chunk-size")
        if args.chunk_overlap < 0:
            parser.error("--chunk-overlap must be >= 0")
    if args.output:
        output_path = _normalized_path(args.output)
        if output_path == _normalized_path(args.reference):
            parser.error(f"--output resolves to reference path: {args.reference}")
        if output_path == _normalized_path(args.candidate):
            parser.error(f"--output resolves to candidate path: {args.candidate}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    try:
        report = compare_mesh_sequences(
            args.reference,
            args.candidate,
            support_base_dir=args.support_base_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except ValueError as exc:
        parser.error(str(exc))
    report_json = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_json + "\n")
    else:
        print(report_json)
    return report


if __name__ == "__main__":
    main()
