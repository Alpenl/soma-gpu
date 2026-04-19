import copy
import hashlib
import importlib.util
import io
import json
import platform
import pickle
import statistics
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np


PREVIEW_RENDER_BENCHMARK_WORKLOAD = {
    "arch": "cpu",
    "fps": 30,
    "width": 128,
    "height": 128,
}

REFERENCE_STAGEII_POSE_WINDOW_SIZE = 15

_POSE_REGION_NAMES = ("body_pose", "left_hand_pose", "right_hand_pose", "all_hands_pose")
_POSE_REGION_METRIC_NAMES = tuple(
    metric_name
    for region_name in _POSE_REGION_NAMES
    for metric_name in (f"{region_name}_frame_delta_l2", f"{region_name}_jitter_l2")
)


@dataclass
class NormalizedStageIISample:
    sample_path: Path
    sample_format: str
    gender: str
    surface_model_type: str
    stageii_elapsed_s: float | None
    mocap_frame_rate: float
    mocap_time_length: int
    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    markers_latent: np.ndarray
    latent_labels: list[str]
    markers_obs: np.ndarray
    marker_labels: list[str]


def _load_pickle_compat(sample_path):
    sample_path = Path(sample_path)
    with sample_path.open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _safe_find_spec(module_name):
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return None


def _string_list(values):
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return [str(value) for value in values]


def _coerce_2d(array_like):
    array = np.asarray(array_like, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :]
    return array


def _coerce_3d(array_like):
    array = np.asarray(array_like, dtype=np.float32)
    if array.ndim == 2:
        return array[None, :, :]
    return array


def _optional_float(value):
    if value is None:
        return None
    return float(value)


def _first_label_row(labels):
    if labels is None:
        return []
    if isinstance(labels, np.ndarray):
        if labels.ndim == 1:
            return _string_list(labels)
        if labels.ndim >= 2:
            return _string_list(labels[0])
    if isinstance(labels, (list, tuple)) and labels:
        first = labels[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return _string_list(first)
    return _string_list(labels)


def _labels_for_frame(labels, frame_idx):
    if labels is None:
        return []
    if isinstance(labels, np.ndarray):
        if labels.ndim == 1:
            return _string_list(labels)
        if labels.ndim >= 2 and frame_idx < labels.shape[0]:
            return _string_list(labels[frame_idx])
        return []
    if isinstance(labels, (list, tuple)):
        if not labels:
            return []
        first = labels[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            if frame_idx < len(labels):
                return _string_list(labels[frame_idx])
            return []
    return _string_list(labels)


def _get_nested(mapping, first_key, second_key, default="unknown"):
    try:
        first = mapping[first_key]
    except Exception:
        return default
    try:
        return str(first[second_key])
    except Exception:
        return default


def normalize_stageii_sample(sample_path):
    sample_path = Path(sample_path)
    data = _load_pickle_compat(sample_path)

    if "fullpose" in data and "trans" in data:
        stageii_debug = data.get("stageii_debug_details", {})
        cfg = stageii_debug.get("cfg", {})
        normalized = NormalizedStageIISample(
            sample_path=sample_path,
            sample_format="stageii_pkl",
            gender=_get_nested(cfg, "surface_model", "gender", data.get("gender", "unknown")),
            surface_model_type=_get_nested(
                cfg, "surface_model", "type", data.get("surface_model_type", "unknown")
            ),
            stageii_elapsed_s=_optional_float(stageii_debug.get("stageii_elapsed_time")),
            mocap_frame_rate=float(stageii_debug.get("mocap_frame_rate", data.get("mocap_frame_rate", 0.0))),
            mocap_time_length=int(stageii_debug.get("mocap_time_length", len(data["trans"]))),
            poses=_coerce_2d(data["fullpose"]),
            trans=_coerce_2d(data["trans"]),
            betas=_coerce_2d(data["betas"]),
            markers_latent=np.asarray(data["markers_latent"], dtype=np.float32),
            latent_labels=_string_list(data["latent_labels"]),
            markers_obs=_coerce_3d(stageii_debug.get("markers_obs", data.get("markers_obs"))),
            marker_labels=_first_label_row(stageii_debug.get("labels_obs", data.get("labels_obs"))),
        )
    else:
        cfg = data.get("ps", {})
        normalized = NormalizedStageIISample(
            sample_path=sample_path,
            sample_format="legacy_stageii_pkl",
            gender=str(cfg.get("gender", "unknown")),
            surface_model_type=str(cfg.get("fitting_model", "unknown")),
            stageii_elapsed_s=None,
            mocap_frame_rate=float(data.get("mocap_framerate", 0.0)),
            mocap_time_length=int(data.get("mocap_timelength", len(data["pose_est_fullposes"]))),
            poses=_coerce_2d(data["pose_est_fullposes"]),
            trans=_coerce_2d(data["pose_est_trans"]),
            betas=_coerce_2d(data["shape_est_betas"]),
            markers_latent=np.asarray(data["shape_est_lmrks"], dtype=np.float32),
            latent_labels=_string_list(data["shape_est_lmlabels"]),
            markers_obs=_coerce_3d(data["pose_est_obmrks"]),
            marker_labels=_first_label_row(data["pose_est_mrk_labels"]),
        )

    if normalized.markers_obs.shape[0] != normalized.poses.shape[0]:
        raise ValueError(
            "markers_obs frame count does not match poses frame count: "
            f"{normalized.markers_obs.shape[0]} != {normalized.poses.shape[0]}"
        )
    if normalized.trans.shape[0] != normalized.poses.shape[0]:
        raise ValueError(
            "trans frame count does not match poses frame count: "
            f"{normalized.trans.shape[0]} != {normalized.poses.shape[0]}"
        )

    return normalized


def _sha256_file(sample_path):
    digest = hashlib.sha256()
    with Path(sample_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_path(path):
    return Path(path).expanduser().resolve(strict=False)


def _validate_distinct_reference_path(*, sample_path, reference_path, reference_label):
    if reference_path is None:
        return
    if _normalized_path(sample_path) == _normalized_path(reference_path):
        raise ValueError(f"{reference_label} resolves to sample_path: {sample_path}")


def _validate_mesh_reference_path(*, sample_path, mesh_reference_path):
    _validate_distinct_reference_path(
        sample_path=sample_path,
        reference_path=mesh_reference_path,
        reference_label="mesh_reference_path",
    )


def _validate_stageii_reference_path(*, sample_path, stageii_reference_path):
    _validate_distinct_reference_path(
        sample_path=sample_path,
        reference_path=stageii_reference_path,
        reference_label="stageii_reference_path",
    )


def validate_benchmark_output_path(output_path, *, protected_paths):
    output_path = Path(output_path)
    for label, protected_path in protected_paths:
        if protected_path is None:
            continue
        if _normalized_path(output_path) == _normalized_path(protected_path):
            raise ValueError(f"benchmark output resolves to {label} path: {protected_path}")


def _max_abs_diff(lhs, rhs):
    lhs = np.nan_to_num(np.asarray(lhs, dtype=np.float64), copy=False)
    rhs = np.nan_to_num(np.asarray(rhs, dtype=np.float64), copy=False)
    return float(np.max(np.abs(lhs - rhs)))


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summarize_numeric_samples(samples, *, include_samples=True):
    if not samples:
        return None
    sample_mean = statistics.mean(samples)
    sample_stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    summary = {
        "count": len(samples),
        "mean": sample_mean,
        "stdev": sample_stdev,
        "min": min(samples),
        "max": max(samples),
        "p50": _percentile(samples, 50),
        "p90": _percentile(samples, 90),
        "p99": _percentile(samples, 99),
    }
    if include_samples:
        summary["samples"] = [float(value) for value in samples]
    return summary


def _summarize_latency_samples(latency_samples):
    return _summarize_numeric_samples(latency_samples, include_samples=True)


def _summarize_metric_ratio(numerator_summary, denominator_summary):
    if numerator_summary is None or denominator_summary is None:
        return None

    ratio = {}
    for stat_name in ("mean", "p90", "max"):
        numerator = numerator_summary.get(stat_name)
        denominator = denominator_summary.get(stat_name)
        if numerator is None or denominator is None:
            continue
        denominator = float(denominator)
        if denominator <= 0.0:
            continue
        ratio[stat_name] = float(numerator) / denominator

    return ratio or None


def _numeric_arrays(sample):
    return {
        "poses": sample.poses,
        "trans": sample.trans,
        "betas": sample.betas,
        "markers_latent": sample.markers_latent,
        "markers_obs": sample.markers_obs,
    }


def _core_numeric_arrays(sample):
    return {
        "poses": sample.poses,
        "trans": sample.trans,
        "betas": sample.betas,
        "markers_latent": sample.markers_latent,
    }


def _all_finite(sample):
    return all(np.isfinite(array).all() for array in _core_numeric_arrays(sample).values())


def _coerce_marker_frame(marker_frame):
    marker_frame = np.asarray(marker_frame, dtype=np.float32)
    if marker_frame.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if marker_frame.ndim == 1:
        marker_frame = marker_frame[None, :]
    if marker_frame.ndim != 2 or marker_frame.shape[1] != 3:
        raise ValueError(f"marker frame must have shape (M, 3), got {marker_frame.shape}")
    return marker_frame


def _select_visible_marker_rows(observed_frame, simulated_frame, *, frame_labels, latent_labels):
    if not frame_labels:
        return observed_frame, simulated_frame
    if observed_frame.shape[0] == len(frame_labels):
        return observed_frame, simulated_frame
    if observed_frame.shape[0] != len(latent_labels):
        return observed_frame, simulated_frame

    label_to_latent_id = {label: idx for idx, label in enumerate(latent_labels)}
    marker_ids = [label_to_latent_id[label] for label in frame_labels if label in label_to_latent_id]
    if not marker_ids:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, empty
    return observed_frame[marker_ids], simulated_frame[marker_ids]


def _iter_marker_frames(stageii_data):
    stageii_debug = stageii_data.get("stageii_debug_details", {})
    markers_obs = stageii_debug.get("markers_obs")
    markers_sim = stageii_debug.get("markers_sim")
    if markers_obs is not None and markers_sim is not None:
        latent_labels = _string_list(stageii_data.get("latent_labels"))
        labels_obs = stageii_debug.get("labels_obs")
        for frame_idx, (observed_frame, simulated_frame) in enumerate(zip(markers_obs, markers_sim)):
            observed_frame = _coerce_marker_frame(observed_frame)
            simulated_frame = _coerce_marker_frame(simulated_frame)
            observed_frame, simulated_frame = _select_visible_marker_rows(
                observed_frame,
                simulated_frame,
                frame_labels=_labels_for_frame(labels_obs, frame_idx),
                latent_labels=latent_labels,
            )
            yield observed_frame, simulated_frame
        return

    legacy_obs = stageii_data.get("pose_est_obmrks")
    legacy_sim = stageii_data.get("pose_est_simmrks")
    if legacy_obs is None or legacy_sim is None:
        return
    for observed_frame, simulated_frame in zip(legacy_obs, legacy_sim):
        yield _coerce_marker_frame(observed_frame), _coerce_marker_frame(simulated_frame)


def _marker_residual_l2_samples(stageii_data):
    residual_samples = []
    for observed_frame, simulated_frame in _iter_marker_frames(stageii_data):
        if observed_frame.shape != simulated_frame.shape:
            raise ValueError(
                "observed and simulated marker frames must match, got "
                f"{observed_frame.shape} != {simulated_frame.shape}"
            )
        if observed_frame.shape[0] == 0:
            continue
        residual = simulated_frame - observed_frame
        finite_mask = np.isfinite(residual).all(axis=1)
        if not finite_mask.any():
            continue
        residual_norm = np.linalg.norm(
            np.asarray(residual[finite_mask], dtype=np.float64),
            axis=1,
        )
        residual_samples.extend(float(value) for value in residual_norm.tolist())
    return residual_samples


def _frame_delta_l2_samples(array_like):
    array = np.asarray(array_like, dtype=np.float64)
    if array.shape[0] < 2:
        return []
    delta = np.diff(array, axis=0)
    delta = np.nan_to_num(delta, copy=False)
    delta = delta.reshape(delta.shape[0], -1)
    delta_norm = np.linalg.norm(delta, axis=1)
    finite_mask = np.isfinite(delta_norm)
    return [float(value) for value in delta_norm[finite_mask].tolist()]


def _temporal_accel_l2_samples(array_like):
    array = np.asarray(array_like, dtype=np.float64)
    if array.shape[0] < 3:
        return []
    accel = np.diff(array, n=2, axis=0)
    accel = np.nan_to_num(accel, copy=False)
    accel = accel.reshape(accel.shape[0], -1)
    accel_norm = np.linalg.norm(accel, axis=1)
    finite_mask = np.isfinite(accel_norm)
    return [float(value) for value in accel_norm[finite_mask].tolist()]


def _frame_delta_l2_series(array_like):
    array = np.asarray(array_like, dtype=np.float64)
    series = np.full(array.shape[0], np.nan, dtype=np.float64)
    if array.shape[0] < 2:
        return series
    delta = np.diff(array, axis=0)
    delta = np.nan_to_num(delta, copy=False)
    delta = delta.reshape(delta.shape[0], -1)
    delta_norm = np.linalg.norm(delta, axis=1)
    finite_mask = np.isfinite(delta_norm)
    series[1:][finite_mask] = delta_norm[finite_mask]
    return series


def _temporal_accel_l2_series(array_like):
    array = np.asarray(array_like, dtype=np.float64)
    series = np.full(array.shape[0], np.nan, dtype=np.float64)
    if array.shape[0] < 3:
        return series
    accel = np.diff(array, n=2, axis=0)
    accel = np.nan_to_num(accel, copy=False)
    accel = accel.reshape(accel.shape[0], -1)
    accel_norm = np.linalg.norm(accel, axis=1)
    finite_mask = np.isfinite(accel_norm)
    series[2:][finite_mask] = accel_norm[finite_mask]
    return series


def _pose_region_slices(surface_model_type, pose_dim):
    surface_model_type = str(surface_model_type or "").lower()
    if pose_dim >= 165:
        return {
            "body_pose": slice(3, 66),
            "left_hand_pose": slice(75, 120),
            "right_hand_pose": slice(120, 165),
        }
    if pose_dim >= 156:
        return {
            "body_pose": slice(3, 66),
            "left_hand_pose": slice(66, 111),
            "right_hand_pose": slice(111, 156),
        }
    if pose_dim >= 66 or surface_model_type in {"smpl", "smplh", "smplx"}:
        return {
            "body_pose": slice(3, min(66, pose_dim)),
            "left_hand_pose": None,
            "right_hand_pose": None,
        }
    return {
        "body_pose": None,
        "left_hand_pose": None,
        "right_hand_pose": None,
    }


def _pose_region_arrays(sample):
    pose_array = np.asarray(sample.poses, dtype=np.float64)
    if pose_array.ndim != 2:
        return {region_name: None for region_name in _POSE_REGION_NAMES}

    region_slices = _pose_region_slices(sample.surface_model_type, pose_array.shape[1])
    region_arrays = {}
    for region_name in ("body_pose", "left_hand_pose", "right_hand_pose"):
        region_slice = region_slices.get(region_name)
        if region_slice is None:
            region_arrays[region_name] = None
            continue
        region_array = pose_array[:, region_slice]
        region_arrays[region_name] = region_array if region_array.shape[1] > 0 else None

    left_hand = region_arrays["left_hand_pose"]
    right_hand = region_arrays["right_hand_pose"]
    if left_hand is None or right_hand is None:
        region_arrays["all_hands_pose"] = None
    else:
        region_arrays["all_hands_pose"] = np.concatenate((left_hand, right_hand), axis=1)
    return region_arrays


def _window_ranges(length, *, window_size):
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    return [(start, min(start + window_size, length)) for start in range(0, length, window_size)]


def _summarize_compared_temporal_window_metric(
    candidate_series,
    reference_series,
    *,
    window_size,
    top_k,
    positive_threshold,
    negative_threshold,
):
    candidate_series = np.asarray(candidate_series, dtype=np.float64)
    reference_series = np.asarray(reference_series, dtype=np.float64)
    if candidate_series.shape != reference_series.shape:
        raise ValueError("candidate and reference metric series must match in shape")

    windows = []
    for start, end in _window_ranges(candidate_series.shape[0], window_size=window_size):
        candidate_window = candidate_series[start:end]
        reference_window = reference_series[start:end]
        finite_mask = np.isfinite(candidate_window) & np.isfinite(reference_window)
        if not finite_mask.any():
            continue
        candidate_values = candidate_window[finite_mask]
        reference_values = reference_window[finite_mask]
        delta_values = candidate_values - reference_values
        frame_indices = np.nonzero(finite_mask)[0] + start
        windows.append(
            {
                "frame_start": int(frame_indices[0]),
                "frame_end": int(frame_indices[-1]),
                "frame_count": int(frame_indices.shape[0]),
                "candidate_mean": float(np.mean(candidate_values)),
                "reference_mean": float(np.mean(reference_values)),
                "delta_mean": float(np.mean(delta_values)),
                "candidate_max": float(np.max(candidate_values)),
                "reference_max": float(np.max(reference_values)),
                "delta_max": float(np.max(delta_values)),
                "delta_min": float(np.min(delta_values)),
            }
        )

    positive_peaks = sorted(
        [window for window in windows if window["delta_mean"] >= positive_threshold],
        key=lambda window: window["delta_mean"],
        reverse=True,
    )
    negative_peaks = sorted(
        [window for window in windows if window["delta_mean"] <= -negative_threshold],
        key=lambda window: window["delta_mean"],
    )
    return {
        "window_count": len(windows),
        "positive_peak_count": len(positive_peaks),
        "negative_peak_count": len(negative_peaks),
        "positive_peaks": positive_peaks[:top_k],
        "negative_peaks": negative_peaks[:top_k],
    }


def _sequence_chunk_ranges(total_frames, chunk_size, overlap):
    if total_frames <= 0:
        return []
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    overlap = int(overlap)
    if overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    step = max(chunk_size - overlap, 1)
    ranges = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size, total_frames)
        ranges.append((start, end))
        if end >= total_frames:
            break
        start += step
    return ranges


def _sequence_chunk_config(stageii_data):
    stageii_debug = stageii_data.get("stageii_debug_details", {})
    cfg = stageii_debug.get("cfg")
    if not isinstance(cfg, dict):
        return None
    runtime = cfg.get("runtime")
    if not isinstance(runtime, dict):
        return None
    chunk_size = runtime.get("sequence_chunk_size", 1)
    if chunk_size is None:
        chunk_size = 1
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("runtime.sequence_chunk_size must be > 0")
    chunk_overlap = runtime.get("sequence_chunk_overlap", 0)
    if chunk_overlap is None:
        chunk_overlap = 0
    chunk_overlap = int(chunk_overlap)
    if chunk_overlap < 0:
        raise ValueError("runtime.sequence_chunk_overlap must be >= 0")
    if chunk_size <= 1:
        return None
    return chunk_size, chunk_overlap


def _sequence_chunk_keep_starts(stageii_data, *, total_frames, chunk_size, overlap):
    stageii_debug = stageii_data.get("stageii_debug_details", {})
    keep_starts = stageii_debug.get("sequence_chunk_keep_starts")
    if keep_starts is None:
        return None
    if isinstance(keep_starts, np.ndarray):
        keep_starts = keep_starts.tolist()
    if not isinstance(keep_starts, (list, tuple)):
        raise ValueError("stageii_debug_details.sequence_chunk_keep_starts must be a list when present")
    chunk_ranges = _sequence_chunk_ranges(total_frames, chunk_size, overlap)
    if len(keep_starts) != len(chunk_ranges):
        raise ValueError("stageii_debug_details.sequence_chunk_keep_starts must align with chunk count")
    normalized_keep_starts = []
    for chunk_idx, ((row_start, row_end), keep_start) in enumerate(zip(chunk_ranges, keep_starts)):
        chunk_length = row_end - row_start
        keep_start = int(keep_start)
        max_keep_start = min(overlap, chunk_length)
        if keep_start < 0 or keep_start > max_keep_start:
            raise ValueError("stageii_debug_details.sequence_chunk_keep_starts contains an out-of-range value")
        if chunk_idx == 0 and keep_start != 0:
            raise ValueError("stageii_debug_details.sequence_chunk_keep_starts must start with 0 for the first chunk")
        normalized_keep_starts.append(keep_start)
    return normalized_keep_starts


def _chunk_seam_plans(total_frames, *, chunk_size, overlap, keep_starts=None):
    chunk_ranges = _sequence_chunk_ranges(total_frames, chunk_size, overlap)
    if keep_starts is None:
        keep_starts = [
            0 if chunk_idx == 0 else min(overlap, row_end - row_start)
            for chunk_idx, (row_start, row_end) in enumerate(chunk_ranges)
        ]
    elif len(keep_starts) != len(chunk_ranges):
        raise ValueError("keep_starts must align with chunk count")

    seam_plans = []
    output_length = 0
    for chunk_idx, ((row_start, row_end), keep_start) in enumerate(zip(chunk_ranges, keep_starts)):
        chunk_length = row_end - row_start
        overlap_count = 0 if chunk_idx == 0 else min(overlap, chunk_length)
        keep_start = int(keep_start)
        kept_length = max(chunk_length - keep_start, 0)
        if chunk_idx == 0:
            output_length += kept_length
            continue
        trim_count = max(overlap_count - keep_start, 0)
        output_length = max(output_length - trim_count, 0)
        if kept_length <= 0:
            continue
        seam_plans.append(
            {
                "chunk_index": chunk_idx,
                "row_start": row_start,
                "row_end": row_end,
                "keep_start": keep_start,
                "trim_count": trim_count,
                "seam_index": output_length,
            }
        )
        output_length += kept_length
    return seam_plans


def _chunk_seam_indices(total_frames, *, chunk_size, overlap, keep_starts=None):
    return [
        seam_plan["seam_index"]
        for seam_plan in _chunk_seam_plans(
            total_frames,
            chunk_size=chunk_size,
            overlap=overlap,
            keep_starts=keep_starts,
        )
    ]


def _frame_delta_l2_at(array, left_idx, right_idx):
    if left_idx < 0 or right_idx >= array.shape[0]:
        return None
    delta = np.nan_to_num(array[right_idx] - array[left_idx], copy=False)
    return float(np.linalg.norm(delta.reshape(-1)))


def _accel_l2_at(array, start_idx):
    if start_idx < 0 or start_idx + 2 >= array.shape[0]:
        return None
    accel = np.nan_to_num(array[start_idx + 2] - 2.0 * array[start_idx + 1] + array[start_idx], copy=False)
    return float(np.linalg.norm(accel.reshape(-1)))


def _local_transition_diagnostics_at(array, seam_index):
    return {
        "prev_frame_delta_l2": _frame_delta_l2_at(array, seam_index - 2, seam_index - 1),
        "seam_jump_l2": _frame_delta_l2_at(array, seam_index - 1, seam_index),
        "next_frame_delta_l2": _frame_delta_l2_at(array, seam_index, seam_index + 1),
        "pre_accel_l2": _accel_l2_at(array, seam_index - 2),
        "post_accel_l2": _accel_l2_at(array, seam_index - 1),
    }


def _chunk_seam_jump_l2_samples(array_like, *, chunk_size, overlap, keep_starts=None):
    array = np.asarray(array_like, dtype=np.float64)
    if array.shape[0] < 2:
        return []
    seam_samples = []
    for seam_index in _chunk_seam_indices(
        array.shape[0],
        chunk_size=chunk_size,
        overlap=overlap,
        keep_starts=keep_starts,
    ):
        if seam_index <= 0 or seam_index >= array.shape[0]:
            continue
        seam_delta = np.nan_to_num(array[seam_index] - array[seam_index - 1], copy=False)
        seam_samples.append(float(np.linalg.norm(seam_delta.reshape(-1))))
    return seam_samples


def _chunk_seam_local_diagnostics(array_like, *, chunk_size, overlap, keep_starts=None):
    array = np.asarray(array_like, dtype=np.float64)
    if array.shape[0] < 2:
        return []

    diagnostics = []
    for seam_plan in _chunk_seam_plans(
        array.shape[0],
        chunk_size=chunk_size,
        overlap=overlap,
        keep_starts=keep_starts,
    ):
        seam_index = seam_plan["seam_index"]
        diagnostics.append(
            {
                **seam_plan,
                **_local_transition_diagnostics_at(array, seam_index),
            }
        )
    return diagnostics


def summarize_stageii_chunk_seam_diagnostics(sample_path):
    baseline = normalize_stageii_sample(sample_path)
    stageii_data = _load_pickle_compat(sample_path)
    chunk_config = _sequence_chunk_config(stageii_data)
    if chunk_config is None:
        return None

    chunk_size, chunk_overlap = chunk_config
    chunk_keep_starts = _sequence_chunk_keep_starts(
        stageii_data,
        total_frames=baseline.trans.shape[0],
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_keep_starts": chunk_keep_starts,
        "transl": _chunk_seam_local_diagnostics(
            baseline.trans,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            keep_starts=chunk_keep_starts,
        ),
        "pose": _chunk_seam_local_diagnostics(
            baseline.poses,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            keep_starts=chunk_keep_starts,
        ),
    }


def compare_stageii_chunk_seam_diagnostics(reference_path, candidate_path):
    reference = normalize_stageii_sample(reference_path)
    candidate = normalize_stageii_sample(candidate_path)
    if reference.trans.shape[0] != candidate.trans.shape[0]:
        raise ValueError("reference and candidate stageii samples must have matching frame counts")

    candidate_diagnostics = summarize_stageii_chunk_seam_diagnostics(candidate_path)
    if candidate_diagnostics is None:
        return None

    def _compare_axis(rows, reference_array):
        reference_array = np.asarray(reference_array, dtype=np.float64)
        compared_rows = []
        for row in rows:
            seam_index = int(row["seam_index"])
            reference_metrics = _local_transition_diagnostics_at(reference_array, seam_index)
            delta_metrics = {}
            for key, candidate_value in row.items():
                if key in {"chunk_index", "row_start", "row_end", "keep_start", "trim_count", "seam_index"}:
                    continue
                reference_value = reference_metrics.get(key)
                if candidate_value is None or reference_value is None:
                    delta_metrics[key] = None
                else:
                    delta_metrics[key] = float(candidate_value) - float(reference_value)
            compared_rows.append(
                {
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
                }
            )
        return compared_rows

    return {
        "chunk_size": candidate_diagnostics["chunk_size"],
        "chunk_overlap": candidate_diagnostics["chunk_overlap"],
        "chunk_keep_starts": candidate_diagnostics["chunk_keep_starts"],
        "transl": _compare_axis(candidate_diagnostics["transl"], reference.trans),
        "pose": _compare_axis(candidate_diagnostics["pose"], reference.poses),
    }


_COMPARED_STAGEII_CHUNK_SEAM_FOCUS_KEYS = ("seam_jump_l2", "pre_accel_l2", "post_accel_l2")


def _summarize_compared_stageii_chunk_seam_rows(rows, *, top_k, positive_threshold, negative_threshold):
    summarized_rows = []
    for row in rows:
        delta = row["delta"]
        focus_values = {
            key: 0.0 if delta.get(key) is None else float(delta[key])
            for key in _COMPARED_STAGEII_CHUNK_SEAM_FOCUS_KEYS
        }
        peak_metric, peak_value = max(focus_values.items(), key=lambda item: item[1])
        trough_metric, trough_value = min(focus_values.items(), key=lambda item: item[1])
        summarized_rows.append(
            {
                **row,
                "delta_peak_metric": peak_metric,
                "delta_peak_value": peak_value,
                "delta_trough_metric": trough_metric,
                "delta_trough_value": trough_value,
            }
        )

    positive_peaks = sorted(
        [row for row in summarized_rows if row["delta_peak_value"] >= positive_threshold],
        key=lambda row: row["delta_peak_value"],
        reverse=True,
    )
    negative_peaks = sorted(
        [row for row in summarized_rows if row["delta_trough_value"] <= -negative_threshold],
        key=lambda row: row["delta_trough_value"],
    )
    return {
        "seam_count": len(rows),
        "positive_peak_count": len(positive_peaks),
        "negative_peak_count": len(negative_peaks),
        "positive_peaks": positive_peaks[:top_k],
        "negative_peaks": negative_peaks[:top_k],
    }


def summarize_compared_stageii_chunk_seam_diagnostics(
    comparison,
    *,
    top_k=5,
    positive_threshold=0.0,
    negative_threshold=0.0,
):
    if comparison is None:
        return None
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if positive_threshold < 0:
        raise ValueError("positive_threshold must be non-negative")
    if negative_threshold < 0:
        raise ValueError("negative_threshold must be non-negative")

    chunk_overlap = int(comparison["chunk_overlap"])
    seam_rows = comparison.get("pose") or comparison.get("transl") or []
    nondefault_keep_starts = [
        {
            "chunk_index": row["chunk_index"],
            "keep_start": row["keep_start"],
            "seam_index": row["seam_index"],
        }
        for row in seam_rows
        if int(row["keep_start"]) != chunk_overlap
    ]
    return {
        "chunk_size": comparison["chunk_size"],
        "chunk_overlap": chunk_overlap,
        "chunk_keep_starts": comparison["chunk_keep_starts"],
        "nondefault_keep_starts": nondefault_keep_starts,
        "transl": _summarize_compared_stageii_chunk_seam_rows(
            comparison["transl"],
            top_k=top_k,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        ),
        "pose": _summarize_compared_stageii_chunk_seam_rows(
            comparison["pose"],
            top_k=top_k,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        ),
    }


def summarize_compared_stageii_pose_window_hotspots(
    reference_path,
    candidate_path,
    *,
    window_size=REFERENCE_STAGEII_POSE_WINDOW_SIZE,
    top_k=5,
    positive_threshold=0.0,
    negative_threshold=0.0,
):
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if positive_threshold < 0:
        raise ValueError("positive_threshold must be non-negative")
    if negative_threshold < 0:
        raise ValueError("negative_threshold must be non-negative")

    reference = normalize_stageii_sample(reference_path)
    candidate = normalize_stageii_sample(candidate_path)
    if reference.poses.shape[0] != candidate.poses.shape[0]:
        raise ValueError("reference and candidate stageii samples must have matching frame counts")

    reference_regions = _pose_region_arrays(reference)
    candidate_regions = _pose_region_arrays(candidate)
    summary = {"window_size": int(window_size)}
    for region_name in _POSE_REGION_NAMES:
        reference_region = reference_regions.get(region_name)
        candidate_region = candidate_regions.get(region_name)
        frame_delta_metric_name = f"{region_name}_frame_delta_l2"
        jitter_metric_name = f"{region_name}_jitter_l2"
        if (
            reference_region is None
            or candidate_region is None
            or reference_region.shape != candidate_region.shape
        ):
            summary[frame_delta_metric_name] = None
            summary[jitter_metric_name] = None
            continue
        summary[frame_delta_metric_name] = _summarize_compared_temporal_window_metric(
            _frame_delta_l2_series(candidate_region),
            _frame_delta_l2_series(reference_region),
            window_size=window_size,
            top_k=top_k,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
        summary[jitter_metric_name] = _summarize_compared_temporal_window_metric(
            _temporal_accel_l2_series(candidate_region),
            _temporal_accel_l2_series(reference_region),
            window_size=window_size,
            top_k=top_k,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
    return summary


def _summarize_stageii_quality(sample_path, baseline):
    trans_frame_delta = _summarize_numeric_samples(
        _frame_delta_l2_samples(baseline.trans),
        include_samples=False,
    )
    pose_frame_delta = _summarize_numeric_samples(
        _frame_delta_l2_samples(baseline.poses),
        include_samples=False,
    )
    trans_jitter = _summarize_numeric_samples(
        _temporal_accel_l2_samples(baseline.trans),
        include_samples=False,
    )
    pose_jitter = _summarize_numeric_samples(
        _temporal_accel_l2_samples(baseline.poses),
        include_samples=False,
    )
    stageii_data = _load_pickle_compat(sample_path)
    quality = {
        "marker_residual_l2": _summarize_numeric_samples(
            _marker_residual_l2_samples(stageii_data),
            include_samples=False,
        ),
        "trans_frame_delta_l2": trans_frame_delta,
        "pose_frame_delta_l2": pose_frame_delta,
        "trans_jitter_l2": trans_jitter,
        "pose_jitter_l2": pose_jitter,
        **{metric_name: None for metric_name in _POSE_REGION_METRIC_NAMES},
        "chunk_seam_transl_jump_l2": None,
        "chunk_seam_pose_jump_l2": None,
        "chunk_seam_transl_jump_over_trans_frame_delta_ratio": None,
        "chunk_seam_pose_jump_over_pose_frame_delta_ratio": None,
    }
    for region_name, region_array in _pose_region_arrays(baseline).items():
        if region_array is None:
            continue
        quality[f"{region_name}_frame_delta_l2"] = _summarize_numeric_samples(
            _frame_delta_l2_samples(region_array),
            include_samples=False,
        )
        quality[f"{region_name}_jitter_l2"] = _summarize_numeric_samples(
            _temporal_accel_l2_samples(region_array),
            include_samples=False,
        )
    chunk_config = _sequence_chunk_config(stageii_data)
    if chunk_config is None:
        return quality

    chunk_size, chunk_overlap = chunk_config
    chunk_keep_starts = _sequence_chunk_keep_starts(
        stageii_data,
        total_frames=baseline.trans.shape[0],
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )
    quality["chunk_seam_transl_jump_l2"] = _summarize_numeric_samples(
        _chunk_seam_jump_l2_samples(
            baseline.trans,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            keep_starts=chunk_keep_starts,
        ),
        include_samples=False,
    )
    quality["chunk_seam_pose_jump_l2"] = _summarize_numeric_samples(
        _chunk_seam_jump_l2_samples(
            baseline.poses,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            keep_starts=chunk_keep_starts,
        ),
        include_samples=False,
    )
    quality["chunk_seam_transl_jump_over_trans_frame_delta_ratio"] = _summarize_metric_ratio(
        quality["chunk_seam_transl_jump_l2"],
        trans_frame_delta,
    )
    quality["chunk_seam_pose_jump_over_pose_frame_delta_ratio"] = _summarize_metric_ratio(
        quality["chunk_seam_pose_jump_l2"],
        pose_frame_delta,
    )
    return quality


def _summarize_mesh_compare(
    sample_path,
    *,
    reference_path,
    support_base_dir=None,
    chunk_size=None,
    chunk_overlap=None,
):
    from utils.mesh_compare import compare_mesh_sequences

    report = compare_mesh_sequences(
        reference_path,
        sample_path,
        support_base_dir=support_base_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    report["reference_path"] = str(reference_path)
    return report


def _normalize_reference_stageii_sample(reference_path):
    reference_path = Path(reference_path)
    if reference_path.suffix.lower() != ".pkl":
        return None
    return normalize_stageii_sample(reference_path)


def _summarize_reference_stageii_quality(reference_sample):
    if reference_sample is None:
        return None
    return _summarize_stageii_quality(reference_sample.sample_path, reference_sample)


def _summarize_reference_stageii_delta(candidate_quality, reference_quality):
    if reference_quality is None:
        return None

    delta = {}
    for metric_name, candidate_metric in candidate_quality.items():
        reference_metric = reference_quality.get(metric_name)
        if candidate_metric is None or reference_metric is None:
            delta[metric_name] = None
            continue

        metric_delta = {}
        for stat_name in ("mean", "p90", "max"):
            if stat_name in candidate_metric and stat_name in reference_metric:
                metric_delta[stat_name] = float(candidate_metric[stat_name]) - float(reference_metric[stat_name])
        delta[metric_name] = metric_delta or None

    return delta


def _summarize_reference_stageii_chunk_seam_hotspots(reference_path, candidate_path):
    reference_sample = _normalize_reference_stageii_sample(reference_path)
    if reference_sample is None:
        return None
    comparison = compare_stageii_chunk_seam_diagnostics(reference_path, candidate_path)
    return summarize_compared_stageii_chunk_seam_diagnostics(
        comparison,
        positive_threshold=0.05,
        negative_threshold=0.05,
    )


def _summarize_reference_stageii_pose_window_hotspots(reference_path, candidate_path):
    reference_sample = _normalize_reference_stageii_sample(reference_path)
    if reference_sample is None:
        return None
    return summarize_compared_stageii_pose_window_hotspots(
        reference_path,
        candidate_path,
        window_size=REFERENCE_STAGEII_POSE_WINDOW_SIZE,
        top_k=5,
        positive_threshold=0.0,
        negative_threshold=0.0,
    )


def _support_files_root(repo_root):
    return Path(repo_root) / "support_files"


def _support_model_assets(repo_root, *, suffix):
    return sorted(_support_files_root(repo_root).glob(f"**/model{suffix}"))


def _mesh_export_block_reason(repo_root):
    npz_models = _support_model_assets(repo_root, suffix=".npz")
    if npz_models:
        if _safe_find_spec("human_body_prior.body_model.body_model") is None:
            return (
                "save_smplx_verts.py mesh export found support_files model.npz assets but "
                "human_body_prior.body_model.body_model is unavailable in the current Python environment"
            )
        return None

    pkl_models = _support_model_assets(repo_root, suffix=".pkl")
    if pkl_models:
        if _safe_find_spec("smplx") is None:
            return (
                "save_smplx_verts.py mesh export found support_files model.pkl assets but "
                "smplx is unavailable in the current Python environment"
            )
        return None

    return "support_files does not include SMPL-X model.npz/model.pkl assets needed by save_smplx_verts.py mesh export"


def _preview_render_block_reason(repo_root):
    if _safe_find_spec("taichi") is None:
        return "render_video.py preview renderer requires taichi, which is unavailable in the current Python environment"
    if _safe_find_spec("cv2") is None:
        return "render_video.py preview renderer requires cv2, which is unavailable in the current Python environment"

    npz_models = _support_model_assets(repo_root, suffix=".npz")
    if npz_models:
        if _safe_find_spec("human_body_prior.body_model.body_model") is None:
            return (
                "render_video.py preview renderer found support_files model.npz assets but "
                "human_body_prior.body_model.body_model is unavailable in the current Python environment"
            )
        return None

    pkl_models = _support_model_assets(repo_root, suffix=".pkl")
    if pkl_models:
        if _safe_find_spec("smplx") is None:
            return (
                "render_video.py preview renderer found support_files model.pkl assets but "
                "smplx is unavailable in the current Python environment"
            )
        return None

    return "support_files does not include SMPL-X model.npz/model.pkl assets needed by render_video.py preview renderer"


def _preview_render_model_path(repo_root, *, gender):
    support_root = _support_files_root(repo_root) / "smplx"
    candidate_genders = [gender, "neutral"]
    for candidate_gender in candidate_genders:
        if not candidate_gender or candidate_gender == "unknown":
            continue
        npz_path = support_root / candidate_gender / "model.npz"
        if npz_path.exists():
            return npz_path
        pkl_path = support_root / candidate_gender / "model.pkl"
        if pkl_path.exists():
            return pkl_path

    npz_assets = _support_model_assets(repo_root, suffix=".npz")
    if npz_assets:
        return npz_assets[0]

    pkl_assets = _support_model_assets(repo_root, suffix=".pkl")
    if pkl_assets:
        return pkl_assets[0]

    return None


def _benchmark_preview_vertex_decode(sample_path, baseline, *, repo_root, warmup_runs, measured_runs):
    model_path = _preview_render_model_path(repo_root, gender=baseline.gender)
    if model_path is None:
        return None

    try:
        import render_video
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        model = render_video.load_render_model(model_path)
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        return None

    for _ in range(warmup_runs):
        render_video.load_vertices(sample_path, model)

    latency_samples = []
    for _ in range(measured_runs):
        started_at = perf_counter()
        render_video.load_vertices(sample_path, model)
        latency_samples.append((perf_counter() - started_at) * 1000.0)

    return _summarize_latency_samples(latency_samples)


def _benchmark_mesh_export(sample_path, baseline, *, repo_root, warmup_runs, measured_runs):
    model_path = _preview_render_model_path(repo_root, gender=baseline.gender)
    if model_path is None:
        return None

    try:
        import render_video
        import save_smplx_verts
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        model = render_video.load_render_model(model_path)
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        return None

    def _export_once():
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            obj_out = temp_dir / "benchmark_stageii.obj"
            pc2_out = temp_dir / "benchmark_stageii.pc2"
            result = save_smplx_verts.export_stageii_meshes(
                input_pkl=sample_path,
                model_path=model_path,
                model=model,
                obj_out=obj_out,
                pc2_out=pc2_out,
            )
            if not obj_out.exists() or not pc2_out.exists():
                raise FileNotFoundError(f"mesh export did not create expected outputs: {result}")

    try:
        for _ in range(warmup_runs):
            _export_once()
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        return None

    latency_samples = []
    for _ in range(measured_runs):
        started_at = perf_counter()
        _export_once()
        latency_samples.append((perf_counter() - started_at) * 1000.0)

    return _summarize_latency_samples(latency_samples)


def _benchmark_mp4_render(sample_path, baseline, *, repo_root, warmup_runs, measured_runs):
    model_path = _preview_render_model_path(repo_root, gender=baseline.gender)
    if model_path is None:
        return None

    try:
        import render_video
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        model = render_video.load_render_model(model_path)
        vertices = render_video.load_vertices(sample_path, model)
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        return None

    def _render_once():
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            video_path = temp_dir / "benchmark_stageii.mp4"
            suppressed = io.StringIO()
            with redirect_stdout(suppressed), redirect_stderr(suppressed):
                render_video.render_vertices_to_video(
                    vertices=vertices,
                    faces=model.faces,
                    output_path=video_path,
                    show_progress=False,
                    **PREVIEW_RENDER_BENCHMARK_WORKLOAD,
                )
            if not video_path.exists() or video_path.stat().st_size == 0:
                raise FileNotFoundError("preview render did not create a non-empty mp4 output")

    try:
        for _ in range(warmup_runs):
            _render_once()
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        return None

    latency_samples = []
    for _ in range(measured_runs):
        started_at = perf_counter()
        _render_once()
        latency_samples.append((perf_counter() - started_at) * 1000.0)

    return _summarize_latency_samples(latency_samples)


def _benchmark_artifact_bundle_export(sample_path, baseline, *, repo_root, warmup_runs, measured_runs):
    model_path = _preview_render_model_path(repo_root, gender=baseline.gender)
    if model_path is None:
        return None

    try:
        import export_stageii_artifacts
        import render_video
    except (ImportError, ModuleNotFoundError):
        return None

    try:
        model = render_video.load_render_model(model_path)
        vertices = render_video.load_vertices(sample_path, model)
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        return None

    def _export_once():
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            suppressed = io.StringIO()
            with redirect_stdout(suppressed), redirect_stderr(suppressed):
                result = export_stageii_artifacts.export_stageii_artifacts(
                    input_pkl=sample_path,
                    model_path=model_path,
                    model=model,
                    vertices=vertices,
                    output_dir=temp_dir,
                    show_progress=False,
                    **PREVIEW_RENDER_BENCHMARK_WORKLOAD,
                )
            for key in ("obj_path", "pc2_path", "video_path"):
                artifact_path = Path(result[key])
                if not artifact_path.exists() or artifact_path.stat().st_size == 0:
                    raise FileNotFoundError(f"artifact bundle export did not create expected output: {result}")

    try:
        for _ in range(warmup_runs):
            _export_once()
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        return None

    latency_samples = []
    for _ in range(measured_runs):
        started_at = perf_counter()
        _export_once()
        latency_samples.append((perf_counter() - started_at) * 1000.0)

    return _summarize_latency_samples(latency_samples)


def _mosh_head_loader_block_reason():
    try:
        from moshpp.mosh_head import run_moshpp_once
    except (ImportError, ModuleNotFoundError) as exc:
        return f"moshpp.mosh_head.run_moshpp_once import failed: {exc}"
    except Exception as exc:
        return f"moshpp.mosh_head.run_moshpp_once import raised {type(exc).__name__}: {exc}"

    if not callable(run_moshpp_once):
        return "moshpp.mosh_head.run_moshpp_once is unavailable after import"
    return None


def _blocked_stages(repo_root):
    blocked = []

    mosh_head_loader_reason = _mosh_head_loader_block_reason()
    if mosh_head_loader_reason is not None:
        blocked.append(
            {
                "stage": "mosh_head_loader",
                "reason": mosh_head_loader_reason,
            }
        )
    mesh_export_reason = _mesh_export_block_reason(repo_root)
    if mesh_export_reason is not None:
        blocked.append(
            {
                "stage": "mesh_export",
                "reason": mesh_export_reason,
            }
        )

    preview_render_reason = _preview_render_block_reason(repo_root)
    if preview_render_reason is not None:
        blocked.append(
            {
                "stage": "mp4_render",
                "reason": preview_render_reason,
            }
        )

    return blocked


def _public_stageii_sample_path(repo_root):
    return Path(repo_root) / "support_data" / "tests" / "mosh_stageii.pkl"


def _is_public_stageii_sample(sample_path, repo_root):
    try:
        return Path(sample_path).resolve() == _public_stageii_sample_path(repo_root).resolve()
    except FileNotFoundError:
        return False


def run_public_stageii_benchmark(
    sample_path,
    *,
    warmup_runs=1,
    measured_runs=5,
    mesh_reference_path=None,
    mesh_support_base_dir=None,
    mesh_chunk_size=None,
    mesh_chunk_overlap=None,
    lean_benchmark=False,
    stageii_reference_path=None,
):
    sample_path = Path(sample_path)
    _validate_mesh_reference_path(sample_path=sample_path, mesh_reference_path=mesh_reference_path)
    _validate_stageii_reference_path(sample_path=sample_path, stageii_reference_path=stageii_reference_path)
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if measured_runs <= 0:
        raise ValueError("measured_runs must be > 0")

    baseline = normalize_stageii_sample(sample_path)
    for _ in range(warmup_runs):
        normalize_stageii_sample(sample_path)

    latencies_ms = []
    repeatability_max = 0.0
    for _ in range(measured_runs):
        started_at = perf_counter()
        current = normalize_stageii_sample(sample_path)
        latencies_ms.append((perf_counter() - started_at) * 1000.0)

        for key, baseline_array in _numeric_arrays(baseline).items():
            repeatability_max = max(repeatability_max, _max_abs_diff(baseline_array, _numeric_arrays(current)[key]))

    repo_root = Path(__file__).resolve().parents[1]
    latency_summary = _summarize_latency_samples(latencies_ms)
    preview_vertex_decode_summary = None
    mesh_export_summary = None
    mp4_render_summary = None
    artifact_bundle_export_summary = None
    if not lean_benchmark:
        preview_vertex_decode_summary = _benchmark_preview_vertex_decode(
            sample_path,
            baseline,
            repo_root=repo_root,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        mesh_export_summary = _benchmark_mesh_export(
            sample_path,
            baseline,
            repo_root=repo_root,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        mp4_render_summary = _benchmark_mp4_render(
            sample_path,
            baseline,
            repo_root=repo_root,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        artifact_bundle_export_summary = _benchmark_artifact_bundle_export(
            sample_path,
            baseline,
            repo_root=repo_root,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
    candidate_quality = dict(_summarize_stageii_quality(sample_path, baseline))
    quality_summary = dict(candidate_quality)
    quality_summary["reference_stageii_quality"] = None
    quality_summary["reference_stageii_delta"] = None
    quality_summary["reference_stageii_chunk_seam_hotspots"] = None
    quality_summary["reference_stageii_pose_window_hotspots"] = None
    quality_summary["mesh_compare"] = None
    reference_stageii_elapsed_s = None
    reference_stageii_elapsed_delta_s = None
    reference_stageii_path = stageii_reference_path or mesh_reference_path
    if reference_stageii_path is not None:
        reference_sample = _normalize_reference_stageii_sample(reference_stageii_path)
        reference_quality = _summarize_reference_stageii_quality(reference_sample)
        quality_summary["reference_stageii_quality"] = reference_quality
        quality_summary["reference_stageii_delta"] = _summarize_reference_stageii_delta(
            candidate_quality,
            reference_quality,
        )
        quality_summary["reference_stageii_chunk_seam_hotspots"] = _summarize_reference_stageii_chunk_seam_hotspots(
            reference_stageii_path,
            sample_path,
        )
        quality_summary["reference_stageii_pose_window_hotspots"] = _summarize_reference_stageii_pose_window_hotspots(
            reference_stageii_path,
            sample_path,
        )
        if reference_sample is not None:
            reference_stageii_elapsed_s = reference_sample.stageii_elapsed_s
            if baseline.stageii_elapsed_s is not None and reference_stageii_elapsed_s is not None:
                reference_stageii_elapsed_delta_s = baseline.stageii_elapsed_s - reference_stageii_elapsed_s
    if mesh_reference_path is not None:
        quality_summary["mesh_compare"] = _summarize_mesh_compare(
            sample_path,
            reference_path=mesh_reference_path,
            support_base_dir=mesh_support_base_dir,
            chunk_size=mesh_chunk_size,
            chunk_overlap=mesh_chunk_overlap,
        )

    return {
        "sample": {
            "path": str(sample_path),
            "sha256": _sha256_file(sample_path),
            "bytes": sample_path.stat().st_size,
            "format": baseline.sample_format,
            "surface_model_type": baseline.surface_model_type,
            "gender": baseline.gender,
        },
        "workload": {
            "frames": int(baseline.poses.shape[0]),
            "pose_dim": int(baseline.poses.shape[1]),
            "marker_count": int(baseline.markers_obs.shape[1]),
            "latent_marker_count": int(baseline.markers_latent.shape[0]),
            "mocap_frame_rate": baseline.mocap_frame_rate,
            "mocap_time_length": baseline.mocap_time_length,
            "preview_render_workload": copy.deepcopy(PREVIEW_RENDER_BENCHMARK_WORKLOAD),
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
        },
        "speed": {
            "latency_ms": latency_summary,
            "throughput_ops_s": 1000.0 / latency_summary["mean"] if latency_summary["mean"] > 0 else 0.0,
            "stageii_elapsed_s": baseline.stageii_elapsed_s,
            "reference_stageii_elapsed_s": reference_stageii_elapsed_s,
            "reference_stageii_elapsed_delta_s": reference_stageii_elapsed_delta_s,
            "preview_vertex_decode_ms": preview_vertex_decode_summary,
            "mesh_export_ms": mesh_export_summary,
            "mp4_render_ms": mp4_render_summary,
            "artifact_bundle_export_ms": artifact_bundle_export_summary,
        },
        "error": {
            "repeatability": {
                "max_abs_diff": repeatability_max,
            },
            "all_finite": _all_finite(baseline),
            "markers_obs_nan_count": int((~np.isfinite(baseline.markers_obs)).sum()),
        },
        "quality": quality_summary,
        "artifact": {
            "public_sample_present": _is_public_stageii_sample(sample_path, repo_root),
            "report_path": None,
            "blocked_stages": _blocked_stages(repo_root),
        },
        "engineering": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def write_benchmark_report(report, output_path):
    output_path = Path(output_path)
    payload = copy.deepcopy(report)
    payload.setdefault("artifact", {})["report_path"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def default_benchmark_output_path(sample_path):
    sample_path = Path(sample_path)
    if sample_path.name.endswith("_stageii.pkl"):
        benchmark_name = f"{sample_path.name[: -len('_stageii.pkl')]}_benchmark.json"
    else:
        benchmark_name = f"{sample_path.stem}_benchmark.json"
    return sample_path.with_name(benchmark_name)
