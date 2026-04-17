import copy
import hashlib
import importlib.util
import json
import platform
import pickle
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np


@dataclass
class NormalizedStageIISample:
    sample_path: Path
    sample_format: str
    gender: str
    surface_model_type: str
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


def _max_abs_diff(lhs, rhs):
    lhs = np.nan_to_num(np.asarray(lhs, dtype=np.float64), copy=False)
    rhs = np.nan_to_num(np.asarray(rhs, dtype=np.float64), copy=False)
    return float(np.max(np.abs(lhs - rhs)))


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summarize_latency_samples(latency_samples):
    latency_mean = statistics.mean(latency_samples)
    latency_stdev = statistics.stdev(latency_samples) if len(latency_samples) > 1 else 0.0
    return {
        "count": len(latency_samples),
        "samples": [float(latency_ms) for latency_ms in latency_samples],
        "mean": latency_mean,
        "stdev": latency_stdev,
        "min": min(latency_samples),
        "max": max(latency_samples),
        "p50": _percentile(latency_samples, 50),
        "p90": _percentile(latency_samples, 90),
        "p99": _percentile(latency_samples, 99),
    }


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


def _blocked_stages(repo_root):
    blocked = []

    if _safe_find_spec("body_visualizer.mesh") is None:
        blocked.append(
            {
                "stage": "mosh_head_loader",
                "reason": "body_visualizer.mesh is unavailable, so direct MoSh loader imports fail in this environment",
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


def run_public_stageii_benchmark(sample_path, *, warmup_runs=1, measured_runs=5):
    sample_path = Path(sample_path)
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
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
        },
        "speed": {
            "latency_ms": latency_summary,
            "throughput_ops_s": 1000.0 / latency_summary["mean"] if latency_summary["mean"] > 0 else 0.0,
            "preview_vertex_decode_ms": preview_vertex_decode_summary,
            "mesh_export_ms": mesh_export_summary,
        },
        "error": {
            "repeatability": {
                "max_abs_diff": repeatability_max,
            },
            "all_finite": _all_finite(baseline),
            "markers_obs_nan_count": int((~np.isfinite(baseline.markers_obs)).sum()),
        },
        "artifact": {
            "public_sample_present": True,
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
