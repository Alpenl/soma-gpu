import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_stageii_public
import utils.stageii_benchmark as stageii_benchmark
from utils.stageii_benchmark import (
    normalize_stageii_sample,
    run_public_stageii_benchmark,
    write_benchmark_report,
)


def test_normalize_stageii_sample_reads_legacy_support_sample():
    result = normalize_stageii_sample(ROOT / "support_data/tests/mosh_stageii.pkl")

    assert result.sample_format == "legacy_stageii_pkl"
    assert result.trans.shape[0] == result.poses.shape[0] == result.markers_obs.shape[0]
    assert result.poses.shape[1] > 100
    assert result.markers_obs.shape[2] == 3
    assert len(result.latent_labels) == result.markers_latent.shape[0]
    assert len(result.marker_labels) == result.markers_obs.shape[1]
    assert result.mocap_frame_rate > 0


def test_run_public_stageii_benchmark_reports_repeatable_summary(tmp_path, monkeypatch):
    perf_counter_points = iter(
        [
            0.000,
            0.005,
            1.000,
            1.020,
            2.000,
            2.050,
            3.000,
            3.100,
            4.000,
            4.200,
        ]
    )
    monkeypatch.setattr(stageii_benchmark, "perf_counter", lambda: next(perf_counter_points))
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=5,
    )

    assert report["sample"]["format"] == "legacy_stageii_pkl"
    assert report["workload"]["frames"] > 0
    assert report["speed"]["latency_ms"]["count"] == 5
    assert report["speed"]["latency_ms"]["samples"] == pytest.approx([5.0, 20.0, 50.0, 100.0, 200.0])
    assert report["speed"]["throughput_ops_s"] > 0
    assert report["speed"]["latency_ms"]["p50"] == pytest.approx(50.0)
    assert report["speed"]["latency_ms"]["p90"] == pytest.approx(160.0)
    assert report["speed"]["latency_ms"]["p99"] == pytest.approx(196.0)
    assert report["speed"]["stageii_elapsed_s"] is None
    assert report["speed"]["reference_stageii_elapsed_s"] is None
    assert report["speed"]["reference_stageii_elapsed_delta_s"] is None
    assert report["speed"]["preview_vertex_decode_ms"] is None
    assert report["speed"]["mesh_export_ms"] is None
    assert report["speed"]["mp4_render_ms"] is None
    assert report["speed"]["artifact_bundle_export_ms"] is None
    assert report["quality"]["chunk_seam_transl_jump_over_trans_frame_delta_ratio"] is None
    assert report["quality"]["chunk_seam_pose_jump_over_pose_frame_delta_ratio"] is None
    assert report["workload"]["preview_render_workload"] == stageii_benchmark.PREVIEW_RENDER_BENCHMARK_WORKLOAD
    assert report["error"]["repeatability"]["max_abs_diff"] == 0.0
    assert report["error"]["all_finite"] is True

    output_path = tmp_path / "report.json"
    write_benchmark_report(report, output_path)

    loaded = json.loads(output_path.read_text())
    assert loaded["sample"]["sha256"] == report["sample"]["sha256"]
    assert loaded["artifact"]["report_path"].endswith("report.json")
    assert loaded["speed"]["latency_ms"]["samples"] == pytest.approx([5.0, 20.0, 50.0, 100.0, 200.0])


def test_run_public_stageii_benchmark_marks_non_public_input_as_not_shipped_sample(
    tmp_path, monkeypatch
):
    sample_path = tmp_path / "synthetic_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.zeros((2, 3), dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.zeros((2, 3), dtype=np.float32),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        }
                    },
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 2,
                    "markers_obs": np.zeros((2, 1, 3), dtype=np.float32),
                    "markers_sim": np.zeros((2, 1, 3), dtype=np.float32),
                    "labels_obs": [["A"]] * 2,
                },
            }
        )
    )

    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        sample_path,
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["artifact"]["public_sample_present"] is False


def test_run_public_stageii_benchmark_reports_stageii_elapsed_from_stageii_debug_details(
    tmp_path, monkeypatch
):
    sample_path = tmp_path / "synthetic_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.zeros((2, 3), dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.zeros((2, 3), dtype=np.float32),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        }
                    },
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 2,
                    "stageii_elapsed_time": 12.34,
                    "markers_obs": np.zeros((2, 1, 3), dtype=np.float32),
                    "markers_sim": np.zeros((2, 1, 3), dtype=np.float32),
                    "labels_obs": [["A"]] * 2,
                },
            }
        )
    )

    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        sample_path,
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["speed"]["stageii_elapsed_s"] == pytest.approx(12.34)


def test_run_public_stageii_benchmark_includes_preview_vertex_decode_metric_when_available(monkeypatch):
    preview_metric = {
        "count": 2,
        "samples": [10.0, 14.0],
        "mean": 12.0,
        "stdev": pytest.approx(2.8284271247461903),
        "min": 10.0,
        "max": 14.0,
        "p50": 12.0,
        "p90": 13.6,
        "p99": 13.96,
    }
    monkeypatch.setattr(
        stageii_benchmark,
        "_benchmark_preview_vertex_decode",
        lambda *args, **kwargs: preview_metric,
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["speed"]["preview_vertex_decode_ms"] == preview_metric


def test_run_public_stageii_benchmark_includes_mesh_export_metric_when_available(monkeypatch):
    mesh_export_metric = {
        "count": 2,
        "samples": [20.0, 28.0],
        "mean": 24.0,
        "stdev": pytest.approx(5.656854249492381),
        "min": 20.0,
        "max": 28.0,
        "p50": 24.0,
        "p90": 27.2,
        "p99": 27.92,
    }
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        stageii_benchmark,
        "_benchmark_mesh_export",
        lambda *args, **kwargs: mesh_export_metric,
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["speed"]["mesh_export_ms"] == mesh_export_metric


def test_run_public_stageii_benchmark_includes_mp4_render_metric_when_available(monkeypatch):
    mp4_render_metric = {
        "count": 2,
        "samples": [30.0, 42.0],
        "mean": 36.0,
        "stdev": pytest.approx(8.48528137423857),
        "min": 30.0,
        "max": 42.0,
        "p50": 36.0,
        "p90": 40.8,
        "p99": 41.88,
    }
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        stageii_benchmark,
        "_benchmark_mp4_render",
        lambda *args, **kwargs: mp4_render_metric,
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["speed"]["mp4_render_ms"] == mp4_render_metric
    assert report["workload"]["preview_render_workload"] == stageii_benchmark.PREVIEW_RENDER_BENCHMARK_WORKLOAD


def test_run_public_stageii_benchmark_includes_artifact_bundle_export_metric_when_available(monkeypatch):
    artifact_bundle_metric = {
        "count": 2,
        "samples": [40.0, 44.0],
        "mean": 42.0,
        "stdev": pytest.approx(2.8284271247461903),
        "min": 40.0,
        "max": 44.0,
        "p50": 42.0,
        "p90": 43.6,
        "p99": 43.96,
    }
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        stageii_benchmark,
        "_benchmark_artifact_bundle_export",
        lambda *args, **kwargs: artifact_bundle_metric,
        raising=False,
    )

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["speed"]["artifact_bundle_export_ms"] == artifact_bundle_metric


def test_run_public_stageii_benchmark_lean_mode_skips_optional_speed_probes(monkeypatch):
    calls = {
        "preview": 0,
        "mesh_export": 0,
        "mp4_render": 0,
        "artifact_bundle": 0,
    }

    def _unexpected_preview(*args, **kwargs):
        calls["preview"] += 1
        return {"mean": 1.0}

    def _unexpected_mesh_export(*args, **kwargs):
        calls["mesh_export"] += 1
        return {"mean": 2.0}

    def _unexpected_mp4_render(*args, **kwargs):
        calls["mp4_render"] += 1
        return {"mean": 3.0}

    def _unexpected_artifact_bundle(*args, **kwargs):
        calls["artifact_bundle"] += 1
        return {"mean": 4.0}

    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", _unexpected_preview)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", _unexpected_mesh_export)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", _unexpected_mp4_render)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", _unexpected_artifact_bundle)

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=1,
        lean_benchmark=True,
    )

    assert calls == {
        "preview": 0,
        "mesh_export": 0,
        "mp4_render": 0,
        "artifact_bundle": 0,
    }
    assert report["speed"]["preview_vertex_decode_ms"] is None
    assert report["speed"]["mesh_export_ms"] is None
    assert report["speed"]["mp4_render_ms"] is None
    assert report["speed"]["artifact_bundle_export_ms"] is None
    assert report["speed"]["latency_ms"]["count"] == 1


def test_run_public_stageii_benchmark_includes_quality_summary_when_available(monkeypatch):
    quality_summary = {
        "marker_residual_l2": {"count": 3, "mean": 1.0},
        "trans_jitter_l2": {"count": 2, "mean": 0.5},
        "pose_jitter_l2": {"count": 2, "mean": 0.25},
        "chunk_seam_transl_jump_l2": None,
        "chunk_seam_pose_jump_l2": None,
    }
    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_stageii_quality",
        lambda *args, **kwargs: quality_summary,
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["quality"] == {
        **quality_summary,
        "reference_stageii_quality": None,
        "reference_stageii_delta": None,
        "reference_stageii_chunk_seam_hotspots": None,
        "reference_stageii_pose_window_hotspots": None,
        "mesh_compare": None,
    }


def test_run_public_stageii_benchmark_includes_optional_mesh_compare_summary(monkeypatch):
    mesh_compare_summary = {
        "reference_path": "/tmp/baseline_stageii.pkl",
        "reference": {"mesh_accel_l2": {"mean": 1.0}},
        "candidate": {"mesh_accel_l2": {"mean": 0.5}},
        "frame_delta_l2": {"mean": 0.25},
    }
    captured = {}

    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_stageii_quality",
        lambda *args, **kwargs: {
            "marker_residual_l2": {"count": 3, "mean": 1.0},
            "trans_jitter_l2": {"count": 2, "mean": 0.5},
            "pose_jitter_l2": {"count": 2, "mean": 0.25},
            "chunk_seam_transl_jump_l2": None,
            "chunk_seam_pose_jump_l2": None,
        },
        raising=False,
    )
    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_mesh_compare",
        lambda sample_path, *, reference_path, support_base_dir, chunk_size, chunk_overlap: (
            captured.update(
                {
                    "sample_path": sample_path,
                    "reference_path": reference_path,
                    "support_base_dir": support_base_dir,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
            )
            or mesh_compare_summary
        ),
        raising=False,
    )
    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_reference_stageii_quality",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        stageii_benchmark,
        "_normalize_reference_stageii_sample",
        lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=1,
        mesh_reference_path="/tmp/baseline_stageii.pkl",
        mesh_support_base_dir="/tmp/support_files",
        mesh_chunk_size=32,
        mesh_chunk_overlap=4,
    )

    assert report["quality"]["mesh_compare"] == mesh_compare_summary
    assert report["quality"]["reference_stageii_quality"] is None
    assert report["quality"]["reference_stageii_delta"] is None
    assert captured == {
        "sample_path": ROOT / "support_data/tests/mosh_stageii.pkl",
        "reference_path": "/tmp/baseline_stageii.pkl",
        "support_base_dir": "/tmp/support_files",
        "chunk_size": 32,
        "chunk_overlap": 4,
    }


def _write_synthetic_stageii_sample(sample_path, *, residual_offset, stageii_elapsed_time=None):
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        },
                        "runtime": {
                            "sequence_chunk_size": 3,
                            "sequence_chunk_overlap": 1,
                        },
                    },
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 3,
                    "stageii_elapsed_time": stageii_elapsed_time,
                    "markers_obs": np.zeros((3, 1, 3), dtype=np.float32),
                    "markers_sim": np.tile(
                        np.asarray([[[residual_offset, 0.0, 0.0]]], dtype=np.float32),
                        (3, 1, 1),
                    ),
                    "labels_obs": [["A"]] * 3,
                },
            }
        )
    )


def test_run_public_stageii_benchmark_includes_chunk_seam_to_frame_delta_ratios(
    tmp_path, monkeypatch
):
    sample_path = tmp_path / "chunked_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.asarray([[0.0], [2.0], [4.0], [10.0], [12.0]], dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [12.0, 0.0, 0.0],
                        [13.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        },
                        "runtime": {
                            "sequence_chunk_size": 3,
                            "sequence_chunk_overlap": 1,
                        },
                    },
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 5,
                    "markers_obs": np.zeros((5, 1, 3), dtype=np.float32),
                    "markers_sim": np.zeros((5, 1, 3), dtype=np.float32),
                    "labels_obs": [["A"]] * 5,
                },
            }
        )
    )

    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        sample_path,
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["quality"]["chunk_seam_transl_jump_over_trans_frame_delta_ratio"] == {
        "mean": pytest.approx(10.0 / 3.25),
        "p90": pytest.approx(10.0 / 7.3),
        "max": pytest.approx(1.0),
    }
    assert report["quality"]["chunk_seam_pose_jump_over_pose_frame_delta_ratio"] == {
        "mean": pytest.approx(2.0),
        "p90": pytest.approx(1.25),
        "max": pytest.approx(1.0),
    }


def test_run_public_stageii_benchmark_includes_reference_stageii_quality_for_stageii_mesh_reference(
    tmp_path, monkeypatch
):
    candidate_path = tmp_path / "candidate_stageii.pkl"
    reference_path = tmp_path / "baseline_stageii.pkl"
    _write_synthetic_stageii_sample(
        candidate_path,
        residual_offset=1.0,
        stageii_elapsed_time=12.5,
    )
    _write_synthetic_stageii_sample(
        reference_path,
        residual_offset=2.0,
        stageii_elapsed_time=10.0,
    )

    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_mesh_compare",
        lambda *args, **kwargs: {"frame_delta_l2": {"mean": 0.25}},
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        candidate_path,
        warmup_runs=0,
        measured_runs=1,
        mesh_reference_path=reference_path,
        mesh_support_base_dir="/tmp/support_files",
    )

    reference_quality = report["quality"]["reference_stageii_quality"]
    assert reference_quality is not None
    assert reference_quality["marker_residual_l2"]["mean"] == pytest.approx(2.0)
    assert reference_quality["trans_frame_delta_l2"]["mean"] == pytest.approx(1.5)
    assert reference_quality["pose_frame_delta_l2"]["mean"] == pytest.approx(1.0)
    assert reference_quality["trans_jitter_l2"]["mean"] == pytest.approx(1.0)
    assert report["speed"]["reference_stageii_elapsed_s"] == pytest.approx(10.0)
    assert report["speed"]["reference_stageii_elapsed_delta_s"] == pytest.approx(2.5)
    assert report["quality"]["reference_stageii_delta"] == {
        "marker_residual_l2": {
            "mean": pytest.approx(-1.0),
            "p90": pytest.approx(-1.0),
            "max": pytest.approx(-1.0),
        },
        "trans_frame_delta_l2": {
            "mean": pytest.approx(0.0),
            "p90": pytest.approx(0.0),
            "max": pytest.approx(0.0),
        },
        "pose_frame_delta_l2": {
            "mean": pytest.approx(0.0),
            "p90": pytest.approx(0.0),
            "max": pytest.approx(0.0),
        },
        "trans_jitter_l2": {
            "mean": pytest.approx(0.0),
            "p90": pytest.approx(0.0),
            "max": pytest.approx(0.0),
        },
        "pose_jitter_l2": {
            "mean": pytest.approx(0.0),
            "p90": pytest.approx(0.0),
            "max": pytest.approx(0.0),
        },
        "chunk_seam_transl_jump_over_trans_frame_delta_ratio": None,
        "chunk_seam_pose_jump_over_pose_frame_delta_ratio": None,
        "chunk_seam_transl_jump_l2": None,
        "chunk_seam_pose_jump_l2": None,
        "body_pose_frame_delta_l2": None,
        "body_pose_jitter_l2": None,
        "left_hand_pose_frame_delta_l2": None,
        "left_hand_pose_jitter_l2": None,
        "right_hand_pose_frame_delta_l2": None,
        "right_hand_pose_jitter_l2": None,
        "all_hands_pose_frame_delta_l2": None,
        "all_hands_pose_jitter_l2": None,
    }
    assert report["quality"]["mesh_compare"] == {"frame_delta_l2": {"mean": 0.25}}


def test_run_public_stageii_benchmark_includes_reference_stageii_chunk_seam_hotspots(
    tmp_path, monkeypatch
):
    def _write_stageii(sample_path, values):
        sample_path.write_bytes(
            pickle.dumps(
                {
                    "fullpose": np.asarray([[value] for value in values], dtype=np.float32),
                    "betas": np.zeros(10, dtype=np.float32),
                    "trans": np.asarray([[value, 0.0, 0.0] for value in values], dtype=np.float32),
                    "markers_latent": np.zeros((1, 3), dtype=np.float32),
                    "latent_labels": ["A"],
                    "stageii_debug_details": {
                        "cfg": {
                            "surface_model": {
                                "type": "smplx",
                                "gender": "male",
                            },
                            "runtime": {
                                "sequence_chunk_size": 4,
                                "sequence_chunk_overlap": 2,
                            },
                        },
                        "sequence_chunk_keep_starts": [0, 1],
                        "mocap_frame_rate": 120.0,
                        "mocap_time_length": len(values),
                        "markers_obs": np.zeros((len(values), 1, 3), dtype=np.float32),
                        "markers_sim": np.zeros((len(values), 1, 3), dtype=np.float32),
                        "labels_obs": [["A"]] * len(values),
                    },
                }
            )
        )

    candidate_path = tmp_path / "candidate_stageii.pkl"
    reference_path = tmp_path / "baseline_stageii.pkl"
    _write_stageii(candidate_path, [0.0, 1.0, 50.0, 90.0, 91.0, 92.0])
    _write_stageii(reference_path, [0.0, 1.0, 50.0, 70.0, 71.0, 72.0])

    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_mesh_compare",
        lambda *args, **kwargs: {"frame_delta_l2": {"mean": 0.25}},
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        candidate_path,
        warmup_runs=0,
        measured_runs=1,
        mesh_reference_path=reference_path,
        mesh_support_base_dir="/tmp/support_files",
    )

    hotspots = report["quality"]["reference_stageii_chunk_seam_hotspots"]
    assert hotspots is not None
    assert hotspots["chunk_size"] == 4
    assert hotspots["chunk_overlap"] == 2
    assert hotspots["chunk_keep_starts"] == [0, 1]
    assert hotspots["nondefault_keep_starts"] == [{"chunk_index": 1, "keep_start": 1, "seam_index": 3}]
    assert hotspots["pose"]["positive_peak_count"] == 1
    assert hotspots["pose"]["negative_peak_count"] == 1
    assert hotspots["pose"]["positive_peaks"][0]["seam_index"] == 3
    assert hotspots["pose"]["positive_peaks"][0]["delta_peak_metric"] == "seam_jump_l2"
    assert hotspots["pose"]["positive_peaks"][0]["delta_peak_value"] == pytest.approx(20.0)
    assert hotspots["pose"]["negative_peaks"][0]["seam_index"] == 3
    assert hotspots["pose"]["negative_peaks"][0]["delta_trough_metric"] == "pre_accel_l2"
    assert hotspots["pose"]["negative_peaks"][0]["delta_trough_value"] == pytest.approx(-20.0)
    assert hotspots["transl"]["positive_peak_count"] == 1
    assert hotspots["transl"]["negative_peak_count"] == 1
    assert hotspots["transl"]["positive_peaks"][0]["seam_index"] == 3
    assert hotspots["transl"]["positive_peaks"][0]["delta_peak_metric"] == "seam_jump_l2"
    assert hotspots["transl"]["positive_peaks"][0]["delta_peak_value"] == pytest.approx(20.0)
    assert hotspots["transl"]["negative_peaks"][0]["seam_index"] == 3
    assert hotspots["transl"]["negative_peaks"][0]["delta_trough_metric"] == "pre_accel_l2"
    assert hotspots["transl"]["negative_peaks"][0]["delta_trough_value"] == pytest.approx(-20.0)


def test_run_public_stageii_benchmark_includes_reference_stageii_pose_window_hotspots(
    tmp_path, monkeypatch
):
    def _write_stageii(sample_path, *, body_values, left_hand_values, right_hand_values):
        fullpose = np.zeros((len(body_values), 165), dtype=np.float32)
        fullpose[:, 3] = np.asarray(body_values, dtype=np.float32)
        fullpose[:, 75] = np.asarray(left_hand_values, dtype=np.float32)
        fullpose[:, 120] = np.asarray(right_hand_values, dtype=np.float32)
        sample_path.write_bytes(
            pickle.dumps(
                {
                    "fullpose": fullpose,
                    "betas": np.zeros(10, dtype=np.float32),
                    "trans": np.zeros((len(body_values), 3), dtype=np.float32),
                    "markers_latent": np.zeros((1, 3), dtype=np.float32),
                    "latent_labels": ["A"],
                    "stageii_debug_details": {
                        "cfg": {
                            "surface_model": {
                                "type": "smplx",
                                "gender": "male",
                            }
                        },
                        "mocap_frame_rate": 120.0,
                        "mocap_time_length": len(body_values),
                        "markers_obs": np.zeros((len(body_values), 1, 3), dtype=np.float32),
                        "markers_sim": np.zeros((len(body_values), 1, 3), dtype=np.float32),
                        "labels_obs": [["A"]] * len(body_values),
                    },
                }
            )
        )

    candidate_path = tmp_path / "candidate_stageii.pkl"
    reference_path = tmp_path / "baseline_stageii.pkl"
    _write_stageii(
        candidate_path,
        body_values=[0.0, 2.0, 4.0, 4.0, 4.0],
        left_hand_values=[0.0, 0.0, 4.0, 4.0, 4.0],
        right_hand_values=[0.0, 1.0, 1.0, 4.0, 4.0],
    )
    _write_stageii(
        reference_path,
        body_values=[0.0, 0.0, 0.0, 0.0, 0.0],
        left_hand_values=[0.0, 1.0, 2.0, 3.0, 4.0],
        right_hand_values=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_mesh_compare",
        lambda *args, **kwargs: {"frame_delta_l2": {"mean": 0.25}},
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        candidate_path,
        warmup_runs=0,
        measured_runs=1,
        mesh_reference_path=reference_path,
        mesh_support_base_dir="/tmp/support_files",
    )

    hotspots = report["quality"]["reference_stageii_pose_window_hotspots"]
    assert hotspots is not None
    assert hotspots["window_size"] == stageii_benchmark.REFERENCE_STAGEII_POSE_WINDOW_SIZE
    assert hotspots["left_hand_pose_frame_delta_l2"]["window_count"] == 1
    assert hotspots["left_hand_pose_frame_delta_l2"]["positive_peak_count"] >= 1
    assert hotspots["body_pose_jitter_l2"]["positive_peak_count"] >= 1
    assert hotspots["body_pose_jitter_l2"]["positive_peaks"][0]["delta_mean"] > 0.0


def test_run_public_stageii_benchmark_leaves_reference_stageii_quality_empty_for_non_stageii_mesh_reference(
    tmp_path, monkeypatch
):
    candidate_path = tmp_path / "candidate_stageii.pkl"
    reference_path = tmp_path / "baseline.pc2"
    _write_synthetic_stageii_sample(candidate_path, residual_offset=1.0)
    reference_path.write_bytes(b"pc2")

    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_mesh_compare",
        lambda *args, **kwargs: {"frame_delta_l2": {"mean": 0.5}},
        raising=False,
    )
    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    report = run_public_stageii_benchmark(
        candidate_path,
        warmup_runs=0,
        measured_runs=1,
        mesh_reference_path=reference_path,
        mesh_support_base_dir="/tmp/support_files",
    )

    assert report["quality"]["reference_stageii_quality"] is None
    assert report["quality"]["reference_stageii_delta"] is None
    assert report["speed"]["reference_stageii_elapsed_s"] is None
    assert report["speed"]["reference_stageii_elapsed_delta_s"] is None
    assert report["quality"]["mesh_compare"] == {"frame_delta_l2": {"mean": 0.5}}


def test_run_public_stageii_benchmark_rejects_mesh_reference_that_matches_input_path(
    tmp_path, monkeypatch
):
    candidate_path = tmp_path / "candidate_stageii.pkl"
    _write_synthetic_stageii_sample(candidate_path, residual_offset=1.0)

    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        stageii_benchmark,
        "_summarize_mesh_compare",
        lambda *args, **kwargs: pytest.fail("mesh compare should not run for self reference"),
        raising=False,
    )

    with pytest.raises(ValueError, match="mesh_reference_path resolves to sample_path"):
        run_public_stageii_benchmark(
            candidate_path,
            warmup_runs=0,
            measured_runs=1,
            mesh_reference_path=candidate_path,
        )


def test_benchmark_stageii_public_main_passes_optional_mesh_reference_args(monkeypatch, capsys):
    captured = {}

    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda sample_path, *, warmup_runs, measured_runs, mesh_reference_path, mesh_support_base_dir, mesh_chunk_size, mesh_chunk_overlap, lean_benchmark: (
            captured.update(
                {
                    "sample_path": sample_path,
                    "warmup_runs": warmup_runs,
                    "measured_runs": measured_runs,
                    "mesh_reference_path": mesh_reference_path,
                    "mesh_support_base_dir": mesh_support_base_dir,
                    "mesh_chunk_size": mesh_chunk_size,
                    "mesh_chunk_overlap": mesh_chunk_overlap,
                    "lean_benchmark": lean_benchmark,
                }
            )
            or {
                "sample": {"path": sample_path},
                "quality": {"mesh_compare": None},
                "artifact": {"report_path": None},
            }
        ),
    )
    monkeypatch.setattr(
        benchmark_stageii_public,
        "write_benchmark_report",
        lambda report, output_path: captured.update({"report_path": str(output_path)})
        or {
            **report,
            "artifact": {"report_path": str(output_path)},
        },
    )

    benchmark_stageii_public.main(
        [
            "--input",
            "candidate_stageii.pkl",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
            "--mesh-reference",
            "baseline_stageii.pkl",
            "--mesh-support-base-dir",
            "/tmp/support_files",
            "--mesh-chunk-size",
            "64",
            "--mesh-chunk-overlap",
            "8",
        ]
    )

    assert captured == {
        "sample_path": "candidate_stageii.pkl",
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": "baseline_stageii.pkl",
        "mesh_support_base_dir": "/tmp/support_files",
        "mesh_chunk_size": 64,
        "mesh_chunk_overlap": 8,
        "lean_benchmark": False,
        "report_path": "candidate_benchmark.json",
    }
    output = json.loads(capsys.readouterr().out)
    assert output["sample"]["path"] == "candidate_stageii.pkl"


def test_benchmark_stageii_public_main_writes_default_report_next_to_input(monkeypatch, capsys):
    captured = {}

    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda sample_path, *, warmup_runs, measured_runs, mesh_reference_path, mesh_support_base_dir, mesh_chunk_size, mesh_chunk_overlap, lean_benchmark: (
            captured.update(
                {
                    "sample_path": sample_path,
                    "warmup_runs": warmup_runs,
                    "measured_runs": measured_runs,
                    "mesh_reference_path": mesh_reference_path,
                    "mesh_support_base_dir": mesh_support_base_dir,
                    "mesh_chunk_size": mesh_chunk_size,
                    "mesh_chunk_overlap": mesh_chunk_overlap,
                    "lean_benchmark": lean_benchmark,
                }
            )
            or {
                "sample": {"path": sample_path},
                "quality": {"mesh_compare": None},
                "artifact": {"report_path": None},
            }
        ),
    )
    monkeypatch.setattr(
        benchmark_stageii_public,
        "write_benchmark_report",
        lambda report, output_path: captured.update({"report_path": str(output_path)})
        or {
            **report,
            "artifact": {"report_path": str(output_path)},
        },
    )

    benchmark_stageii_public.main(
        [
            "--input",
            "/tmp/work/input/wolf001/candidate_stageii.pkl",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ]
    )

    assert captured == {
        "sample_path": "/tmp/work/input/wolf001/candidate_stageii.pkl",
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": None,
        "mesh_support_base_dir": None,
        "mesh_chunk_size": None,
        "mesh_chunk_overlap": None,
        "lean_benchmark": False,
        "report_path": "/tmp/work/input/wolf001/candidate_benchmark.json",
    }
    output = json.loads(capsys.readouterr().out)
    assert output["artifact"]["report_path"] == "/tmp/work/input/wolf001/candidate_benchmark.json"


def test_benchmark_stageii_public_main_defaults_mesh_support_base_dir_when_mesh_reference_present(
    monkeypatch, capsys
):
    captured = {}

    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda sample_path, *, warmup_runs, measured_runs, mesh_reference_path, mesh_support_base_dir, mesh_chunk_size, mesh_chunk_overlap, lean_benchmark: (
            captured.update(
                {
                    "sample_path": sample_path,
                    "warmup_runs": warmup_runs,
                    "measured_runs": measured_runs,
                    "mesh_reference_path": mesh_reference_path,
                    "mesh_support_base_dir": mesh_support_base_dir,
                    "mesh_chunk_size": mesh_chunk_size,
                    "mesh_chunk_overlap": mesh_chunk_overlap,
                    "lean_benchmark": lean_benchmark,
                }
            )
            or {
                "sample": {"path": sample_path},
                "quality": {"mesh_compare": None},
                "artifact": {"report_path": None},
            }
        ),
    )
    monkeypatch.setattr(
        benchmark_stageii_public,
        "write_benchmark_report",
        lambda report, output_path: {
            **report,
            "artifact": {"report_path": str(output_path)},
        },
    )

    benchmark_stageii_public.main(
        [
            "--input",
            "candidate_stageii.pkl",
            "--mesh-reference",
            "baseline_stageii.pkl",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ]
    )

    assert captured == {
        "sample_path": "candidate_stageii.pkl",
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": "baseline_stageii.pkl",
        "mesh_support_base_dir": "support_files",
        "mesh_chunk_size": None,
        "mesh_chunk_overlap": None,
        "lean_benchmark": False,
    }
    assert json.loads(capsys.readouterr().out)["sample"]["path"] == "candidate_stageii.pkl"


def test_benchmark_stageii_public_main_passes_lean_benchmark_flag(monkeypatch, capsys):
    captured = {}

    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda sample_path, *, warmup_runs, measured_runs, mesh_reference_path, mesh_support_base_dir, mesh_chunk_size, mesh_chunk_overlap, lean_benchmark: (
            captured.update(
                {
                    "sample_path": sample_path,
                    "lean_benchmark": lean_benchmark,
                }
            )
            or {
                "sample": {"path": sample_path},
                "quality": {"mesh_compare": None},
                "artifact": {"report_path": None},
            }
        ),
    )
    monkeypatch.setattr(
        benchmark_stageii_public,
        "write_benchmark_report",
        lambda report, output_path: {
            **report,
            "artifact": {"report_path": str(output_path)},
        },
    )

    benchmark_stageii_public.main(
        [
            "--input",
            "candidate_stageii.pkl",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
            "--lean-benchmark",
        ]
    )

    assert captured == {
        "sample_path": "candidate_stageii.pkl",
        "lean_benchmark": True,
    }
    assert json.loads(capsys.readouterr().out)["sample"]["path"] == "candidate_stageii.pkl"


def test_benchmark_stageii_public_main_rejects_mesh_chunk_size_without_mesh_reference(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                "candidate_stageii.pkl",
                "--mesh-chunk-size",
                "64",
            ]
        )

    assert "--mesh-chunk-size requires --mesh-reference" in capsys.readouterr().err


def test_benchmark_stageii_public_main_rejects_negative_warmup_runs(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                "candidate_stageii.pkl",
                "--warmup-runs",
                "-1",
            ]
        )

    assert "--warmup-runs must be >= 0" in capsys.readouterr().err


def test_benchmark_stageii_public_main_rejects_non_positive_mesh_chunk_size(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                "candidate_stageii.pkl",
                "--mesh-reference",
                "baseline.pc2",
                "--mesh-chunk-size",
                "0",
            ]
        )

    assert "--mesh-chunk-size must be > 0" in capsys.readouterr().err


def test_benchmark_stageii_public_main_rejects_mesh_support_base_dir_without_mesh_reference(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                "candidate_stageii.pkl",
                "--mesh-support-base-dir",
                "/tmp/support_files",
            ]
        )

    assert "--mesh-support-base-dir requires --mesh-reference" in capsys.readouterr().err


def test_benchmark_stageii_public_main_rejects_mesh_chunk_overlap_without_mesh_chunk_size(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                "candidate_stageii.pkl",
                "--mesh-reference",
                "baseline.pc2",
                "--mesh-chunk-overlap",
                "4",
            ]
        )

    assert "--mesh-chunk-overlap requires --mesh-chunk-size" in capsys.readouterr().err


def test_benchmark_stageii_public_main_errors_when_mesh_reference_matches_input(
    tmp_path, monkeypatch, capsys
):
    candidate_path = tmp_path / "candidate_stageii.pkl"
    _write_synthetic_stageii_sample(candidate_path, residual_offset=1.0)

    monkeypatch.setattr(stageii_benchmark, "_benchmark_preview_vertex_decode", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mesh_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_mp4_render", lambda *args, **kwargs: None)
    monkeypatch.setattr(stageii_benchmark, "_benchmark_artifact_bundle_export", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                str(candidate_path),
                "--mesh-reference",
                str(candidate_path),
            ]
        )

    assert "mesh_reference_path resolves to sample_path" in capsys.readouterr().err


def test_benchmark_stageii_public_main_rejects_output_matching_input(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: {
            "sample": {"path": "candidate_stageii.pkl"},
            "artifact": {"report_path": None},
        },
    )
    monkeypatch.setattr(
        benchmark_stageii_public,
        "write_benchmark_report",
        lambda *args, **kwargs: pytest.fail("write_benchmark_report should not run when output matches input"),
    )

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                "candidate_stageii.pkl",
                "--output",
                "candidate_stageii.pkl",
            ]
        )

    assert "benchmark output resolves to benchmark input path" in capsys.readouterr().err


def test_benchmark_stageii_public_main_rejects_output_matching_mesh_reference(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: {
            "sample": {"path": "candidate_stageii.pkl"},
            "artifact": {"report_path": None},
        },
    )
    monkeypatch.setattr(
        benchmark_stageii_public,
        "write_benchmark_report",
        lambda *args, **kwargs: pytest.fail(
            "write_benchmark_report should not run when output matches mesh reference"
        ),
    )

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                "candidate_stageii.pkl",
                "--mesh-reference",
                "baseline_stageii.pkl",
                "--output",
                "baseline_stageii.pkl",
            ]
        )

    assert "benchmark output resolves to mesh reference path" in capsys.readouterr().err


def test_benchmark_stageii_public_main_errors_when_benchmark_writer_returns_drifted_report_path(
    tmp_path, monkeypatch, capsys
):
    expected_output = tmp_path / "candidate_benchmark.json"

    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: {
            "sample": {"path": "candidate_stageii.pkl"},
            "artifact": {"report_path": None},
        },
    )
    monkeypatch.setattr(
        benchmark_stageii_public,
        "write_benchmark_report",
        lambda report, output_path: {
            **report,
            "artifact": {"report_path": str(tmp_path / "drifted_benchmark.json")},
        },
    )

    with pytest.raises(SystemExit):
        benchmark_stageii_public.main(
            [
                "--input",
                "candidate_stageii.pkl",
                "--output",
                str(expected_output),
            ]
        )

    assert "benchmark payload report_path drifted from requested output path" in capsys.readouterr().err


def test_summarize_stageii_quality_reports_marker_jitter_and_seam_metrics_for_new_format(
    tmp_path,
):
    sample_path = tmp_path / "synthetic_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.asarray([[0.0], [1.0], [2.0], [10.0], [11.0]], dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [11.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        },
                        "runtime": {
                            "sequence_chunk_size": 3,
                            "sequence_chunk_overlap": 1,
                        },
                    },
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 5,
                    "markers_obs": np.zeros((5, 1, 3), dtype=np.float32),
                    "markers_sim": np.tile(
                        np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32),
                        (5, 1, 1),
                    ),
                    "labels_obs": [["A"]] * 5,
                },
            }
        )
    )

    quality = stageii_benchmark._summarize_stageii_quality(
        sample_path,
        stageii_benchmark.normalize_stageii_sample(sample_path),
    )

    assert quality["marker_residual_l2"]["count"] == 5
    assert quality["marker_residual_l2"]["mean"] == pytest.approx(1.0)
    assert quality["trans_frame_delta_l2"]["count"] == 4
    assert quality["trans_frame_delta_l2"]["mean"] == pytest.approx(2.75)
    assert quality["trans_frame_delta_l2"]["max"] == pytest.approx(8.0)
    assert quality["pose_frame_delta_l2"]["count"] == 4
    assert quality["pose_frame_delta_l2"]["mean"] == pytest.approx(2.75)
    assert quality["pose_frame_delta_l2"]["max"] == pytest.approx(8.0)
    assert quality["trans_jitter_l2"]["max"] == pytest.approx(7.0)
    assert quality["pose_jitter_l2"]["max"] == pytest.approx(7.0)
    assert quality["chunk_seam_transl_jump_l2"]["count"] == 1
    assert quality["chunk_seam_transl_jump_l2"]["mean"] == pytest.approx(8.0)
    assert quality["chunk_seam_pose_jump_l2"]["mean"] == pytest.approx(8.0)


def test_summarize_stageii_quality_reports_hand_and_body_pose_metrics_for_smplx_fullpose(tmp_path):
    sample_path = tmp_path / "synthetic_stageii.pkl"
    fullpose = np.zeros((5, 165), dtype=np.float32)
    fullpose[:, 3] = np.asarray([0.0, 1.0, 2.0, 4.0, 7.0], dtype=np.float32)
    fullpose[:, 75] = np.asarray([0.0, 2.0, 2.0, 2.0, 3.0], dtype=np.float32)
    fullpose[:, 120] = np.asarray([0.0, 0.0, 1.0, 1.0, 4.0], dtype=np.float32)
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": fullpose,
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [11.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        },
                        "runtime": {
                            "sequence_chunk_size": 3,
                            "sequence_chunk_overlap": 1,
                        },
                    },
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 5,
                    "markers_obs": np.zeros((5, 1, 3), dtype=np.float32),
                    "markers_sim": np.zeros((5, 1, 3), dtype=np.float32),
                    "labels_obs": [["A"]] * 5,
                },
            }
        )
    )

    quality = stageii_benchmark._summarize_stageii_quality(
        sample_path,
        stageii_benchmark.normalize_stageii_sample(sample_path),
    )

    assert quality["body_pose_frame_delta_l2"]["count"] == 4
    assert quality["body_pose_frame_delta_l2"]["mean"] == pytest.approx(1.75)
    assert quality["body_pose_frame_delta_l2"]["max"] == pytest.approx(3.0)
    assert quality["body_pose_jitter_l2"]["count"] == 3
    assert quality["body_pose_jitter_l2"]["mean"] == pytest.approx(2.0 / 3.0)
    assert quality["left_hand_pose_frame_delta_l2"]["count"] == 4
    assert quality["left_hand_pose_frame_delta_l2"]["mean"] == pytest.approx(0.75)
    assert quality["left_hand_pose_jitter_l2"]["max"] == pytest.approx(2.0)
    assert quality["right_hand_pose_frame_delta_l2"]["mean"] == pytest.approx(1.0)
    assert quality["right_hand_pose_jitter_l2"]["mean"] == pytest.approx(5.0 / 3.0)
    assert quality["all_hands_pose_frame_delta_l2"]["mean"] == pytest.approx((3.0 + np.sqrt(10.0)) / 4.0)
    assert quality["all_hands_pose_jitter_l2"]["max"] == pytest.approx(np.sqrt(10.0))


def test_summarize_stageii_quality_rejects_non_positive_runtime_chunk_size(tmp_path):
    sample_path = tmp_path / "synthetic_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.zeros((3, 3), dtype=np.float32),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        },
                        "runtime": {
                            "sequence_chunk_size": 0,
                            "sequence_chunk_overlap": 1,
                        },
                    },
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 3,
                    "markers_obs": np.zeros((3, 1, 3), dtype=np.float32),
                    "markers_sim": np.zeros((3, 1, 3), dtype=np.float32),
                    "labels_obs": [["A"]] * 3,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="runtime.sequence_chunk_size must be > 0"):
        stageii_benchmark._summarize_stageii_quality(
            sample_path,
            stageii_benchmark.normalize_stageii_sample(sample_path),
        )


def test_summarize_stageii_quality_uses_explicit_chunk_keep_starts_when_present(tmp_path):
    sample_path = tmp_path / "synthetic_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.asarray([[0.0], [1.0], [50.0], [90.0], [91.0], [92.0]], dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [50.0, 0.0, 0.0],
                        [90.0, 0.0, 0.0],
                        [91.0, 0.0, 0.0],
                        [92.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        },
                        "runtime": {
                            "sequence_chunk_size": 4,
                            "sequence_chunk_overlap": 2,
                        },
                    },
                    "sequence_chunk_keep_starts": [0, 1],
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 6,
                    "markers_obs": np.zeros((6, 1, 3), dtype=np.float32),
                    "markers_sim": np.tile(
                        np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32),
                        (6, 1, 1),
                    ),
                    "labels_obs": [["A"]] * 6,
                },
            }
        )
    )

    quality = stageii_benchmark._summarize_stageii_quality(
        sample_path,
        stageii_benchmark.normalize_stageii_sample(sample_path),
    )

    assert quality["chunk_seam_transl_jump_l2"]["count"] == 1
    assert quality["chunk_seam_transl_jump_l2"]["mean"] == pytest.approx(40.0)
    assert quality["chunk_seam_pose_jump_l2"]["mean"] == pytest.approx(40.0)


def test_chunk_seam_local_diagnostics_reports_delta_and_accel_around_keep_start():
    diagnostics = stageii_benchmark._chunk_seam_local_diagnostics(
        np.asarray([[0.0], [1.0], [50.0], [90.0], [91.0], [92.0]], dtype=np.float32),
        chunk_size=4,
        overlap=2,
        keep_starts=[0, 1],
    )

    assert diagnostics == [
        {
            "chunk_index": 1,
            "row_start": 2,
            "row_end": 6,
            "keep_start": 1,
            "trim_count": 1,
            "seam_index": 3,
            "prev_frame_delta_l2": pytest.approx(49.0),
            "seam_jump_l2": pytest.approx(40.0),
            "next_frame_delta_l2": pytest.approx(1.0),
            "pre_accel_l2": pytest.approx(9.0),
            "post_accel_l2": pytest.approx(39.0),
        }
    ]


def test_summarize_stageii_chunk_seam_diagnostics_uses_runtime_chunk_config_and_keep_starts(tmp_path):
    sample_path = tmp_path / "synthetic_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.asarray([[0.0], [1.0], [50.0], [90.0], [91.0], [92.0]], dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [50.0, 0.0, 0.0],
                        [90.0, 0.0, 0.0],
                        [91.0, 0.0, 0.0],
                        [92.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        },
                        "runtime": {
                            "sequence_chunk_size": 4,
                            "sequence_chunk_overlap": 2,
                        },
                    },
                    "sequence_chunk_keep_starts": [0, 1],
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 6,
                    "markers_obs": np.zeros((6, 1, 3), dtype=np.float32),
                    "markers_sim": np.zeros((6, 1, 3), dtype=np.float32),
                    "labels_obs": [["A"]] * 6,
                },
            }
        )
    )

    diagnostics = stageii_benchmark.summarize_stageii_chunk_seam_diagnostics(sample_path)

    assert diagnostics["chunk_size"] == 4
    assert diagnostics["chunk_overlap"] == 2
    assert diagnostics["chunk_keep_starts"] == [0, 1]
    assert diagnostics["transl"][0]["seam_index"] == 3
    assert diagnostics["transl"][0]["seam_jump_l2"] == pytest.approx(40.0)
    assert diagnostics["pose"][0]["post_accel_l2"] == pytest.approx(39.0)


def test_compare_stageii_chunk_seam_diagnostics_reports_same_frame_reference_deltas(tmp_path):
    def _write_stageii(sample_path, values):
        sample_path.write_bytes(
            pickle.dumps(
                {
                    "fullpose": np.asarray([[value] for value in values], dtype=np.float32),
                    "betas": np.zeros(10, dtype=np.float32),
                    "trans": np.asarray([[value, 0.0, 0.0] for value in values], dtype=np.float32),
                    "markers_latent": np.zeros((1, 3), dtype=np.float32),
                    "latent_labels": ["A"],
                    "stageii_debug_details": {
                        "cfg": {
                            "surface_model": {
                                "type": "smplx",
                                "gender": "male",
                            },
                            "runtime": {
                                "sequence_chunk_size": 4,
                                "sequence_chunk_overlap": 2,
                            },
                        },
                        "sequence_chunk_keep_starts": [0, 1],
                        "mocap_frame_rate": 120.0,
                        "mocap_time_length": len(values),
                        "markers_obs": np.zeros((len(values), 1, 3), dtype=np.float32),
                        "markers_sim": np.zeros((len(values), 1, 3), dtype=np.float32),
                        "labels_obs": [["A"]] * len(values),
                    },
                }
            )
        )

    reference_path = tmp_path / "reference_stageii.pkl"
    candidate_path = tmp_path / "candidate_stageii.pkl"
    _write_stageii(reference_path, [0.0, 1.0, 50.0, 70.0, 71.0, 72.0])
    _write_stageii(candidate_path, [0.0, 1.0, 50.0, 90.0, 91.0, 92.0])

    comparison = stageii_benchmark.compare_stageii_chunk_seam_diagnostics(reference_path, candidate_path)

    assert comparison["chunk_keep_starts"] == [0, 1]
    assert comparison["pose"][0]["seam_index"] == 3
    assert comparison["pose"][0]["candidate"]["seam_jump_l2"] == pytest.approx(40.0)
    assert comparison["pose"][0]["reference"]["seam_jump_l2"] == pytest.approx(20.0)
    assert comparison["pose"][0]["delta"]["seam_jump_l2"] == pytest.approx(20.0)
    assert comparison["pose"][0]["candidate"]["post_accel_l2"] == pytest.approx(39.0)
    assert comparison["pose"][0]["reference"]["post_accel_l2"] == pytest.approx(19.0)
    assert comparison["pose"][0]["delta"]["post_accel_l2"] == pytest.approx(20.0)


def test_summarize_compared_stageii_chunk_seam_diagnostics_reports_hotspots_and_nondefault_keeps():
    comparison = {
        "chunk_size": 48,
        "chunk_overlap": 8,
        "chunk_keep_starts": [0, 8, 6, 8, 5],
        "pose": [
            {
                "chunk_index": 1,
                "row_start": 40,
                "row_end": 88,
                "keep_start": 8,
                "trim_count": 0,
                "seam_index": 88,
                "candidate": {
                    "prev_frame_delta_l2": 0.3,
                    "seam_jump_l2": 0.2,
                    "next_frame_delta_l2": 0.4,
                    "pre_accel_l2": 0.5,
                    "post_accel_l2": 0.9,
                },
                "reference": {
                    "prev_frame_delta_l2": 0.25,
                    "seam_jump_l2": 0.15,
                    "next_frame_delta_l2": 0.35,
                    "pre_accel_l2": 0.45,
                    "post_accel_l2": 0.4,
                },
                "delta": {
                    "prev_frame_delta_l2": 0.05,
                    "seam_jump_l2": 0.05,
                    "next_frame_delta_l2": 0.05,
                    "pre_accel_l2": 0.05,
                    "post_accel_l2": 0.5,
                },
            },
            {
                "chunk_index": 2,
                "row_start": 80,
                "row_end": 128,
                "keep_start": 6,
                "trim_count": 2,
                "seam_index": 126,
                "candidate": {
                    "prev_frame_delta_l2": 0.35,
                    "seam_jump_l2": 0.7,
                    "next_frame_delta_l2": 0.3,
                    "pre_accel_l2": 0.6,
                    "post_accel_l2": 0.8,
                },
                "reference": {
                    "prev_frame_delta_l2": 0.3,
                    "seam_jump_l2": 0.3,
                    "next_frame_delta_l2": 0.2,
                    "pre_accel_l2": 0.55,
                    "post_accel_l2": 0.45,
                },
                "delta": {
                    "prev_frame_delta_l2": 0.05,
                    "seam_jump_l2": 0.4,
                    "next_frame_delta_l2": 0.1,
                    "pre_accel_l2": 0.05,
                    "post_accel_l2": 0.35,
                },
            },
            {
                "chunk_index": 3,
                "row_start": 120,
                "row_end": 168,
                "keep_start": 5,
                "trim_count": 3,
                "seam_index": 165,
                "candidate": {
                    "prev_frame_delta_l2": 0.25,
                    "seam_jump_l2": 0.15,
                    "next_frame_delta_l2": 0.2,
                    "pre_accel_l2": 0.45,
                    "post_accel_l2": 0.3,
                },
                "reference": {
                    "prev_frame_delta_l2": 0.2,
                    "seam_jump_l2": 0.45,
                    "next_frame_delta_l2": 0.15,
                    "pre_accel_l2": 0.4,
                    "post_accel_l2": 0.35,
                },
                "delta": {
                    "prev_frame_delta_l2": 0.05,
                    "seam_jump_l2": -0.3,
                    "next_frame_delta_l2": 0.05,
                    "pre_accel_l2": 0.05,
                    "post_accel_l2": -0.05,
                },
            },
            {
                "chunk_index": 4,
                "row_start": 160,
                "row_end": 208,
                "keep_start": 8,
                "trim_count": 0,
                "seam_index": 208,
                "candidate": {
                    "prev_frame_delta_l2": 0.2,
                    "seam_jump_l2": 0.3,
                    "next_frame_delta_l2": 0.25,
                    "pre_accel_l2": 0.55,
                    "post_accel_l2": 0.4,
                },
                "reference": {
                    "prev_frame_delta_l2": 0.15,
                    "seam_jump_l2": 0.2,
                    "next_frame_delta_l2": 0.2,
                    "pre_accel_l2": 0.85,
                    "post_accel_l2": 0.5,
                },
                "delta": {
                    "prev_frame_delta_l2": 0.05,
                    "seam_jump_l2": 0.1,
                    "next_frame_delta_l2": 0.05,
                    "pre_accel_l2": -0.3,
                    "post_accel_l2": -0.1,
                },
            },
        ],
        "transl": [],
    }

    summary = stageii_benchmark.summarize_compared_stageii_chunk_seam_diagnostics(
        comparison,
        top_k=2,
        positive_threshold=0.3,
        negative_threshold=0.2,
    )

    assert summary["chunk_size"] == 48
    assert summary["chunk_overlap"] == 8
    assert summary["chunk_keep_starts"] == [0, 8, 6, 8, 5]
    assert summary["nondefault_keep_starts"] == [
        {"chunk_index": 2, "keep_start": 6, "seam_index": 126},
        {"chunk_index": 3, "keep_start": 5, "seam_index": 165},
    ]
    assert summary["pose"]["seam_count"] == 4
    assert summary["pose"]["positive_peak_count"] == 2
    assert summary["pose"]["negative_peak_count"] == 2
    assert [row["seam_index"] for row in summary["pose"]["positive_peaks"]] == [88, 126]
    assert summary["pose"]["positive_peaks"][0]["delta_peak_metric"] == "post_accel_l2"
    assert summary["pose"]["positive_peaks"][0]["delta_peak_value"] == pytest.approx(0.5)
    assert [row["seam_index"] for row in summary["pose"]["negative_peaks"]] == [165, 208]
    assert summary["pose"]["negative_peaks"][0]["delta_trough_metric"] == "seam_jump_l2"
    assert summary["pose"]["negative_peaks"][0]["delta_trough_value"] == pytest.approx(-0.3)
    assert summary["transl"]["seam_count"] == 0
    assert summary["transl"]["positive_peak_count"] == 0
    assert summary["transl"]["negative_peak_count"] == 0
    assert summary["transl"]["positive_peaks"] == []
    assert summary["transl"]["negative_peaks"] == []


def test_summarize_compared_stageii_pose_window_hotspots_reports_hand_and_body_windows(tmp_path):
    def _write_stageii(sample_path, *, body_values, left_hand_values, right_hand_values):
        fullpose = np.zeros((len(body_values), 165), dtype=np.float32)
        fullpose[:, 3] = np.asarray(body_values, dtype=np.float32)
        fullpose[:, 75] = np.asarray(left_hand_values, dtype=np.float32)
        fullpose[:, 120] = np.asarray(right_hand_values, dtype=np.float32)
        sample_path.write_bytes(
            pickle.dumps(
                {
                    "fullpose": fullpose,
                    "betas": np.zeros(10, dtype=np.float32),
                    "trans": np.zeros((len(body_values), 3), dtype=np.float32),
                    "markers_latent": np.zeros((1, 3), dtype=np.float32),
                    "latent_labels": ["A"],
                    "stageii_debug_details": {
                        "cfg": {
                            "surface_model": {
                                "type": "smplx",
                                "gender": "male",
                            }
                        },
                        "mocap_frame_rate": 120.0,
                        "mocap_time_length": len(body_values),
                        "markers_obs": np.zeros((len(body_values), 1, 3), dtype=np.float32),
                        "markers_sim": np.zeros((len(body_values), 1, 3), dtype=np.float32),
                        "labels_obs": [["A"]] * len(body_values),
                    },
                }
            )
        )

    reference_path = tmp_path / "reference_stageii.pkl"
    candidate_path = tmp_path / "candidate_stageii.pkl"
    _write_stageii(
        candidate_path,
        body_values=[0.0, 2.0, 4.0, 4.0, 4.0],
        left_hand_values=[0.0, 0.0, 4.0, 4.0, 4.0],
        right_hand_values=[0.0, 1.0, 1.0, 4.0, 4.0],
    )
    _write_stageii(
        reference_path,
        body_values=[0.0, 0.0, 0.0, 0.0, 0.0],
        left_hand_values=[0.0, 1.0, 2.0, 3.0, 4.0],
        right_hand_values=[0.0, 1.0, 2.0, 3.0, 4.0],
    )

    summary = stageii_benchmark.summarize_compared_stageii_pose_window_hotspots(
        reference_path,
        candidate_path,
        window_size=1,
        top_k=2,
        positive_threshold=0.5,
        negative_threshold=0.5,
    )

    assert summary["window_size"] == 1
    assert summary["left_hand_pose_frame_delta_l2"]["positive_peak_count"] == 1
    assert summary["left_hand_pose_frame_delta_l2"]["positive_peaks"][0]["frame_start"] == 2
    assert summary["left_hand_pose_frame_delta_l2"]["positive_peaks"][0]["delta_mean"] == pytest.approx(3.0)
    assert summary["left_hand_pose_frame_delta_l2"]["negative_peak_count"] == 3
    assert summary["body_pose_jitter_l2"]["positive_peak_count"] == 1
    assert summary["body_pose_jitter_l2"]["positive_peaks"][0]["frame_start"] == 3
    assert summary["body_pose_jitter_l2"]["positive_peaks"][0]["delta_mean"] == pytest.approx(2.0)
    assert summary["all_hands_pose_frame_delta_l2"]["positive_peak_count"] == 2


def test_summarize_stageii_quality_reads_legacy_marker_residual_from_public_sample():
    sample_path = ROOT / "support_data/tests/mosh_stageii.pkl"
    quality = stageii_benchmark._summarize_stageii_quality(
        sample_path,
        stageii_benchmark.normalize_stageii_sample(sample_path),
    )

    assert quality["marker_residual_l2"]["count"] > 0
    assert quality["marker_residual_l2"]["mean"] >= 0.0
    assert quality["trans_frame_delta_l2"]["count"] > 0
    assert quality["pose_frame_delta_l2"]["count"] > 0
    assert quality["trans_jitter_l2"]["count"] > 0
    assert quality["chunk_seam_transl_jump_l2"] is None


def test_marker_residual_l2_samples_ignore_unobserved_markers_in_chunked_full_lattice_frames():
    stageii_data = {
        "latent_labels": ["A", "B"],
        "stageii_debug_details": {
            "markers_obs": np.asarray(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ],
                dtype=np.float32,
            ),
            "markers_sim": np.asarray(
                [
                    [
                        [1.0, 0.0, 0.0],
                        [50.0, 0.0, 0.0],
                    ],
                    [
                        [60.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ],
                ],
                dtype=np.float32,
            ),
            "labels_obs": [["A"], ["B"]],
        },
    }

    residuals = stageii_benchmark._marker_residual_l2_samples(stageii_data)

    assert residuals == pytest.approx([1.0, 1.0])


def test_summarize_stageii_quality_omits_bulky_raw_samples(tmp_path):
    sample_path = tmp_path / "synthetic_stageii.pkl"
    sample_path.write_bytes(
        pickle.dumps(
            {
                "fullpose": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
                "betas": np.zeros(10, dtype=np.float32),
                "trans": np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                "markers_latent": np.zeros((1, 3), dtype=np.float32),
                "latent_labels": ["A"],
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": "smplx",
                            "gender": "male",
                        }
                    },
                    "mocap_frame_rate": 120.0,
                    "mocap_time_length": 3,
                    "markers_obs": np.zeros((3, 1, 3), dtype=np.float32),
                    "markers_sim": np.ones((3, 1, 3), dtype=np.float32),
                    "labels_obs": [["A"]] * 3,
                },
            }
        )
    )

    quality = stageii_benchmark._summarize_stageii_quality(
        sample_path,
        stageii_benchmark.normalize_stageii_sample(sample_path),
    )

    assert "samples" not in quality["marker_residual_l2"]
    assert "samples" not in quality["trans_frame_delta_l2"]
    assert "samples" not in quality["pose_frame_delta_l2"]
    assert "samples" not in quality["trans_jitter_l2"]
    assert "samples" not in quality["pose_jitter_l2"]


def test_benchmark_preview_vertex_decode_propagates_unexpected_render_failures(monkeypatch):
    baseline = normalize_stageii_sample(ROOT / "support_data/tests/mosh_stageii.pkl")
    fake_render_video = type(
        "FakeRenderVideo",
        (),
        {
            "load_render_model": staticmethod(lambda model_path: (_ for _ in ()).throw(RuntimeError("boom"))),
            "load_vertices": staticmethod(lambda sample_path, model: None),
        },
    )

    monkeypatch.setattr(
        stageii_benchmark,
        "_preview_render_model_path",
        lambda repo_root, *, gender: ROOT / "support_files/smplx/male/model.npz",
    )
    monkeypatch.setitem(sys.modules, "render_video", fake_render_video)

    with pytest.raises(RuntimeError, match="boom"):
        stageii_benchmark._benchmark_preview_vertex_decode(
            ROOT / "support_data/tests/mosh_stageii.pkl",
            baseline,
            repo_root=ROOT,
            warmup_runs=0,
            measured_runs=1,
        )


def test_benchmark_mesh_export_propagates_unexpected_export_failures(monkeypatch):
    baseline = normalize_stageii_sample(ROOT / "support_data/tests/mosh_stageii.pkl")
    fake_save_smplx_verts = type(
        "FakeSaveSmplxVerts",
        (),
        {
            "export_stageii_meshes": staticmethod(
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
            ),
        },
    )

    monkeypatch.setattr(
        stageii_benchmark,
        "_preview_render_model_path",
        lambda repo_root, *, gender: ROOT / "support_files/smplx/male/model.npz",
    )
    monkeypatch.setitem(sys.modules, "save_smplx_verts", fake_save_smplx_verts)

    with pytest.raises(RuntimeError, match="boom"):
        stageii_benchmark._benchmark_mesh_export(
            ROOT / "support_data/tests/mosh_stageii.pkl",
            baseline,
            repo_root=ROOT,
            warmup_runs=0,
            measured_runs=1,
        )


def test_benchmark_mp4_render_propagates_unexpected_render_failures(monkeypatch):
    baseline = normalize_stageii_sample(ROOT / "support_data/tests/mosh_stageii.pkl")
    preloaded_model = type("FakeModel", (), {"faces": np.asarray([[0, 1, 2]], dtype=np.int32)})()
    fake_render_video = type(
        "FakeRenderVideo",
        (),
        {
            "load_render_model": staticmethod(lambda model_path: preloaded_model),
            "load_vertices": staticmethod(lambda sample_path, model: np.zeros((2, 3, 3), dtype=np.float32)),
            "render_vertices_to_video": staticmethod(
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
            ),
        },
    )

    monkeypatch.setattr(
        stageii_benchmark,
        "_preview_render_model_path",
        lambda repo_root, *, gender: ROOT / "support_files/smplx/male/model.npz",
    )
    monkeypatch.setitem(sys.modules, "render_video", fake_render_video)

    with pytest.raises(RuntimeError, match="boom"):
        stageii_benchmark._benchmark_mp4_render(
            ROOT / "support_data/tests/mosh_stageii.pkl",
            baseline,
            repo_root=ROOT,
            warmup_runs=0,
            measured_runs=1,
        )


def test_benchmark_mesh_export_loads_model_once_and_reuses_it(monkeypatch):
    baseline = normalize_stageii_sample(ROOT / "support_data/tests/mosh_stageii.pkl")
    load_calls = []
    export_models = []
    preloaded_model = object()

    fake_render_video = type(
        "FakeRenderVideo",
        (),
        {
            "load_render_model": staticmethod(lambda model_path: load_calls.append(model_path) or preloaded_model),
        },
    )

    def fake_export_stageii_meshes(*, input_pkl, model_path=None, model=None, obj_out=None, pc2_out=None):
        export_models.append(model)
        Path(obj_out).write_text("obj")
        Path(pc2_out).write_bytes(b"pc2")
        return str(obj_out), str(pc2_out)

    fake_save_smplx_verts = type(
        "FakeSaveSmplxVerts",
        (),
        {
            "export_stageii_meshes": staticmethod(fake_export_stageii_meshes),
        },
    )

    monkeypatch.setattr(
        stageii_benchmark,
        "_preview_render_model_path",
        lambda repo_root, *, gender: ROOT / "support_files/smplx/male/model.npz",
    )
    monkeypatch.setitem(sys.modules, "render_video", fake_render_video)
    monkeypatch.setitem(sys.modules, "save_smplx_verts", fake_save_smplx_verts)

    summary = stageii_benchmark._benchmark_mesh_export(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        baseline,
        repo_root=ROOT,
        warmup_runs=1,
        measured_runs=2,
    )

    assert summary["count"] == 2
    assert len(load_calls) == 1
    assert export_models == [preloaded_model, preloaded_model, preloaded_model]


def test_benchmark_mp4_render_loads_render_inputs_once_and_reuses_them(monkeypatch):
    baseline = normalize_stageii_sample(ROOT / "support_data/tests/mosh_stageii.pkl")
    load_calls = []
    vertex_calls = []
    render_calls = []
    preloaded_model = type("FakeModel", (), {"faces": np.asarray([[0, 1, 2]], dtype=np.int32)})()
    predecoded_vertices = np.zeros((2, 3, 3), dtype=np.float32)

    fake_render_video = type(
        "FakeRenderVideo",
        (),
        {
            "load_render_model": staticmethod(lambda model_path: load_calls.append(model_path) or preloaded_model),
            "load_vertices": staticmethod(
                lambda sample_path, model: vertex_calls.append((sample_path, model)) or predecoded_vertices
            ),
            "render_vertices_to_video": staticmethod(
                lambda *, vertices, faces, output_path, **kwargs: render_calls.append((vertices, faces, kwargs))
                or Path(output_path).write_bytes(b"mp4")
            ),
        },
    )

    monkeypatch.setattr(
        stageii_benchmark,
        "_preview_render_model_path",
        lambda repo_root, *, gender: ROOT / "support_files/smplx/male/model.npz",
    )
    monkeypatch.setitem(sys.modules, "render_video", fake_render_video)

    summary = stageii_benchmark._benchmark_mp4_render(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        baseline,
        repo_root=ROOT,
        warmup_runs=1,
        measured_runs=2,
    )

    assert summary["count"] == 2
    assert len(load_calls) == 1
    assert vertex_calls == [(ROOT / "support_data/tests/mosh_stageii.pkl", preloaded_model)]
    assert len(render_calls) == 3
    assert all(vertices is predecoded_vertices for vertices, _, _ in render_calls)
    assert all(np.array_equal(faces, preloaded_model.faces) for _, faces, _ in render_calls)


def test_benchmark_artifact_bundle_export_loads_render_inputs_once_and_reuses_them(monkeypatch):
    baseline = normalize_stageii_sample(ROOT / "support_data/tests/mosh_stageii.pkl")
    load_calls = []
    vertex_calls = []
    bundle_calls = []
    preloaded_model = type("FakeModel", (), {"faces": np.asarray([[0, 1, 2]], dtype=np.int32)})()
    predecoded_vertices = np.zeros((2, 3, 3), dtype=np.float32)

    fake_render_video = type(
        "FakeRenderVideo",
        (),
        {
            "load_render_model": staticmethod(lambda model_path: load_calls.append(model_path) or preloaded_model),
            "load_vertices": staticmethod(
                lambda sample_path, model: vertex_calls.append((sample_path, model)) or predecoded_vertices
            ),
        },
    )

    def fake_export_stageii_artifacts(*, input_pkl, model_path=None, model=None, vertices=None, **kwargs):
        bundle_calls.append((input_pkl, model, vertices, kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "benchmark_stageii.obj").write_text("obj")
        (output_dir / "benchmark_stageii.pc2").write_bytes(b"pc2")
        (output_dir / "benchmark_stageii.mp4").write_bytes(b"mp4")
        return {
            "obj_path": str(output_dir / "benchmark_stageii.obj"),
            "pc2_path": str(output_dir / "benchmark_stageii.pc2"),
            "video_path": str(output_dir / "benchmark_stageii.mp4"),
        }

    fake_export_stageii_artifacts_module = type(
        "FakeExportStageIIArtifacts",
        (),
        {
            "export_stageii_artifacts": staticmethod(fake_export_stageii_artifacts),
        },
    )

    monkeypatch.setattr(
        stageii_benchmark,
        "_preview_render_model_path",
        lambda repo_root, *, gender: ROOT / "support_files/smplx/male/model.npz",
    )
    monkeypatch.setitem(sys.modules, "render_video", fake_render_video)
    monkeypatch.setitem(sys.modules, "export_stageii_artifacts", fake_export_stageii_artifacts_module)

    summary = stageii_benchmark._benchmark_artifact_bundle_export(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        baseline,
        repo_root=ROOT,
        warmup_runs=1,
        measured_runs=2,
    )

    assert summary["count"] == 2
    assert len(load_calls) == 1
    assert vertex_calls == [(ROOT / "support_data/tests/mosh_stageii.pkl", preloaded_model)]
    assert len(bundle_calls) == 3
    assert all(call[1] is preloaded_model for call in bundle_calls)
    assert all(call[2] is predecoded_vertices for call in bundle_calls)


def test_blocked_stages_use_preview_render_stack_and_support_files_assets(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    preview_model = repo_root / "support_files" / "smplx" / "male" / "model.npz"
    preview_model.parent.mkdir(parents=True)
    preview_model.write_bytes(b"npz")

    available_modules = {
        "taichi",
        "cv2",
        "human_body_prior.body_model.body_model",
    }

    monkeypatch.setattr(
        stageii_benchmark,
        "_safe_find_spec",
        lambda module_name: object() if module_name in available_modules else None,
    )

    blocked = stageii_benchmark._blocked_stages(repo_root)
    blocked_stages = [entry["stage"] for entry in blocked]
    blocked_reasons = [entry["reason"] for entry in blocked]

    assert "mp4_render" not in blocked_stages
    assert "mesh_export" not in blocked_stages
    assert not any("model.npz assets needed for mesh export" in reason for reason in blocked_reasons)


def test_blocked_stages_do_not_flag_mosh_head_loader_when_entrypoint_imports(monkeypatch):
    monkeypatch.setattr(
        stageii_benchmark,
        "_safe_find_spec",
        lambda module_name: None if module_name == "body_visualizer.mesh" else object(),
    )

    blocked = stageii_benchmark._blocked_stages(ROOT)
    blocked_stages = [entry["stage"] for entry in blocked]

    assert "mosh_head_loader" not in blocked_stages
