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
    assert report["speed"]["preview_vertex_decode_ms"] is None
    assert report["speed"]["mesh_export_ms"] is None
    assert report["speed"]["mp4_render_ms"] is None
    assert report["speed"]["artifact_bundle_export_ms"] is None
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

    assert report["quality"] == {**quality_summary, "mesh_compare": None}


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
    assert captured == {
        "sample_path": ROOT / "support_data/tests/mosh_stageii.pkl",
        "reference_path": "/tmp/baseline_stageii.pkl",
        "support_base_dir": "/tmp/support_files",
        "chunk_size": 32,
        "chunk_overlap": 4,
    }


def test_benchmark_stageii_public_main_passes_optional_mesh_reference_args(monkeypatch, capsys):
    captured = {}

    monkeypatch.setattr(
        benchmark_stageii_public,
        "run_public_stageii_benchmark",
        lambda sample_path, *, warmup_runs, measured_runs, mesh_reference_path, mesh_support_base_dir, mesh_chunk_size, mesh_chunk_overlap: (
            captured.update(
                {
                    "sample_path": sample_path,
                    "warmup_runs": warmup_runs,
                    "measured_runs": measured_runs,
                    "mesh_reference_path": mesh_reference_path,
                    "mesh_support_base_dir": mesh_support_base_dir,
                    "mesh_chunk_size": mesh_chunk_size,
                    "mesh_chunk_overlap": mesh_chunk_overlap,
                }
            )
            or {
                "sample": {"path": sample_path},
                "quality": {"mesh_compare": None},
                "artifact": {"report_path": None},
            }
        ),
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
    }
    output = json.loads(capsys.readouterr().out)
    assert output["sample"]["path"] == "candidate_stageii.pkl"


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
    assert quality["trans_jitter_l2"]["max"] == pytest.approx(7.0)
    assert quality["pose_jitter_l2"]["max"] == pytest.approx(7.0)
    assert quality["chunk_seam_transl_jump_l2"]["count"] == 1
    assert quality["chunk_seam_transl_jump_l2"]["mean"] == pytest.approx(8.0)
    assert quality["chunk_seam_pose_jump_l2"]["mean"] == pytest.approx(8.0)


def test_summarize_stageii_quality_reads_legacy_marker_residual_from_public_sample():
    sample_path = ROOT / "support_data/tests/mosh_stageii.pkl"
    quality = stageii_benchmark._summarize_stageii_quality(
        sample_path,
        stageii_benchmark.normalize_stageii_sample(sample_path),
    )

    assert quality["marker_residual_l2"]["count"] > 0
    assert quality["marker_residual_l2"]["mean"] >= 0.0
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
