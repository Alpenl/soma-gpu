import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
