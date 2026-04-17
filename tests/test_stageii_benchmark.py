import json
import sys
from pathlib import Path

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

    report = run_public_stageii_benchmark(
        ROOT / "support_data/tests/mosh_stageii.pkl",
        warmup_runs=0,
        measured_runs=1,
    )

    assert report["speed"]["preview_vertex_decode_ms"] == preview_metric


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
    assert "mesh_export" in blocked_stages
    assert not any("model.npz assets needed for mesh export" in reason for reason in blocked_reasons)
