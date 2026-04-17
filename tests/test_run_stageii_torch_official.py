import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_stageii_torch_official


def test_run_stageii_torch_official_main_builds_cfg_and_wires_benchmark(tmp_path, monkeypatch):
    stageii_path = tmp_path / "candidate_stageii.pkl"
    benchmark_output = tmp_path / "candidate_benchmark.json"
    captured = {}

    def fake_prepare_cfg(**kwargs):
        captured["prepare_cfg"] = kwargs
        return SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))

    def fake_run_moshpp_once(cfg):
        captured["run_cfg"] = cfg
        stageii_path.write_bytes(b"stageii")

    def fake_run_public_stageii_benchmark(
        sample_path,
        *,
        warmup_runs,
        measured_runs,
        mesh_reference_path,
        mesh_support_base_dir,
        mesh_chunk_size,
        mesh_chunk_overlap,
    ):
        captured["benchmark_call"] = {
            "sample_path": sample_path,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "mesh_reference_path": mesh_reference_path,
            "mesh_support_base_dir": mesh_support_base_dir,
            "mesh_chunk_size": mesh_chunk_size,
            "mesh_chunk_overlap": mesh_chunk_overlap,
        }
        return {"sample": {"path": str(sample_path)}, "quality": {"mesh_compare": None}}

    def fake_write_benchmark_report(report, output_path):
        captured["benchmark_output"] = (report, output_path)
        return {
            **report,
            "artifact": {"report_path": str(output_path)},
        }

    monkeypatch.setattr(run_stageii_torch_official, "MoSh", SimpleNamespace(prepare_cfg=fake_prepare_cfg))
    monkeypatch.setattr(run_stageii_torch_official, "run_moshpp_once", fake_run_moshpp_once)
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        fake_run_public_stageii_benchmark,
    )
    monkeypatch.setattr(run_stageii_torch_official, "write_benchmark_report", fake_write_benchmark_report)

    payload = run_stageii_torch_official.main(
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--cfg",
            "surface_model.gender=male",
            "--cfg",
            "runtime.sequence_lr=0.05",
            "--benchmark-output",
            str(benchmark_output),
            "--mesh-reference",
            str(tmp_path / "baseline_stageii.pkl"),
            "--mesh-support-base-dir",
            str(tmp_path / "mesh_support"),
            "--mesh-chunk-size",
            "32",
            "--mesh-chunk-overlap",
            "4",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ]
    )

    assert captured["prepare_cfg"] == {
        "surface_model.gender": "male",
        "runtime.sequence_lr": "0.05",
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }
    assert captured["run_cfg"].dirs.stageii_fname == str(stageii_path)
    assert captured["benchmark_call"] == {
        "sample_path": str(stageii_path),
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": str(tmp_path / "baseline_stageii.pkl"),
        "mesh_support_base_dir": str(tmp_path / "mesh_support"),
        "mesh_chunk_size": 32,
        "mesh_chunk_overlap": 4,
    }
    assert captured["benchmark_output"][1] == str(benchmark_output)
    assert payload["stageii_path"] == str(stageii_path)
    assert payload["benchmark"]["artifact"]["report_path"] == str(benchmark_output)


def test_run_stageii_torch_official_main_applies_real_mcp_baseline_preset_before_explicit_cfg(
    tmp_path, monkeypatch
):
    stageii_path = tmp_path / "candidate_stageii.pkl"
    captured = {}

    def fake_prepare_cfg(**kwargs):
        captured["prepare_cfg"] = kwargs
        return SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))

    monkeypatch.setattr(run_stageii_torch_official, "MoSh", SimpleNamespace(prepare_cfg=fake_prepare_cfg))
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: stageii_path.write_bytes(b"stageii"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )

    payload = run_stageii_torch_official.main(
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--preset",
            "real-mcp-baseline",
            "--cfg",
            "moshpp.optimize_fingers=false",
            "--cfg",
            "runtime.sequence_lr=0.07",
            "--skip-benchmark",
        ]
    )

    assert captured["prepare_cfg"] == {
        "moshpp.optimize_fingers": "false",
        "runtime.sequence_chunk_size": "32",
        "runtime.sequence_chunk_overlap": "4",
        "runtime.sequence_seed_refine_iters": "5",
        "runtime.refine_lr": "0.05",
        "runtime.sequence_lr": "0.07",
        "runtime.sequence_max_iters": "30",
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_can_skip_benchmark(tmp_path, monkeypatch, capsys):
    stageii_path = tmp_path / "candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: stageii_path.write_bytes(b"stageii"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )

    payload = run_stageii_torch_official.main(
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--skip-benchmark",
            "--cfg",
            "surface_model.gender=male",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert payload == printed
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }
