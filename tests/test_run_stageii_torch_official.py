import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_stageii_torch_official


def _required_official_runner_args(tmp_path, *, mocap_ext=".mcp", preset="real-mcp-baseline"):
    args = [
        "--mocap-fname",
        str(tmp_path / "input" / "wolf001" / f"capture{mocap_ext}"),
        "--support-base-dir",
        str(tmp_path / "support_files"),
        "--work-base-dir",
        str(tmp_path / "work"),
    ]
    if preset is not None:
        args.extend(["--preset", preset])
    return args


def _expected_real_mcp_prepare_cfg(tmp_path, *, mocap_fname=None, **overrides):
    return {
        **run_stageii_torch_official.REAL_MCP_BASELINE_PRESET,
        **overrides,
        "mocap.fname": str(mocap_fname or (tmp_path / "input" / "wolf001" / "capture.mcp")),
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }


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
        lean_benchmark,
    ):
        captured["benchmark_call"] = {
            "sample_path": sample_path,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "mesh_reference_path": mesh_reference_path,
            "mesh_support_base_dir": mesh_support_base_dir,
            "mesh_chunk_size": mesh_chunk_size,
            "mesh_chunk_overlap": mesh_chunk_overlap,
            "lean_benchmark": lean_benchmark,
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
        _required_official_runner_args(tmp_path)
        + [
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

    assert captured["prepare_cfg"] == _expected_real_mcp_prepare_cfg(
        tmp_path,
        **{
            "surface_model.gender": "male",
            "runtime.sequence_lr": "0.05",
        },
    )
    assert captured["run_cfg"].dirs.stageii_fname == str(stageii_path)
    assert captured["benchmark_call"] == {
        "sample_path": str(stageii_path),
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": str(tmp_path / "baseline_stageii.pkl"),
        "mesh_support_base_dir": str(tmp_path / "mesh_support"),
        "mesh_chunk_size": 32,
        "mesh_chunk_overlap": 4,
        "lean_benchmark": False,
    }
    assert captured["benchmark_output"][1] == str(benchmark_output)
    assert payload["stageii_path"] == str(stageii_path)
    assert payload["benchmark"]["artifact"]["report_path"] == str(benchmark_output)


def test_run_stageii_torch_official_main_writes_default_benchmark_report_next_to_stageii(
    tmp_path, monkeypatch
):
    stageii_path = tmp_path / "work" / "input" / "wolf001" / "manual_name_candidate_stageii.pkl"
    captured = {}

    def fake_prepare_cfg(**kwargs):
        captured["prepare_cfg"] = kwargs
        return SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))

    def fake_run_moshpp_once(cfg):
        Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.dirs.stageii_fname).write_bytes(b"stageii")

    def fake_run_public_stageii_benchmark(
        sample_path,
        *,
        warmup_runs,
        measured_runs,
        mesh_reference_path,
        mesh_support_base_dir,
        mesh_chunk_size,
        mesh_chunk_overlap,
        lean_benchmark,
    ):
        captured["benchmark_call"] = {
            "sample_path": sample_path,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "mesh_reference_path": mesh_reference_path,
            "mesh_support_base_dir": mesh_support_base_dir,
            "mesh_chunk_size": mesh_chunk_size,
            "mesh_chunk_overlap": mesh_chunk_overlap,
            "lean_benchmark": lean_benchmark,
        }
        return {"sample": {"path": str(sample_path)}, "artifact": {"report_path": None}}

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
        _required_official_runner_args(tmp_path)
        + [
            "--cfg",
            "mocap.basename=manual_name",
            "--output-suffix",
            "_candidate",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ]
    )

    expected_output = stageii_path.with_name("manual_name_candidate_benchmark.json")
    assert captured["benchmark_call"] == {
        "sample_path": str(stageii_path),
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": None,
        "mesh_support_base_dir": str(tmp_path / "support_files"),
        "mesh_chunk_size": None,
        "mesh_chunk_overlap": None,
        "lean_benchmark": False,
    }
    assert captured["benchmark_output"][1] == str(expected_output)
    assert payload["benchmark"]["artifact"]["report_path"] == str(expected_output)


def test_run_stageii_torch_official_main_plans_mesh_reference_from_output_suffix_without_second_prepare_cfg(
    tmp_path, monkeypatch
):
    captured = {"prepare_cfg_calls": []}

    def fake_prepare_cfg(**kwargs):
        captured["prepare_cfg_calls"].append(kwargs)
        basename = kwargs.get("mocap.basename", Path(kwargs["mocap.fname"]).stem)
        return SimpleNamespace(
            dirs=SimpleNamespace(
                stageii_fname=str(tmp_path / "work" / "input" / "wolf001" / f"{basename}_stageii.pkl")
            )
        )

    def fake_run_moshpp_once(cfg):
        stageii_path = Path(cfg.dirs.stageii_fname)
        stageii_path.parent.mkdir(parents=True, exist_ok=True)
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
        lean_benchmark,
    ):
        captured["benchmark_call"] = {
            "sample_path": sample_path,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "mesh_reference_path": mesh_reference_path,
            "mesh_support_base_dir": mesh_support_base_dir,
            "mesh_chunk_size": mesh_chunk_size,
            "mesh_chunk_overlap": mesh_chunk_overlap,
            "lean_benchmark": lean_benchmark,
        }
        return {"sample": {"path": str(sample_path)}, "quality": {"mesh_compare": None}}

    monkeypatch.setattr(run_stageii_torch_official, "MoSh", SimpleNamespace(prepare_cfg=fake_prepare_cfg))
    monkeypatch.setattr(run_stageii_torch_official, "run_moshpp_once", fake_run_moshpp_once)
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        fake_run_public_stageii_benchmark,
    )

    payload = run_stageii_torch_official.main(
        _required_official_runner_args(tmp_path)
        + [
            "--cfg",
            "mocap.basename=manual_name",
            "--output-suffix",
            "_candidate",
            "--mesh-reference-output-suffix",
            "_baseline",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ]
    )

    assert captured["prepare_cfg_calls"] == [
        _expected_real_mcp_prepare_cfg(
            tmp_path,
            **{
                "mocap.basename": "manual_name_candidate",
            },
        )
    ]
    assert captured["benchmark_call"] == {
        "sample_path": str(tmp_path / "work" / "input" / "wolf001" / "manual_name_candidate_stageii.pkl"),
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": str(
            tmp_path / "work" / "input" / "wolf001" / "manual_name_baseline_stageii.pkl"
        ),
        "mesh_support_base_dir": str(tmp_path / "support_files"),
        "mesh_chunk_size": None,
        "mesh_chunk_overlap": None,
        "lean_benchmark": False,
    }
    assert payload["stageii_path"] == str(
        tmp_path / "work" / "input" / "wolf001" / "manual_name_candidate_stageii.pkl"
    )


def test_run_stageii_torch_official_main_plans_mesh_reference_from_output_suffix_with_flat_mocap_path_overrides(
    tmp_path, monkeypatch
):
    captured = {"prepare_cfg_calls": []}

    def fake_prepare_cfg(**kwargs):
        captured["prepare_cfg_calls"].append(kwargs)
        basename = kwargs.get("mocap.basename", Path(kwargs["mocap.fname"]).stem)
        return SimpleNamespace(
            dirs=SimpleNamespace(
                stageii_fname=str(tmp_path / "work" / "demo_ds" / "demo_session" / f"{basename}_stageii.pkl")
            )
        )

    def fake_run_moshpp_once(cfg):
        stageii_path = Path(cfg.dirs.stageii_fname)
        stageii_path.parent.mkdir(parents=True, exist_ok=True)
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
        lean_benchmark,
    ):
        captured["benchmark_call"] = {
            "sample_path": sample_path,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "mesh_reference_path": mesh_reference_path,
            "mesh_support_base_dir": mesh_support_base_dir,
            "mesh_chunk_size": mesh_chunk_size,
            "mesh_chunk_overlap": mesh_chunk_overlap,
            "lean_benchmark": lean_benchmark,
        }
        return {"sample": {"path": str(sample_path)}, "quality": {"mesh_compare": None}}

    monkeypatch.setattr(run_stageii_torch_official, "MoSh", SimpleNamespace(prepare_cfg=fake_prepare_cfg))
    monkeypatch.setattr(run_stageii_torch_official, "run_moshpp_once", fake_run_moshpp_once)
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        fake_run_public_stageii_benchmark,
    )

    payload = run_stageii_torch_official.main(
        [
            "--mocap-fname",
            str(tmp_path / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--preset",
            "real-mcp-baseline",
            "--cfg",
            "mocap.ds_name=demo_ds",
            "--cfg",
            "mocap.session_name=demo_session",
            "--output-suffix",
            "_candidate",
            "--mesh-reference-output-suffix",
            "_baseline",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ]
    )

    assert captured["prepare_cfg_calls"] == [
        _expected_real_mcp_prepare_cfg(
            tmp_path,
            mocap_fname=tmp_path / "capture.mcp",
            **{
                "mocap.ds_name": "demo_ds",
                "mocap.session_name": "demo_session",
                "mocap.basename": "capture_candidate",
            },
        )
    ]
    assert captured["benchmark_call"] == {
        "sample_path": str(tmp_path / "work" / "demo_ds" / "demo_session" / "capture_candidate_stageii.pkl"),
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": str(
            tmp_path / "work" / "demo_ds" / "demo_session" / "capture_baseline_stageii.pkl"
        ),
        "mesh_support_base_dir": str(tmp_path / "support_files"),
        "mesh_chunk_size": None,
        "mesh_chunk_overlap": None,
        "lean_benchmark": False,
    }
    assert payload["stageii_path"] == str(
        tmp_path / "work" / "demo_ds" / "demo_session" / "capture_candidate_stageii.pkl"
    )


def test_run_stageii_torch_official_main_applies_segment_registry_overrides(tmp_path, monkeypatch):
    stageii_path = (
        tmp_path
        / "work"
        / "input"
        / "wolf001"
        / "4090-haonan-73_wolf001_stable_hold_300f_stageii.pkl"
    )
    captured = {}

    def fake_prepare_cfg(**kwargs):
        captured["prepare_cfg"] = kwargs
        return SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))

    def fake_run_moshpp_once(cfg):
        Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.dirs.stageii_fname).write_bytes(b"stageii")

    def fake_run_public_stageii_benchmark(
        sample_path,
        *,
        warmup_runs,
        measured_runs,
        mesh_reference_path,
        mesh_support_base_dir,
        mesh_chunk_size,
        mesh_chunk_overlap,
        lean_benchmark,
    ):
        captured["benchmark_call"] = {
            "sample_path": sample_path,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "mesh_reference_path": mesh_reference_path,
            "mesh_support_base_dir": mesh_support_base_dir,
            "mesh_chunk_size": mesh_chunk_size,
            "mesh_chunk_overlap": mesh_chunk_overlap,
            "lean_benchmark": lean_benchmark,
        }
        return {"sample": {"path": str(sample_path)}, "quality": {"mesh_compare": None}}

    monkeypatch.setattr(run_stageii_torch_official, "MoSh", SimpleNamespace(prepare_cfg=fake_prepare_cfg))
    monkeypatch.setattr(run_stageii_torch_official, "run_moshpp_once", fake_run_moshpp_once)
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        fake_run_public_stageii_benchmark,
    )

    payload = run_stageii_torch_official.main(
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "4090-haonan-73.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--preset",
            "real-mcp-baseline",
            "--segment-id",
            "wolf001-stable-hold-300f",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
        ]
    )

    assert captured["prepare_cfg"] == {
        **run_stageii_torch_official.REAL_MCP_BASELINE_PRESET,
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "4090-haonan-73.mcp"),
        "mocap.start_fidx": "12720",
        "mocap.end_fidx": "13020",
        "mocap.basename": "4090-haonan-73_wolf001_stable_hold_300f",
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }
    assert captured["benchmark_call"] == {
        "sample_path": str(stageii_path),
        "warmup_runs": 0,
        "measured_runs": 1,
        "mesh_reference_path": None,
        "mesh_support_base_dir": str(tmp_path / "support_files"),
        "mesh_chunk_size": None,
        "mesh_chunk_overlap": None,
        "lean_benchmark": False,
    }
    assert payload["stageii_path"] == str(stageii_path)


def test_run_stageii_torch_official_main_passes_lean_benchmark_flag(tmp_path, monkeypatch):
    stageii_path = tmp_path / "candidate_stageii.pkl"
    captured = {}

    def fake_prepare_cfg(**kwargs):
        return SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))

    def fake_run_moshpp_once(cfg):
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
        lean_benchmark,
    ):
        captured["benchmark_call"] = {
            "sample_path": sample_path,
            "lean_benchmark": lean_benchmark,
        }
        return {"sample": {"path": str(sample_path)}, "artifact": {"report_path": None}}

    monkeypatch.setattr(run_stageii_torch_official, "MoSh", SimpleNamespace(prepare_cfg=fake_prepare_cfg))
    monkeypatch.setattr(run_stageii_torch_official, "run_moshpp_once", fake_run_moshpp_once)
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        fake_run_public_stageii_benchmark,
    )

    run_stageii_torch_official.main(
        _required_official_runner_args(tmp_path)
        + [
            "--warmup-runs",
            "0",
            "--measured-runs",
            "1",
            "--lean-benchmark",
        ]
    )

    assert captured["benchmark_call"] == {
        "sample_path": str(stageii_path),
        "lean_benchmark": True,
    }


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


def test_run_stageii_torch_official_main_applies_translation_friendly_candidate_preset(
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
            "real-mcp-transvelo100-seedvelowindow",
            "--skip-benchmark",
        ]
    )

    assert captured["prepare_cfg"] == {
        "moshpp.optimize_fingers": "true",
        "runtime.sequence_chunk_size": "32",
        "runtime.sequence_chunk_overlap": "4",
        "runtime.sequence_seed_refine_iters": "5",
        "runtime.refine_lr": "0.05",
        "runtime.sequence_lr": "0.05",
        "runtime.sequence_max_iters": "30",
        "runtime.sequence_transl_velocity": "100",
        "runtime.sequence_boundary_transl_velocity_reference": "true",
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_applies_structure_only_chunk48_deltapose_preset(
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
            "real-mcp-chunk48ov8-deltapose4",
            "--skip-benchmark",
        ]
    )

    assert captured["prepare_cfg"] == {
        "moshpp.optimize_fingers": "true",
        "runtime.sequence_chunk_size": "48",
        "runtime.sequence_chunk_overlap": "8",
        "runtime.sequence_seed_refine_iters": "5",
        "runtime.refine_lr": "0.05",
        "runtime.sequence_lr": "0.05",
        "runtime.sequence_max_iters": "30",
        "runtime.sequence_delta_pose": "4",
        "runtime.sequence_chunk_stitch_mode": "adaptive_transl_jump_pose_guard",
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_applies_low_risk_translation_candidate_preset(
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
            "real-mcp-transvelo10-seedvelowindow",
            "--skip-benchmark",
        ]
    )

    assert captured["prepare_cfg"] == {
        "moshpp.optimize_fingers": "true",
        "runtime.sequence_chunk_size": "32",
        "runtime.sequence_chunk_overlap": "4",
        "runtime.sequence_seed_refine_iters": "5",
        "runtime.refine_lr": "0.05",
        "runtime.sequence_lr": "0.05",
        "runtime.sequence_max_iters": "30",
        "runtime.sequence_transl_velocity": "10",
        "runtime.sequence_boundary_transl_velocity_reference": "true",
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_applies_mid_risk_translation_candidate_preset(
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
            "real-mcp-transvelo32-seedvelowindow",
            "--skip-benchmark",
        ]
    )

    assert captured["prepare_cfg"] == {
        "moshpp.optimize_fingers": "true",
        "runtime.sequence_chunk_size": "32",
        "runtime.sequence_chunk_overlap": "4",
        "runtime.sequence_seed_refine_iters": "5",
        "runtime.refine_lr": "0.05",
        "runtime.sequence_lr": "0.05",
        "runtime.sequence_max_iters": "30",
        "runtime.sequence_transl_velocity": "32",
        "runtime.sequence_boundary_transl_velocity_reference": "true",
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_rejects_real_mcp_without_preset_or_corrected_baseline_cfgs(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path, preset=None) + ["--skip-benchmark"]
        )

    assert "real .mcp single-run entry requires a corrected baseline anchor" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_manual_frame_bounds_with_segment_id(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "4090-haonan-73.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--preset",
                "real-mcp-baseline",
                "--segment-id",
                "wolf001-stable-hold-300f",
                "--cfg",
                "mocap.start_fidx=0",
            ]
        )

    assert "--segment-id cannot be combined with --cfg mocap.start_fidx/mocap.end_fidx" in capsys.readouterr().err


def test_run_stageii_torch_official_main_allows_real_mcp_without_preset_when_corrected_baseline_cfgs_are_explicit(
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
        _required_official_runner_args(tmp_path, preset=None)
        + [
            "--cfg",
            "moshpp.optimize_fingers=true",
            "--cfg",
            "runtime.refine_lr=0.05",
            "--cfg",
            "runtime.sequence_lr=0.05",
            "--skip-benchmark",
        ]
    )

    assert captured["prepare_cfg"] == {
        "moshpp.optimize_fingers": "true",
        "runtime.refine_lr": "0.05",
        "runtime.sequence_lr": "0.05",
        "mocap.fname": str(tmp_path / "input" / "wolf001" / "capture.mcp"),
        "dirs.support_base_dir": str(tmp_path / "support_files"),
        "dirs.work_base_dir": str(tmp_path / "work"),
        "runtime.backend": "torch",
    }
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_appends_output_suffix_to_cfg_basename(tmp_path, monkeypatch):
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
        _required_official_runner_args(tmp_path)
        + [
            "--cfg",
            "mocap.basename=manual_name",
            "--output-suffix",
            "_candidate",
            "--skip-benchmark",
        ]
    )

    assert captured["prepare_cfg"] == _expected_real_mcp_prepare_cfg(
        tmp_path,
        **{
            "mocap.basename": "manual_name_candidate",
        },
    )
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_can_export_mesh_outputs(tmp_path, monkeypatch):
    stageii_path = tmp_path / "work" / "candidate_stageii.pkl"
    mesh_output_dir = tmp_path / "mesh_exports"
    mesh_support_dir = tmp_path / "mesh_support"
    resolved_model_path = mesh_support_dir / "smplx" / "male" / "model.npz"
    captured = {}

    def fake_prepare_cfg(**kwargs):
        return SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))

    def fake_run_moshpp_once(cfg):
        Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.dirs.stageii_fname).write_bytes(b"stageii")

    def fake_resolve_stageii_model_path(stageii_pkl, *, support_base_dir=None):
        captured["resolve_model"] = {
            "stageii_pkl": stageii_pkl,
            "support_base_dir": support_base_dir,
        }
        return str(resolved_model_path)

    def fake_export_stageii_meshes(
        input_pkl,
        model_path=None,
        *,
        model=None,
        vertices=None,
        obj_out=None,
        pc2_out=None,
    ):
        captured["export_meshes"] = {
            "input_pkl": input_pkl,
            "model_path": model_path,
            "model": model,
            "vertices": vertices,
            "obj_out": obj_out,
            "pc2_out": pc2_out,
        }
        return obj_out, pc2_out

    monkeypatch.setattr(run_stageii_torch_official, "MoSh", SimpleNamespace(prepare_cfg=fake_prepare_cfg))
    monkeypatch.setattr(run_stageii_torch_official, "run_moshpp_once", fake_run_moshpp_once)
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "resolve_stageii_model_path",
        fake_resolve_stageii_model_path,
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "export_stageii_meshes",
        fake_export_stageii_meshes,
    )

    payload = run_stageii_torch_official.main(
        _required_official_runner_args(tmp_path)
        + [
            "--skip-benchmark",
            "--export-mesh",
            "--mesh-output-dir",
            str(mesh_output_dir),
            "--mesh-support-base-dir",
            str(mesh_support_dir),
        ]
    )

    assert captured["resolve_model"] == {
        "stageii_pkl": str(stageii_path),
        "support_base_dir": str(mesh_support_dir),
    }
    assert captured["export_meshes"] == {
        "input_pkl": str(stageii_path),
        "model_path": str(resolved_model_path),
        "model": None,
        "vertices": None,
        "obj_out": str(mesh_output_dir / "candidate_stageii.obj"),
        "pc2_out": str(mesh_output_dir / "candidate_stageii.pc2"),
    }
    assert payload == {
        "benchmark": None,
        "mesh_export": {
            "obj_path": str(mesh_output_dir / "candidate_stageii.obj"),
            "pc2_path": str(mesh_output_dir / "candidate_stageii.pc2"),
        },
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_errors_when_mesh_export_model_path_cannot_be_resolved(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (
            Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True),
            Path(cfg.dirs.stageii_fname).write_bytes(b"stageii"),
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "resolve_stageii_model_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("missing cfg.surface_model.fname")),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--export-mesh",
            ]
        )

    assert "missing cfg.surface_model.fname" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_mesh_export_model_path_file_is_missing(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (
            Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True),
            Path(cfg.dirs.stageii_fname).write_bytes(b"stageii"),
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "resolve_stageii_model_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("model asset missing")),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--export-mesh",
            ]
        )

    assert "model asset missing" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_mesh_export_write_fails(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (
            Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True),
            Path(cfg.dirs.stageii_fname).write_bytes(b"stageii"),
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "resolve_stageii_model_path",
        lambda *args, **kwargs: str(tmp_path / "support_files" / "model.npz"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "export_stageii_meshes",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("mesh export write failed")),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--export-mesh",
            ]
        )

    assert "mesh export write failed" in capsys.readouterr().err


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
        _required_official_runner_args(tmp_path)
        + [
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


def test_run_stageii_torch_official_run_can_suppress_json_output(tmp_path, monkeypatch, capsys):
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

    payload = run_stageii_torch_official.run(
        _required_official_runner_args(tmp_path) + ["--skip-benchmark"],
        emit_json=False,
    )

    assert capsys.readouterr().out == ""
    assert payload == {
        "benchmark": None,
        "stageii_path": str(stageii_path),
    }


def test_run_stageii_torch_official_main_errors_when_official_run_does_not_produce_stageii(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: None,
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should not run when stageii output is missing"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(_required_official_runner_args(tmp_path))

    assert "run_moshpp_once did not produce the expected stageii file" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_prepare_cfg_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: (_ for _ in ()).throw(ValueError("invalid official cfg"))),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run when cfg preparation fails"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(_required_official_runner_args(tmp_path))

    assert "invalid official cfg" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_prepared_stageii_path_drifts_from_expected_contract(
    tmp_path, monkeypatch, capsys
):
    expected_stageii_path = tmp_path / "work" / "input" / "wolf001" / "capture_candidate_stageii.pkl"
    drifted_stageii_path = tmp_path / "work" / "input" / "wolf001" / "capture_drifted_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(drifted_stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run when prepared stageii path drifts from expectation"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--output-suffix",
                "_candidate",
                "--skip-benchmark",
                "--expected-stageii-path",
                str(expected_stageii_path),
            ]
        )

    assert "prepared stageii output path drifted from expected plan" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_benchmark_output_drifts_from_expected_contract(
    tmp_path, monkeypatch, capsys
):
    expected_benchmark_output = tmp_path / "benchmarks" / "capture_candidate_benchmark.json"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run when benchmark output drifts"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "write_benchmark_report",
        lambda *args, **kwargs: pytest.fail("write_benchmark_report should not run when benchmark output drifts"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--expected-benchmark-output",
                str(expected_benchmark_output),
            ]
        )

    assert "resolved benchmark output path drifted from expected plan" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_mesh_export_paths_drift_from_expected_contract(
    tmp_path, monkeypatch, capsys
):
    mesh_output_dir = tmp_path / "mesh_exports"
    mesh_support_dir = tmp_path / "mesh_support"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "resolve_stageii_model_path",
        lambda *args, **kwargs: pytest.fail("resolve_stageii_model_path should not run when mesh paths drift"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "export_stageii_meshes",
        lambda *args, **kwargs: pytest.fail("export_stageii_meshes should not run when mesh paths drift"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--export-mesh",
                "--mesh-output-dir",
                str(mesh_output_dir),
                "--mesh-support-base-dir",
                str(mesh_support_dir),
                "--expected-mesh-obj-path",
                str(mesh_output_dir / "unexpected.obj"),
                "--expected-mesh-pc2-path",
                str(mesh_output_dir / "capture_candidate_stageii.pc2"),
            ]
        )

    assert "resolved mesh export obj path drifted from expected plan" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_benchmark_writer_returns_drifted_report_path(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "input" / "wolf001" / "capture_candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (
            Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True),
            Path(cfg.dirs.stageii_fname).write_bytes(b"stageii"),
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: {"sample": {"path": str(stageii_path)}, "quality": {}},
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "write_benchmark_report",
        lambda report, output_path: {
            **report,
            "artifact": {"report_path": str(tmp_path / "drifted_benchmark.json")},
        },
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(_required_official_runner_args(tmp_path))

    assert "benchmark payload report_path drifted from requested output path" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_mesh_export_returns_drifted_paths(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "input" / "wolf001" / "capture_candidate_stageii.pkl"
    mesh_output_dir = tmp_path / "mesh_exports"
    mesh_support_dir = tmp_path / "mesh_support"
    resolved_model_path = mesh_support_dir / "smplx" / "male" / "model.npz"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (
            Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True),
            Path(cfg.dirs.stageii_fname).write_bytes(b"stageii"),
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "resolve_stageii_model_path",
        lambda *args, **kwargs: str(resolved_model_path),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "export_stageii_meshes",
        lambda *args, **kwargs: (
            str(mesh_output_dir / "drifted.obj"),
            str(mesh_output_dir / "capture_candidate_stageii.pc2"),
        ),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--export-mesh",
                "--mesh-output-dir",
                str(mesh_output_dir),
                "--mesh-support-base-dir",
                str(mesh_support_dir),
            ]
        )

    assert "mesh export payload obj_path drifted from requested output path" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_mesh_export_raises_import_error(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "input" / "wolf001" / "capture_candidate_stageii.pkl"
    mesh_output_dir = tmp_path / "mesh_exports"
    mesh_support_dir = tmp_path / "mesh_support"
    resolved_model_path = mesh_support_dir / "smplx" / "male" / "model.npz"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (
            Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True),
            Path(cfg.dirs.stageii_fname).write_bytes(b"stageii"),
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should be skipped"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "resolve_stageii_model_path",
        lambda *args, **kwargs: str(resolved_model_path),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "export_stageii_meshes",
        lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("missing mesh backend")),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--export-mesh",
                "--mesh-output-dir",
                str(mesh_output_dir),
                "--mesh-support-base-dir",
                str(mesh_support_dir),
            ]
        )

    assert "missing mesh backend" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_run_moshpp_once_raises_runtime_error(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (_ for _ in ()).throw(OSError("mocap load failed")),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(_required_official_runner_args(tmp_path))

    assert "mocap load failed" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_prepare_cfg_raises_key_error(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: (_ for _ in ()).throw(KeyError("missing cfg.surface_model.fname"))
        ),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(_required_official_runner_args(tmp_path))

    assert "missing cfg.surface_model.fname" in capsys.readouterr().err


def test_run_stageii_torch_official_main_errors_when_run_moshpp_once_raises_import_error(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (_ for _ in ()).throw(ModuleNotFoundError("missing torch runtime")),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(_required_official_runner_args(tmp_path))

    assert "missing torch runtime" in capsys.readouterr().err


def test_run_stageii_torch_official_parser_rejects_explicit_mesh_reference_and_output_suffix_together():
    parser = run_stageii_torch_official.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--mocap-fname",
                "capture.mcp",
                "--support-base-dir",
                "support_files",
                "--work-base-dir",
                "work",
                "--mesh-reference",
                "baseline_stageii.pkl",
                "--mesh-reference-output-suffix",
                "_baseline",
            ]
        )


def test_run_stageii_torch_official_main_rejects_mesh_reference_output_suffix_with_explicit_stageii_override(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--cfg",
                f"dirs.stageii_fname={tmp_path / 'manual_candidate_stageii.pkl'}",
                "--mesh-reference-output-suffix",
                "_baseline",
            ]
        )


def test_run_stageii_torch_official_main_preflights_explicit_mesh_reference_that_matches_planned_stageii(
    tmp_path, monkeypatch, capsys
):
    planned_stageii_path = tmp_path / "work" / "input" / "wolf001" / "capture_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should not run for self mesh compare"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--mesh-reference",
                str(planned_stageii_path),
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
            ]
        )

    assert "mesh reference resolves to the current stageii output" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_explicit_mesh_reference_that_matches_output_stageii(
    tmp_path, monkeypatch, capsys
):
    stageii_path = tmp_path / "work" / "candidate_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(
            prepare_cfg=lambda **kwargs: SimpleNamespace(dirs=SimpleNamespace(stageii_fname=str(stageii_path)))
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (
            Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True),
            Path(cfg.dirs.stageii_fname).write_bytes(b"stageii"),
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should not run for self mesh compare"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--mesh-reference",
                str(stageii_path),
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
            ]
        )

    assert "mesh reference resolves to the current stageii output" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_mesh_reference_output_suffix_that_matches_output_stageii(
    tmp_path, monkeypatch, capsys
):
    captured = {"prepare_cfg_calls": []}

    def fake_prepare_cfg(**kwargs):
        captured["prepare_cfg_calls"].append(kwargs)
        basename = kwargs.get("mocap.basename", Path(kwargs["mocap.fname"]).stem)
        return SimpleNamespace(
            dirs=SimpleNamespace(
                stageii_fname=str(tmp_path / "work" / "input" / "wolf001" / f"{basename}_stageii.pkl")
            )
        )

    monkeypatch.setattr(run_stageii_torch_official, "MoSh", SimpleNamespace(prepare_cfg=fake_prepare_cfg))
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: (
            Path(cfg.dirs.stageii_fname).parent.mkdir(parents=True, exist_ok=True),
            Path(cfg.dirs.stageii_fname).write_bytes(b"stageii"),
        ),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark should not run for self mesh compare"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--cfg",
                "mocap.basename=manual_name",
                "--output-suffix",
                "_candidate",
                "--mesh-reference-output-suffix",
                "_candidate",
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
            ]
        )

    assert captured["prepare_cfg_calls"] == []
    assert "mesh reference resolves to the current stageii output" in capsys.readouterr().err


def test_run_stageii_torch_official_main_preflights_benchmark_output_that_matches_planned_stageii(
    tmp_path, monkeypatch, capsys
):
    planned_stageii_path = tmp_path / "work" / "input" / "wolf001" / "capture_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "write_benchmark_report",
        lambda *args, **kwargs: pytest.fail("write_benchmark_report should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--benchmark-output",
                str(planned_stageii_path),
            ]
        )

    assert "benchmark output resolves to current stageii output path" in capsys.readouterr().err


def test_run_stageii_torch_official_main_preflights_benchmark_output_that_matches_planned_mesh_export(
    tmp_path, monkeypatch, capsys
):
    mesh_output_dir = tmp_path / "mesh_exports"
    planned_mesh_obj_path = mesh_output_dir / "capture_stageii.obj"

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "write_benchmark_report",
        lambda *args, **kwargs: pytest.fail("write_benchmark_report should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--export-mesh",
                "--mesh-output-dir",
                str(mesh_output_dir),
                "--benchmark-output",
                str(planned_mesh_obj_path),
            ]
        )

    assert "benchmark output resolves to mesh export obj path" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("mesh_reference_name", "expected_message"),
    [
        ("capture_stageii.obj", "mesh reference resolves to mesh export obj path"),
        ("capture_stageii.pc2", "mesh reference resolves to mesh export pc2 path"),
    ],
    ids=["mesh_reference_matches_obj", "mesh_reference_matches_pc2"],
)
def test_run_stageii_torch_official_main_preflights_mesh_reference_that_matches_planned_mesh_export(
    tmp_path, monkeypatch, capsys, mesh_reference_name, expected_message
):
    mesh_output_dir = tmp_path / "mesh_exports"
    mesh_reference_path = mesh_output_dir / mesh_reference_name

    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--export-mesh",
                "--mesh-output-dir",
                str(mesh_output_dir),
                "--mesh-reference",
                str(mesh_reference_path),
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
            ]
        )

    assert expected_message in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_mesh_chunk_overlap_without_mesh_chunk_size(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_public_stageii_benchmark",
        lambda *args, **kwargs: pytest.fail("benchmark helper should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--mesh-reference",
                str(tmp_path / "baseline.pc2"),
                "--mesh-chunk-overlap",
                "4",
                "--warmup-runs",
                "0",
                "--measured-runs",
                "1",
            ]
        )

    assert "--mesh-chunk-overlap requires --mesh-chunk-size" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_mesh_output_dir_without_export_mesh(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--mesh-output-dir",
                str(tmp_path / "mesh_exports"),
            ]
        )

    assert "--mesh-output-dir requires --export-mesh" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_benchmark_output_when_skip_benchmark(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--benchmark-output",
                str(tmp_path / "candidate_benchmark.json"),
            ]
        )

    assert "--benchmark-output requires benchmark to be enabled" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_expected_benchmark_output_when_skip_benchmark(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--expected-benchmark-output",
                str(tmp_path / "candidate_benchmark.json"),
            ]
        )

    assert "--expected-benchmark-output requires benchmark to be enabled" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_mesh_reference_when_skip_benchmark(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--mesh-reference",
                str(tmp_path / "baseline_stageii.pkl"),
            ]
        )

    assert "--mesh-reference requires benchmark to be enabled" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_expected_mesh_paths_without_export_mesh(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--expected-mesh-obj-path",
                str(tmp_path / "mesh_exports" / "capture_stageii.obj"),
                "--expected-mesh-pc2-path",
                str(tmp_path / "mesh_exports" / "capture_stageii.pc2"),
            ]
        )

    assert "--expected-mesh-obj-path/--expected-mesh-pc2-path require --export-mesh" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_partial_expected_mesh_contract(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--skip-benchmark",
                "--export-mesh",
                "--expected-mesh-obj-path",
                str(tmp_path / "mesh_exports" / "capture_stageii.obj"),
            ]
        )

    assert "--expected-mesh-obj-path and --expected-mesh-pc2-path must be provided together" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_mesh_chunk_size_without_mesh_reference(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--mesh-chunk-size",
                "32",
            ]
        )

    assert "--mesh-chunk-size requires --mesh-reference or --mesh-reference-output-suffix" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_non_positive_measured_runs(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--measured-runs",
                "0",
            ]
        )

    assert "--measured-runs must be > 0" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_negative_mesh_chunk_overlap(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--mesh-reference",
                str(tmp_path / "baseline.pc2"),
                "--mesh-chunk-size",
                "32",
                "--mesh-chunk-overlap",
                "-1",
            ]
        )

    assert "--mesh-chunk-overlap must be >= 0" in capsys.readouterr().err


def test_run_stageii_torch_official_main_rejects_mesh_support_base_dir_without_export_or_mesh_reference(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_official,
        "MoSh",
        SimpleNamespace(prepare_cfg=lambda **kwargs: pytest.fail("prepare_cfg should not run")),
    )
    monkeypatch.setattr(
        run_stageii_torch_official,
        "run_moshpp_once",
        lambda cfg: pytest.fail("run_moshpp_once should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_official.main(
            _required_official_runner_args(tmp_path)
            + [
                "--mesh-support-base-dir",
                str(tmp_path / "mesh_support"),
            ]
        )

    assert "--mesh-support-base-dir requires --export-mesh or --mesh-reference/--mesh-reference-output-suffix" in capsys.readouterr().err
