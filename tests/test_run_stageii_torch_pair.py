import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_stageii_torch_pair


def test_run_stageii_torch_pair_main_runs_baseline_then_candidate_with_shared_and_side_cfgs(
    tmp_path, monkeypatch, capsys
):
    captured = {"calls": []}

    def fake_run(argv, *, emit_json):
        captured["calls"].append((list(argv), emit_json))
        preset = argv[argv.index("--preset") + 1]
        suffix = argv[argv.index("--output-suffix") + 1]
        return {
            "benchmark": None if "--skip-benchmark" in argv else {"artifact": {"report_path": str(tmp_path / f"{suffix}.json")}},
            "stageii_path": str(tmp_path / f"{preset}{suffix}_stageii.pkl"),
        }

    monkeypatch.setattr(run_stageii_torch_pair.run_stageii_torch_official, "run", fake_run)

    payload = run_stageii_torch_pair.main(
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--cfg",
            "surface_model.gender=male",
            "--baseline-cfg",
            "moshpp.optimize_fingers=true",
            "--candidate-cfg",
            "runtime.sequence_transl_velocity=120",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert payload == printed
    assert captured["calls"] == [
        (
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--preset",
                "real-mcp-baseline",
                "--output-suffix",
                "_baseline",
                "--cfg",
                "surface_model.gender=male",
                "--cfg",
                "moshpp.optimize_fingers=true",
                "--skip-benchmark",
            ],
            False,
        ),
        (
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--preset",
                "real-mcp-transvelo100-seedvelowindow",
                "--output-suffix",
                "_candidate",
                "--cfg",
                "surface_model.gender=male",
                "--cfg",
                "runtime.sequence_transl_velocity=120",
                "--mesh-reference",
                str(tmp_path / "real-mcp-baseline_baseline_stageii.pkl"),
            ],
            False,
        ),
    ]
    assert payload == {
        "baseline": {
            "benchmark": None,
            "stageii_path": str(tmp_path / "real-mcp-baseline_baseline_stageii.pkl"),
        },
        "candidate": {
            "benchmark": {"artifact": {"report_path": str(tmp_path / "_candidate.json")}},
            "stageii_path": str(
                tmp_path / "real-mcp-transvelo100-seedvelowindow_candidate_stageii.pkl"
            ),
        },
    }


def test_run_stageii_torch_pair_main_passes_returned_baseline_stageii_path_to_candidate(
    tmp_path, monkeypatch
):
    captured = {"calls": []}
    baseline_stageii = tmp_path / "manual_baseline_stageii.pkl"
    candidate_stageii = tmp_path / "manual_candidate_stageii.pkl"

    def fake_run(argv, *, emit_json):
        captured["calls"].append((list(argv), emit_json))
        preset = argv[argv.index("--preset") + 1]
        if preset == "real-mcp-baseline":
            return {"benchmark": None, "stageii_path": str(baseline_stageii)}
        return {
            "benchmark": {"artifact": {"report_path": str(tmp_path / "candidate_benchmark.json")}},
            "stageii_path": str(candidate_stageii),
        }

    monkeypatch.setattr(run_stageii_torch_pair.run_stageii_torch_official, "run", fake_run)

    payload = run_stageii_torch_pair.main(
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--baseline-cfg",
            "mocap.basename=manual_baseline",
            "--candidate-cfg",
            "mocap.basename=manual_candidate",
        ]
    )

    assert captured["calls"][0] == (
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--preset",
            "real-mcp-baseline",
            "--output-suffix",
            "_baseline",
            "--cfg",
            "mocap.basename=manual_baseline",
            "--skip-benchmark",
        ],
        False,
    )
    assert captured["calls"][1] == (
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--preset",
            "real-mcp-transvelo100-seedvelowindow",
            "--output-suffix",
            "_candidate",
            "--cfg",
            "mocap.basename=manual_candidate",
            "--mesh-reference",
            str(baseline_stageii),
        ],
        False,
    )
    assert payload == {
        "baseline": {
            "benchmark": None,
            "stageii_path": str(baseline_stageii),
        },
        "candidate": {
            "benchmark": {"artifact": {"report_path": str(tmp_path / "candidate_benchmark.json")}},
            "stageii_path": str(candidate_stageii),
        },
    }


def test_run_stageii_torch_pair_main_can_request_baseline_benchmark_output(tmp_path, monkeypatch):
    captured = {"calls": []}

    def fake_run(argv, *, emit_json):
        captured["calls"].append((list(argv), emit_json))
        return {"benchmark": {"artifact": {"report_path": "report.json"}}, "stageii_path": "stageii.pkl"}

    monkeypatch.setattr(run_stageii_torch_pair.run_stageii_torch_official, "run", fake_run)

    payload = run_stageii_torch_pair.main(
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--baseline-benchmark-output",
            str(tmp_path / "baseline_benchmark.json"),
            "--candidate-benchmark-output",
            str(tmp_path / "candidate_benchmark.json"),
        ]
    )

    assert captured["calls"][0] == (
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--preset",
            "real-mcp-baseline",
            "--output-suffix",
            "_baseline",
            "--benchmark-output",
            str(tmp_path / "baseline_benchmark.json"),
        ],
        False,
    )
    assert captured["calls"][1] == (
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--preset",
            "real-mcp-transvelo100-seedvelowindow",
            "--output-suffix",
            "_candidate",
            "--benchmark-output",
            str(tmp_path / "candidate_benchmark.json"),
            "--mesh-reference",
            "stageii.pkl",
        ],
        False,
    )
    assert payload["baseline"]["benchmark"]["artifact"]["report_path"] == "report.json"


def test_run_stageii_torch_pair_main_forwards_mesh_export_flags_to_both_runs(tmp_path, monkeypatch):
    captured = {"calls": []}
    mesh_output_dir = tmp_path / "mesh_exports"
    mesh_support_dir = tmp_path / "mesh_support"

    def fake_run(argv, *, emit_json):
        captured["calls"].append((list(argv), emit_json))
        preset = argv[argv.index("--preset") + 1]
        suffix = argv[argv.index("--output-suffix") + 1]
        return {
            "benchmark": None if "--skip-benchmark" in argv else {"artifact": {"report_path": "candidate.json"}},
            "mesh_export": {
                "obj_path": str(mesh_output_dir / f"{preset}{suffix}.obj"),
                "pc2_path": str(mesh_output_dir / f"{preset}{suffix}.pc2"),
            },
            "stageii_path": str(tmp_path / f"{preset}{suffix}_stageii.pkl"),
        }

    monkeypatch.setattr(run_stageii_torch_pair.run_stageii_torch_official, "run", fake_run)

    payload = run_stageii_torch_pair.main(
        [
            "--mocap-fname",
            str(tmp_path / "input" / "wolf001" / "capture.mcp"),
            "--support-base-dir",
            str(tmp_path / "support_files"),
            "--work-base-dir",
            str(tmp_path / "work"),
            "--export-mesh",
            "--mesh-output-dir",
            str(mesh_output_dir),
            "--mesh-support-base-dir",
            str(mesh_support_dir),
        ]
    )

    assert captured["calls"] == [
        (
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--preset",
                "real-mcp-baseline",
                "--output-suffix",
                "_baseline",
                "--export-mesh",
                "--mesh-output-dir",
                str(mesh_output_dir),
                "--mesh-support-base-dir",
                str(mesh_support_dir),
                "--skip-benchmark",
            ],
            False,
        ),
        (
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--preset",
                "real-mcp-transvelo100-seedvelowindow",
                "--output-suffix",
                "_candidate",
                "--export-mesh",
                "--mesh-output-dir",
                str(mesh_output_dir),
                "--mesh-support-base-dir",
                str(mesh_support_dir),
                "--mesh-reference",
                str(tmp_path / "real-mcp-baseline_baseline_stageii.pkl"),
            ],
            False,
        ),
    ]
    assert payload["baseline"]["mesh_export"] == {
        "obj_path": str(mesh_output_dir / "real-mcp-baseline_baseline.obj"),
        "pc2_path": str(mesh_output_dir / "real-mcp-baseline_baseline.pc2"),
    }
    assert payload["candidate"]["mesh_export"] == {
        "obj_path": str(mesh_output_dir / "real-mcp-transvelo100-seedvelowindow_candidate.obj"),
        "pc2_path": str(mesh_output_dir / "real-mcp-transvelo100-seedvelowindow_candidate.pc2"),
    }


def test_run_stageii_torch_pair_main_rejects_matching_output_suffixes():
    with pytest.raises(SystemExit):
        run_stageii_torch_pair.main(
            [
                "--mocap-fname",
                "capture.mcp",
                "--support-base-dir",
                "support_files",
                "--work-base-dir",
                "work",
                "--baseline-output-suffix",
                "_same",
                "--candidate-output-suffix",
                "_same",
            ]
        )


def test_run_stageii_torch_pair_main_rejects_mesh_output_dir_without_export_mesh(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        run_stageii_torch_pair.run_stageii_torch_official,
        "run",
        lambda *args, **kwargs: pytest.fail("underlying runner should not run"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_pair.main(
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


def test_run_stageii_torch_pair_main_errors_when_baseline_runner_omits_stageii_path(
    tmp_path, monkeypatch, capsys
):
    def fake_run(argv, *, emit_json):
        preset = argv[argv.index("--preset") + 1]
        if preset == "real-mcp-baseline":
            return {"benchmark": None}
        pytest.fail("candidate runner should not run when baseline payload is incomplete")

    monkeypatch.setattr(run_stageii_torch_pair.run_stageii_torch_official, "run", fake_run)

    with pytest.raises(SystemExit):
        run_stageii_torch_pair.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
            ]
        )

    assert "baseline runner did not return stageii_path" in capsys.readouterr().err


def test_run_stageii_torch_pair_main_errors_when_candidate_runner_omits_stageii_path(
    tmp_path, monkeypatch, capsys
):
    def fake_run(argv, *, emit_json):
        preset = argv[argv.index("--preset") + 1]
        if preset == "real-mcp-baseline":
            return {"benchmark": None, "stageii_path": str(tmp_path / "baseline_stageii.pkl")}
        return {"benchmark": {"artifact": {"report_path": "candidate.json"}}}

    monkeypatch.setattr(run_stageii_torch_pair.run_stageii_torch_official, "run", fake_run)

    with pytest.raises(SystemExit):
        run_stageii_torch_pair.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
            ]
        )

    assert "candidate runner did not return stageii_path" in capsys.readouterr().err


def test_run_stageii_torch_pair_main_errors_when_underlying_runner_raises_runtime_error(
    tmp_path, monkeypatch, capsys
):
    def fake_run(argv, *, emit_json):
        preset = argv[argv.index("--preset") + 1]
        if preset == "real-mcp-baseline":
            return {"benchmark": None, "stageii_path": str(tmp_path / "baseline_stageii.pkl")}
        raise FileNotFoundError("candidate stageii missing")

    monkeypatch.setattr(run_stageii_torch_pair.run_stageii_torch_official, "run", fake_run)

    with pytest.raises(SystemExit):
        run_stageii_torch_pair.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
            ]
        )

    assert "candidate stageii missing" in capsys.readouterr().err


def test_run_stageii_torch_pair_main_rejects_colliding_stageii_outputs_from_basename_overrides(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_pair.run_stageii_torch_official,
        "run",
        lambda *args, **kwargs: pytest.fail("underlying runner should not run when stageii outputs collide"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_pair.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--baseline-cfg",
                "mocap.basename=shared",
                "--baseline-output-suffix",
                "_stageii",
                "--candidate-cfg",
                "mocap.basename=shared_stageii",
                "--candidate-output-suffix",
                "",
            ]
        )

    assert (
        "baseline and candidate resolve to the same stageii output path" in capsys.readouterr().err
    )


def test_run_stageii_torch_pair_main_rejects_explicit_colliding_stageii_output_paths(
    tmp_path, monkeypatch, capsys
):
    shared_stageii = tmp_path / "shared_stageii.pkl"

    monkeypatch.setattr(
        run_stageii_torch_pair.run_stageii_torch_official,
        "run",
        lambda *args, **kwargs: pytest.fail("underlying runner should not run when stageii outputs collide"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_pair.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--baseline-cfg",
                f"dirs.stageii_fname={shared_stageii}",
                "--candidate-cfg",
                f"dirs.stageii_fname={shared_stageii}",
            ]
        )

    assert (
        "baseline and candidate resolve to the same stageii output path" in capsys.readouterr().err
    )


def test_run_stageii_torch_pair_main_rejects_colliding_mesh_outputs_under_shared_mesh_output_dir(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        run_stageii_torch_pair.run_stageii_torch_official,
        "run",
        lambda *args, **kwargs: pytest.fail("underlying runner should not run when mesh outputs collide"),
    )

    with pytest.raises(SystemExit):
        run_stageii_torch_pair.main(
            [
                "--mocap-fname",
                str(tmp_path / "input" / "wolf001" / "capture.mcp"),
                "--support-base-dir",
                str(tmp_path / "support_files"),
                "--work-base-dir",
                str(tmp_path / "work"),
                "--export-mesh",
                "--mesh-output-dir",
                str(tmp_path / "mesh_exports"),
                "--baseline-cfg",
                f"dirs.stageii_fname={tmp_path / 'baseline' / 'shared_stageii.pkl'}",
                "--candidate-cfg",
                f"dirs.stageii_fname={tmp_path / 'candidate' / 'shared_stageii.pkl'}",
            ]
        )

    assert "baseline and candidate resolve to the same mesh export output path" in capsys.readouterr().err
