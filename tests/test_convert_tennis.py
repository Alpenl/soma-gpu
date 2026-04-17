import pickle
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import convert_tennis


def _install_fake_run_soma_module(monkeypatch, calls):
    run_module = types.ModuleType("soma.tools.run_soma_multiple")

    def fake_run_soma_on_multiple_settings(**kwargs):
        calls.append(kwargs)

    run_module.run_soma_on_multiple_settings = fake_run_soma_on_multiple_settings

    tools_module = types.ModuleType("soma.tools")
    tools_module.run_soma_multiple = run_module

    soma_module = types.ModuleType("soma")
    soma_module.tools = tools_module

    monkeypatch.setitem(sys.modules, "soma", soma_module)
    monkeypatch.setitem(sys.modules, "soma.tools", tools_module)
    monkeypatch.setitem(sys.modules, "soma.tools.run_soma_multiple", run_module)


def _install_fake_export_module(monkeypatch, calls):
    export_module = types.ModuleType("export_stageii_artifacts")

    def fake_export_stageii_artifacts(**kwargs):
        calls.append(kwargs)
        return {
            "obj_path": str(Path(kwargs["input_pkl"]).with_suffix(".obj")),
            "pc2_path": str(Path(kwargs["input_pkl"]).with_suffix(".pc2")),
            "video_path": str(Path(kwargs["input_pkl"]).with_suffix(".mp4")),
        }

    export_module.export_stageii_artifacts = fake_export_stageii_artifacts
    monkeypatch.setitem(sys.modules, "export_stageii_artifacts", export_module)


def _write_stageii_pickle(path, *, model_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "fname": str(model_path),
                        }
                    }
                }
            }
        )
    )


def test_convert_tennis_exports_matching_stageii_artifacts_after_mosh(monkeypatch, tmp_path):
    dataset = "demo_ds"
    work_dir = tmp_path / "work"
    mocap_base_dir = tmp_path / "mocap"
    support_dir = tmp_path / "support"
    model_path = support_dir / "smplx" / "male" / "model.npz"

    export_calls = []
    run_calls = []
    _install_fake_run_soma_module(monkeypatch, run_calls)
    _install_fake_export_module(monkeypatch, export_calls)

    _write_stageii_pickle(work_dir / dataset / "subject01" / "swing_stageii.pkl", model_path=model_path)
    _write_stageii_pickle(work_dir / dataset / "subject01" / "serve_stageii.pkl", model_path=model_path)

    convert_tennis.main(
        [
            "--dataset",
            dataset,
            "--mocap-base-dir",
            str(mocap_base_dir),
            "--soma-work-base-dir",
            str(work_dir),
            "--support-base-dir",
            str(support_dir),
            "--export-artifacts",
            "--fname-filter",
            "swing",
            "--export-fps",
            "15",
            "--export-width",
            "160",
            "--export-height",
            "120",
            "--export-arch",
            "cpu",
        ]
    )

    assert [call["run_tasks"] for call in run_calls] == [["soma"], ["mosh"]]
    assert len(export_calls) == 1
    assert export_calls[0]["input_pkl"].endswith("swing_stageii.pkl")
    assert export_calls[0]["model_path"] == str(model_path)
    assert export_calls[0]["fps"] == 15
    assert export_calls[0]["width"] == 160
    assert export_calls[0]["height"] == 120
    assert export_calls[0]["arch"] == "cpu"


def test_convert_tennis_can_export_existing_stageii_when_mosh_is_skipped(monkeypatch, tmp_path):
    dataset = "demo_ds"
    work_dir = tmp_path / "work"
    mocap_base_dir = tmp_path / "mocap"
    support_dir = tmp_path / "support"
    model_path = support_dir / "smplx" / "female" / "model.pkl"

    export_calls = []
    run_calls = []
    _install_fake_run_soma_module(monkeypatch, run_calls)
    _install_fake_export_module(monkeypatch, export_calls)

    stageii_path = work_dir / dataset / "subject02" / "existing_stageii.pkl"
    _write_stageii_pickle(stageii_path, model_path=model_path)

    convert_tennis.main(
        [
            "--dataset",
            dataset,
            "--mocap-base-dir",
            str(mocap_base_dir),
            "--soma-work-base-dir",
            str(work_dir),
            "--support-base-dir",
            str(support_dir),
            "--skip-mosh",
            "--export-artifacts",
        ]
    )

    assert [call["run_tasks"] for call in run_calls] == [["soma"]]
    assert len(export_calls) == 1
    assert export_calls[0]["input_pkl"] == str(stageii_path)
    assert export_calls[0]["model_path"] == str(model_path)


def test_convert_tennis_still_rejects_skipping_soma_and_mosh_together(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        convert_tennis.main(
            [
                "--dataset",
                "demo_ds",
                "--mocap-base-dir",
                str(tmp_path / "mocap"),
                "--skip-soma",
                "--skip-mosh",
                "--export-artifacts",
            ]
        )

    assert excinfo.value.code == 2
