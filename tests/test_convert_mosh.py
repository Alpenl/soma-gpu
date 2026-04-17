import pickle
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import convert_mosh


def _install_fake_mosh_manual_module(monkeypatch, calls):
    manual_module = types.ModuleType("soma.amass.mosh_manual")

    def fake_mosh_manual(**kwargs):
        calls.append(kwargs)

    manual_module.mosh_manual = fake_mosh_manual

    amass_module = types.ModuleType("soma.amass")
    amass_module.mosh_manual = manual_module

    soma_module = types.ModuleType("soma")
    soma_module.amass = amass_module

    monkeypatch.setitem(sys.modules, "soma", soma_module)
    monkeypatch.setitem(sys.modules, "soma.amass", amass_module)
    monkeypatch.setitem(sys.modules, "soma.amass.mosh_manual", manual_module)


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


def _write_stageii_pickle(path, *, model_path, surface_model_type="smplx", gender="male"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {
                "stageii_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "type": surface_model_type,
                            "fname": str(model_path),
                            "gender": gender,
                        }
                    }
                }
            }
        )
    )


def test_collect_mocap_fnames_includes_mcp_aliases(tmp_path):
    dataset = "demo_ds"
    subject_dir = tmp_path / dataset / "subject01"
    c3d_path = subject_dir / "serve.c3d"
    mcp_path = subject_dir / "swing.mcp"
    subject_dir.mkdir(parents=True, exist_ok=True)
    c3d_path.write_bytes(b"c3d")
    mcp_path.write_bytes(b"mcp")

    mocap_fnames = convert_mosh.collect_mocap_fnames(str(tmp_path), dataset, [])

    assert mocap_fnames == [str(c3d_path), str(mcp_path)]


def test_convert_mosh_accepts_mcp_inputs_and_exports_matching_stageii_artifacts(
    monkeypatch, tmp_path
):
    dataset = "demo_ds"
    work_dir = tmp_path / "work"
    mocap_base_dir = tmp_path / "mocap"
    support_dir = tmp_path / "support"
    model_path = support_dir / "smplx" / "male" / "model.npz"
    mocap_path = mocap_base_dir / dataset / "subject01" / "swing_take.mcp"

    export_calls = []
    mosh_calls = []
    _install_fake_mosh_manual_module(monkeypatch, mosh_calls)
    _install_fake_export_module(monkeypatch, export_calls)

    mocap_path.parent.mkdir(parents=True, exist_ok=True)
    mocap_path.write_bytes(b"mcp")
    _write_stageii_pickle(
        work_dir / "mosh_results" / dataset / "subject01" / "swing_take_stageii.pkl",
        model_path=model_path,
    )
    _write_stageii_pickle(
        work_dir / "mosh_results" / dataset / "subject01" / "serve_take_stageii.pkl",
        model_path=model_path,
    )

    convert_mosh.main(
        [
            "--dataset",
            dataset,
            "--mocap-base-dir",
            str(mocap_base_dir),
            "--work-base-dir",
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

    assert len(mosh_calls) == 1
    assert mosh_calls[0]["mocap_fnames"] == [str(mocap_path)]
    assert len(export_calls) == 1
    assert export_calls[0]["input_pkl"].endswith("swing_take_stageii.pkl")
    assert export_calls[0]["model_path"] == str(model_path)
    assert export_calls[0]["fps"] == 15
    assert export_calls[0]["width"] == 160
    assert export_calls[0]["height"] == 120
    assert export_calls[0]["arch"] == "cpu"


def test_convert_mosh_export_relocates_model_path_under_support_base_dir(
    monkeypatch, tmp_path
):
    dataset = "demo_ds"
    work_dir = tmp_path / "work"
    mocap_base_dir = tmp_path / "mocap"
    support_dir = tmp_path / "support"
    relocated_model_path = support_dir / "smplx" / "male" / "model.npz"
    mocap_path = mocap_base_dir / dataset / "subject02" / "portable_take.mcp"

    export_calls = []
    mosh_calls = []
    _install_fake_mosh_manual_module(monkeypatch, mosh_calls)
    _install_fake_export_module(monkeypatch, export_calls)

    relocated_model_path.parent.mkdir(parents=True, exist_ok=True)
    relocated_model_path.write_bytes(b"npz")
    mocap_path.parent.mkdir(parents=True, exist_ok=True)
    mocap_path.write_bytes(b"mcp")
    _write_stageii_pickle(
        work_dir / "mosh_results" / dataset / "subject02" / "portable_take_stageii.pkl",
        model_path="/old-machine/support_files/smplx/male/model.pkl",
    )

    convert_mosh.main(
        [
            "--dataset",
            dataset,
            "--mocap-base-dir",
            str(mocap_base_dir),
            "--work-base-dir",
            str(work_dir),
            "--support-base-dir",
            str(support_dir),
            "--export-artifacts",
        ]
    )

    assert len(mosh_calls) == 1
    assert len(export_calls) == 1
    assert export_calls[0]["model_path"] == str(relocated_model_path)


def test_convert_mosh_errors_when_no_c3d_or_mcp_inputs_match(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        convert_mosh.main(
            [
                "--dataset",
                "demo_ds",
                "--mocap-base-dir",
                str(tmp_path / "mocap"),
            ]
        )

    assert excinfo.value.code == 2
