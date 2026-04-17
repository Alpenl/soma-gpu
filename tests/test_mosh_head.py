import importlib
import pickle
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.tools.mocap_interface import MocapSession


def test_mosh_head_module_imports_without_optional_visualization_deps():
    result = subprocess.run(
        [sys.executable, "-c", "from moshpp.mosh_head import run_moshpp_once; print('ok')"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout


def test_run_moshpp_once_skips_chumpy_import_for_cached_stagei_with_torch_backend(monkeypatch, tmp_path):
    module = importlib.import_module("moshpp.mosh_head")

    stagei_path = tmp_path / "cached_stagei.pkl"
    stagei_path.write_bytes(b"cached")
    stageii_path = tmp_path / "missing_stageii.pkl"

    seen = {"imports": []}

    def fake_import_module(name):
        seen["imports"].append(name)
        if name == "moshpp.chmosh_torch":
            return SimpleNamespace(mosh_stageii_torch="torch-backend")
        raise AssertionError(f"unexpected import: {name}")

    class DummyMoSh:
        def __init__(self, **_cfg):
            self.cfg = SimpleNamespace(runtime=SimpleNamespace(stagei_only=False, backend="torch"))
            self.stagei_fname = str(stagei_path)
            self.stageii_fname = str(stageii_path)
            self.stagei_data = None
            self.stageii_data = None

        def mosh_stagei(self, backend):
            seen["stagei_backend"] = backend
            self.stagei_data = {"stagei_debug_details": {"stagei_errs": {}}}

        def mosh_stageii(self, backend):
            seen["stageii_backend"] = backend
            self.stageii_data = {"stageii_debug_details": {"stageii_errs": {}}}

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(module, "MoSh", DummyMoSh)

    module.run_moshpp_once({})

    assert seen["imports"] == ["moshpp.chmosh_torch"]
    assert callable(seen["stagei_backend"])
    assert seen["stageii_backend"] == "torch-backend"


def test_mocap_session_reads_mcp_alias_for_c3d_payload(tmp_path):
    mocap_path = tmp_path / "input" / "wolf001" / "capture.mcp"
    mocap_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "out1.c3d", mocap_path)

    session = MocapSession(mocap_path, mocap_unit="mm")

    assert session.read_status is True
    assert session.markers.shape[1:] == (54, 3)
    assert session.frame_rate == pytest.approx(60.0)
    assert session.subject_names == ["null"]


def test_prepare_cfg_resolves_mcp_subjects_without_manual_subject_overrides(tmp_path):
    module = importlib.import_module("moshpp.mosh_head")
    mocap_path = tmp_path / "input" / "wolf001" / "capture.mcp"
    mocap_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "out1.c3d", mocap_path)

    cfg = module.MoSh.prepare_cfg(**{"mocap.fname": str(mocap_path)})

    assert cfg.mocap.ds_name == "input"
    assert cfg.mocap.session_name == "wolf001"
    assert cfg.mocap.basename == "capture"
    assert cfg.mocap.subject_names == ["null"]
    assert cfg.mocap.subject_name is None
    assert cfg.mocap.multi_subject is False


def test_prepare_cfg_keeps_marker_layout_aliases_without_deprecation_warnings(tmp_path):
    module = importlib.import_module("moshpp.mosh_head")
    mocap_path = tmp_path / "input" / "wolf001" / "capture.mcp"
    mocap_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "out1.c3d", mocap_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = module.MoSh.prepare_cfg(
            **{
                "mocap.fname": str(mocap_path),
                "dirs.support_base_dir": str(tmp_path / "support_files"),
                "dirs.work_base_dir": str(tmp_path / "work"),
                "surface_model.gender": "male",
            }
        )
        legacy_basename = cfg.dirs.markerlyout_basename
        legacy_fname = cfg.dirs.marker_layout_fname

    assert legacy_basename == cfg.dirs.marker_layout.basename
    assert legacy_fname == cfg.dirs.marker_layout.fname
    assert not any(
        "'dirs.markerlyout_basename' is deprecated" in str(warning.message)
        or "'dirs.marker_layout_fname' is deprecated" in str(warning.message)
        for warning in caught
    )


def test_mosh_stagei_accepts_cached_surface_model_when_relative_and_absolute_paths_match(
    tmp_path, monkeypatch
):
    module = importlib.import_module("moshpp.mosh_head")
    monkeypatch.chdir(tmp_path)

    model_path = tmp_path / "support_files" / "smplx" / "male" / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")

    stagei_path = tmp_path / "cached_stagei.pkl"
    with stagei_path.open("wb") as handle:
        pickle.dump(
            {
                "stagei_debug_details": {
                    "cfg": {
                        "surface_model": {
                            "fname": str(model_path.resolve()),
                        }
                    }
                }
            },
            handle,
        )

    dummy = object.__new__(module.MoSh)
    dummy.stagei_fname = str(stagei_path)
    dummy.stagei_data = None
    dummy.cfg = SimpleNamespace(
        surface_model=SimpleNamespace(
            fname="support_files/smplx/male/model.pkl",
            type="smplx",
        )
    )

    module.MoSh.mosh_stagei(dummy, lambda *_args, **_kwargs: pytest.fail("cached stagei should be reused"))

    assert dummy.stagei_data["stagei_debug_details"]["cfg"]["surface_model"]["fname"] == str(
        model_path.resolve()
    )


def test_run_moshpp_once_keeps_null_rotate_when_reusing_prepared_cfg(tmp_path, monkeypatch):
    module = importlib.import_module("moshpp.mosh_head")
    real_mosh_cls = module.MoSh

    mocap_path = tmp_path / "input" / "wolf001" / "capture.mcp"
    mocap_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "out1.c3d", mocap_path)

    prepared_cfg = real_mosh_cls.prepare_cfg(
        **{
            "mocap.fname": str(mocap_path),
            "dirs.support_base_dir": str(tmp_path / "support_files"),
            "dirs.work_base_dir": str(tmp_path / "work"),
            "runtime.backend": "torch",
        }
    )
    assert prepared_cfg.mocap.rotate is None

    seen = {}

    class DummyMoSh:
        def __init__(self, dict_cfg=None, **kwargs):
            seen["dict_cfg"] = dict_cfg
            seen["kwargs"] = kwargs
            self.cfg = dict_cfg if dict_cfg is not None else real_mosh_cls.prepare_cfg(**kwargs)
            self.stagei_fname = str(tmp_path / "cached_stagei.pkl")
            self.stageii_fname = str(tmp_path / "cached_stageii.pkl")
            Path(self.stagei_fname).write_bytes(b"cached")
            Path(self.stageii_fname).write_bytes(b"cached")
            self.stagei_data = None
            self.stageii_data = None

        def mosh_stagei(self, _backend):
            seen["rotate_after_rebuild"] = self.cfg.mocap.rotate
            self.stagei_data = {"stagei_debug_details": {"stagei_errs": {"data": np.zeros(1)}}}

        def mosh_stageii(self, _backend):
            self.stageii_data = {"stageii_debug_details": {"stageii_errs": {"data": np.zeros(1)}}}

    monkeypatch.setattr(module, "MoSh", DummyMoSh)

    module.run_moshpp_once(prepared_cfg)

    assert seen["rotate_after_rebuild"] is None


def test_resolved_debug_cfg_snapshot_avoids_deprecated_marker_layout_alias_warnings(tmp_path):
    module = importlib.import_module("moshpp.mosh_head")

    mocap_path = tmp_path / "input" / "wolf001" / "capture.mcp"
    mocap_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "out1.c3d", mocap_path)

    cfg = module.MoSh.prepare_cfg(
        **{
            "mocap.fname": str(mocap_path),
            "dirs.support_base_dir": str(tmp_path / "support_files"),
            "dirs.work_base_dir": str(tmp_path / "work"),
            "surface_model.gender": "male",
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        snapshot = module._resolved_debug_cfg_snapshot(cfg)

    assert snapshot["dirs"]["marker_layout"]["basename"] == "input_smplx"
    assert snapshot["dirs"]["marker_layout"]["fname"].endswith("input_smplx.json")
    assert "markerlyout_basename" not in snapshot["dirs"]
    assert "marker_layout_fname" not in snapshot["dirs"]
    assert not any(
        "'dirs.markerlyout_basename' is deprecated" in str(warning.message)
        or "'dirs.marker_layout_fname' is deprecated" in str(warning.message)
        for warning in caught
    )
