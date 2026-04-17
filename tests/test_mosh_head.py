import importlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
