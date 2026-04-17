import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


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
