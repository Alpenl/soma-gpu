import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.stageii_backend import choose_stageii_backend, load_stageii_backend


def test_choose_stageii_backend_defaults_to_chumpy():
    def chumpy_backend():
        return "chumpy"

    def torch_backend():
        return "torch"

    assert choose_stageii_backend(None, chumpy_backend, torch_backend) is chumpy_backend
    assert choose_stageii_backend("chumpy", chumpy_backend, torch_backend) is chumpy_backend


def test_choose_stageii_backend_returns_torch_backend():
    def chumpy_backend():
        return "chumpy"

    def torch_backend():
        return "torch"

    assert choose_stageii_backend("torch", chumpy_backend, torch_backend) is torch_backend


def test_load_stageii_backend_keeps_default_path_lazy():
    chumpy_backend = object()
    calls = 0

    def load_torch_backend():
        nonlocal calls
        calls += 1
        return object()

    assert load_stageii_backend(None, chumpy_backend, load_torch_backend) is chumpy_backend
    assert load_stageii_backend("chumpy", chumpy_backend, load_torch_backend) is chumpy_backend
    assert calls == 0


def test_load_stageii_backend_loads_torch_only_when_requested():
    chumpy_backend = object()
    torch_backend = object()
    calls = 0

    def load_torch_backend():
        nonlocal calls
        calls += 1
        return torch_backend

    assert load_stageii_backend("torch", chumpy_backend, load_torch_backend) is torch_backend
    assert calls == 1
