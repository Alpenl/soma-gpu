import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.stageii_backend import choose_stageii_backend


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
