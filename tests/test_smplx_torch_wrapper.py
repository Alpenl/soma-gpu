import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.models.smplx_torch_wrapper import SmplxTorchWrapper


class DummyBodyOutput:
    def __init__(self, vertices, joints):
        self.vertices = vertices
        self.joints = joints


class DummyBodyModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["global_orient"].shape[0]
        vertices = torch.zeros(batch_size, 5, 3)
        joints = torch.zeros(batch_size, 3, 3)
        return DummyBodyOutput(vertices=vertices, joints=joints)


class DummyFactory:
    def __init__(self):
        self.calls = 0
        self.instances = []

    def __call__(self):
        self.calls += 1
        instance = DummyBodyModel()
        self.instances.append(instance)
        return instance


def test_smplx_torch_wrapper_splits_fullpose_and_passes_expected_kwargs():
    body_model = DummyBodyModel()
    wrapper = SmplxTorchWrapper(body_model, surface_model_type="smplx")

    fullpose = torch.arange(2 * 165, dtype=torch.float32).reshape(2, 165)
    betas = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    transl = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    expression = torch.arange(20, dtype=torch.float32).reshape(2, 10)

    result = wrapper(fullpose=fullpose, betas=betas, transl=transl, expression=expression)

    assert result.vertices.shape == (2, 5, 3)
    assert result.joints.shape == (2, 3, 3)

    assert len(body_model.calls) == 1
    call = body_model.calls[0]
    assert torch.equal(call["global_orient"], fullpose[:, 0:3])
    assert torch.equal(call["body_pose"], fullpose[:, 3:66])
    assert torch.equal(call["jaw_pose"], fullpose[:, 66:69])
    assert torch.equal(call["leye_pose"], fullpose[:, 69:72])
    assert torch.equal(call["reye_pose"], fullpose[:, 72:75])
    assert torch.equal(call["left_hand_pose"], fullpose[:, 75:120])
    assert torch.equal(call["right_hand_pose"], fullpose[:, 120:165])
    assert torch.equal(call["betas"], betas)
    assert torch.equal(call["transl"], transl)
    assert torch.equal(call["expression"], expression)


def test_smplx_torch_wrapper_broadcasts_single_beta_row():
    body_model = DummyBodyModel()
    wrapper = SmplxTorchWrapper(body_model, surface_model_type="smplx")

    fullpose = torch.zeros(3, 165)
    betas = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    transl = torch.arange(3, dtype=torch.float32).reshape(1, 3)

    wrapper(fullpose=fullpose, betas=betas, transl=transl)

    call = body_model.calls[0]
    assert call["betas"].shape == (3, 10)
    assert call["transl"].shape == (3, 3)
    assert call["betas"].is_contiguous()
    assert call["transl"].is_contiguous()
    assert torch.equal(call["betas"][0], betas[0])
    assert torch.equal(call["betas"][1], betas[0])
    assert torch.equal(call["transl"][2], transl[0])


def test_smplx_torch_wrapper_builds_body_model_from_factory_once():
    factory = DummyFactory()
    wrapper = SmplxTorchWrapper(body_model_factory=factory, surface_model_type="smplx")

    fullpose = torch.zeros(2, 165)
    betas = torch.zeros(2, 10)
    transl = torch.zeros(2, 3)

    wrapper(fullpose=fullpose, betas=betas, transl=transl)
    wrapper(fullpose=fullpose, betas=betas, transl=transl)

    assert factory.calls == 1
    assert len(factory.instances[0].calls) == 2


def test_smplx_torch_wrapper_omits_expression_when_not_provided():
    body_model = DummyBodyModel()
    wrapper = SmplxTorchWrapper(body_model, surface_model_type="smplx")

    fullpose = torch.zeros(1, 165)
    betas = torch.zeros(1, 10)
    transl = torch.zeros(1, 3)

    wrapper(fullpose=fullpose, betas=betas, transl=transl)

    assert "expression" not in body_model.calls[0]


def test_smplx_torch_wrapper_rejects_invalid_fullpose_shape():
    wrapper = SmplxTorchWrapper(DummyBodyModel(), surface_model_type="smplx")

    fullpose = torch.zeros(165)
    betas = torch.zeros(1, 10)
    transl = torch.zeros(1, 3)

    try:
        wrapper(fullpose=fullpose, betas=betas, transl=transl)
    except ValueError as exc:
        assert "fullpose" in str(exc)
    else:
        raise AssertionError("Expected invalid fullpose shape to raise ValueError.")
