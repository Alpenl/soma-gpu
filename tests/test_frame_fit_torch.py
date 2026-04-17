import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.transformed_lm_torch import build_marker_attachment


def _load_frame_fit_torch_module():
    try:
        spec = importlib.util.find_spec("moshpp.optim.frame_fit_torch")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "moshpp.optim.frame_fit_torch is not available on main yet"
    return importlib.import_module("moshpp.optim.frame_fit_torch")


class DummyBodyOutput:
    def __init__(self, vertices, joints):
        self.vertices = vertices
        self.joints = joints


class TranslOnlyBodyModel:
    def __init__(self, canonical_vertices):
        self.canonical_vertices = canonical_vertices

    def __call__(self, **kwargs):
        transl = kwargs["transl"]
        vertices = self.canonical_vertices.unsqueeze(0) + transl[:, None, :]
        joints = torch.zeros(transl.shape[0], 3, 3, dtype=vertices.dtype, device=vertices.device)
        return DummyBodyOutput(vertices=vertices, joints=joints)


class ZeroPosePrior:
    def __init__(self, dim):
        self.means = torch.zeros(1, dim)

    def __call__(self, x):
        return torch.zeros(x.shape[0], x.shape[1] + 1, dtype=x.dtype, device=x.device)


def _orthonormal_components(rows, cols):
    basis, _ = torch.linalg.qr(torch.randn(cols, cols, dtype=torch.float32))
    return basis[:, :rows].T.contiguous()


def test_encode_decode_stageii_hand_pca_roundtrip():
    module = _load_frame_fit_torch_module()
    layout = module.make_stageii_latent_layout(
        surface_model_type="smplx",
        dof_per_hand=6,
        optimize_fingers=True,
        optimize_face=True,
    )
    hand_pca = module.HandPcaSpec(
        left_components=_orthonormal_components(6, 45),
        right_components=_orthonormal_components(6, 45),
        left_mean=torch.linspace(-0.2, 0.2, 45, dtype=torch.float32),
        right_mean=torch.linspace(0.3, -0.1, 45, dtype=torch.float32),
    )

    latent_pose = torch.zeros(2, layout.latent_dim, dtype=torch.float32)
    latent_pose[:, layout.root_slice] = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.2, -0.1]], dtype=torch.float32)
    latent_pose[:, layout.body_slice] = torch.linspace(-0.5, 0.5, 63, dtype=torch.float32)
    latent_pose[:, layout.jaw_slice] = torch.tensor([[0.01, 0.02, 0.03], [0.03, -0.02, 0.01]], dtype=torch.float32)
    latent_pose[:, layout.leye_slice] = torch.tensor([[0.1, 0.0, -0.1], [-0.1, 0.2, 0.0]], dtype=torch.float32)
    latent_pose[:, layout.reye_slice] = torch.tensor([[0.0, 0.1, -0.1], [0.1, -0.2, 0.3]], dtype=torch.float32)
    latent_pose[:, layout.left_hand_coeff_slice] = torch.tensor(
        [[0.5, -0.2, 0.1, 0.3, -0.4, 0.2], [-0.1, 0.4, -0.3, 0.2, 0.1, -0.5]],
        dtype=torch.float32,
    )
    latent_pose[:, layout.right_hand_coeff_slice] = torch.tensor(
        [[-0.3, 0.2, 0.6, -0.4, 0.1, 0.2], [0.2, -0.1, 0.3, 0.4, -0.2, 0.5]],
        dtype=torch.float32,
    )

    fullpose = module.decode_stageii_latent_pose(latent_pose, layout, hand_pca=hand_pca)
    encoded = module.encode_stageii_fullpose(fullpose, layout, hand_pca=hand_pca)
    roundtrip = module.decode_stageii_latent_pose(encoded, layout, hand_pca=hand_pca)

    assert encoded.shape == latent_pose.shape
    assert torch.allclose(encoded, latent_pose, atol=1e-5)
    assert torch.allclose(roundtrip, fullpose, atol=1e-5)


def test_fit_stageii_frame_torch_recovers_translation_on_synthetic_markers():
    module = _load_frame_fit_torch_module()
    can_body = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    marker_attachment = build_marker_attachment(can_body, can_body)
    target_offset = torch.tensor([0.25, -0.15, 0.35], dtype=torch.float32)
    marker_observations = can_body + target_offset

    layout = module.make_stageii_latent_layout(
        surface_model_type="smplx",
        dof_per_hand=24,
        optimize_fingers=False,
        optimize_face=False,
    )

    result = module.fit_stageii_frame_torch(
        body_model=TranslOnlyBodyModel(can_body),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(1, layout.latent_dim),
        transl_init=torch.zeros(1, 3),
        weights=module.TorchFrameFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
        ),
        options=module.TorchFrameFitOptions(rigid_iters=25, warmup_iters=25, refine_iters=25),
    )

    assert result.predicted_markers.shape == (4, 3)
    assert result.transl.shape == (1, 3)
    assert torch.allclose(result.transl[0], target_offset, atol=1e-3)
    assert torch.allclose(result.predicted_markers, marker_observations, atol=1e-3)
    assert result.loss_terms["data"] < 1e-5


def test_build_stageii_evaluator_supports_optional_torch_compile(monkeypatch):
    module = _load_frame_fit_torch_module()
    layout = module.make_stageii_latent_layout(
        surface_model_type="smplx",
        dof_per_hand=24,
        optimize_fingers=False,
        optimize_face=False,
    )
    recorded = {}

    def fake_compile(compiled_module, *, mode, fullgraph):
        recorded["module"] = compiled_module
        recorded["mode"] = mode
        recorded["fullgraph"] = fullgraph
        return "compiled-evaluator"

    monkeypatch.setattr(torch, "compile", fake_compile)

    compiled = module.build_stageii_evaluator(
        wrapper=object(),
        layout=layout,
        hand_pca=None,
        pose_prior=ZeroPosePrior(63),
        optimize_fingers=False,
        optimize_face=False,
        compile_module=True,
        compile_mode="max-autotune",
        compile_fullgraph=True,
    )

    assert compiled == "compiled-evaluator"
    assert recorded["mode"] == "max-autotune"
    assert recorded["fullgraph"] is True
    assert hasattr(recorded["module"], "forward")


def test_fit_stageii_frame_torch_uses_stageii_evaluator_for_final_pass(monkeypatch):
    module = _load_frame_fit_torch_module()
    layout = module.make_stageii_latent_layout(
        surface_model_type="smplx",
        dof_per_hand=24,
        optimize_fingers=False,
        optimize_face=False,
    )
    can_body = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    marker_attachment = build_marker_attachment(can_body, can_body)
    marker_observations = can_body + torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    recorded = {"calls": 0}

    def fake_evaluate_stageii_frame(**kwargs):
        recorded["calls"] += 1
        return SimpleNamespace(
            total=torch.tensor(0.0, dtype=torch.float32),
            loss_terms={
                "data": torch.tensor(0.0, dtype=torch.float32),
                "poseB": torch.tensor(0.0, dtype=torch.float32),
                "poseH": torch.tensor(0.0, dtype=torch.float32),
                "poseF": torch.tensor(0.0, dtype=torch.float32),
                "expr": torch.tensor(0.0, dtype=torch.float32),
                "velo": torch.tensor(0.0, dtype=torch.float32),
            },
            fullpose=torch.zeros(1, layout.fullpose_dim, dtype=torch.float32),
            body_output=DummyBodyOutput(
                vertices=marker_observations.unsqueeze(0).clone(),
                joints=torch.zeros(1, 3, 3, dtype=torch.float32),
            ),
            predicted_markers=marker_observations.clone(),
        )

    monkeypatch.setattr(module, "evaluate_stageii_frame", fake_evaluate_stageii_frame, raising=False)

    result = module.fit_stageii_frame_torch(
        body_model=TranslOnlyBodyModel(can_body),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(1, layout.latent_dim),
        transl_init=torch.zeros(1, 3),
        weights=module.TorchFrameFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
        ),
        options=module.TorchFrameFitOptions(rigid_iters=0, warmup_iters=0, refine_iters=0),
    )

    assert recorded["calls"] == 1
    assert result.predicted_markers.shape == (4, 3)
    assert result.loss_terms["data"] == 0.0
