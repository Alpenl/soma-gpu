import importlib
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.transformed_lm_torch import build_marker_attachment


def _load_batch_frame_fit_module():
    try:
        spec = importlib.util.find_spec("moshpp.optim.batch_frame_fit_torch")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "moshpp.optim.batch_frame_fit_torch is not available on main yet"
    return importlib.import_module("moshpp.optim.batch_frame_fit_torch")


def _load_frame_fit_module():
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


def test_fit_stageii_frames_batched_torch_recovers_translation_with_visible_mask():
    batch_module = _load_batch_frame_fit_module()
    frame_module = _load_frame_fit_module()

    canonical_markers = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    marker_attachment = build_marker_attachment(canonical_markers, canonical_markers)
    layout = frame_module.make_stageii_latent_layout(
        surface_model_type="smplx",
        dof_per_hand=24,
        optimize_fingers=False,
        optimize_face=False,
    )

    true_transl = torch.tensor(
        [
            [0.20, -0.10, 0.30],
            [-0.15, 0.25, 0.05],
            [0.05, 0.15, -0.20],
        ],
        dtype=torch.float32,
    )
    marker_observations = canonical_markers.unsqueeze(0) + true_transl[:, None, :]
    visible_mask = torch.ones(marker_observations.shape[:2], dtype=torch.bool)
    visible_mask[1, -1] = False
    marker_observations = marker_observations.clone()
    marker_observations[1, -1] = torch.tensor([12.0, 12.0, 12.0], dtype=torch.float32)

    options = frame_module.TorchFrameFitOptions(
        rigid_iters=40,
        warmup_iters=0,
        refine_iters=0,
        rigid_lr=1.0,
        history_size=10,
    )
    weights = [
        frame_module.TorchFrameFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
        )
        for _ in range(marker_observations.shape[0])
    ]

    result = batch_module.fit_stageii_frames_batched_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=visible_mask,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        weights=weights,
        options=options,
        optimize_fingers=False,
        optimize_face=False,
        optimize_toes=False,
    )

    assert result.transl.shape == true_transl.shape
    assert result.predicted_markers.shape == marker_observations.shape
    assert torch.allclose(result.transl, true_transl, atol=2e-3)
    assert torch.allclose(result.predicted_markers[0], canonical_markers + true_transl[0], atol=2e-3)
    assert torch.allclose(result.predicted_markers[2], canonical_markers + true_transl[2], atol=2e-3)
    assert torch.allclose(result.predicted_markers[1, :-1], canonical_markers[:-1] + true_transl[1], atol=2e-3)
    assert not bool(result.fallback_mask.any().item())
