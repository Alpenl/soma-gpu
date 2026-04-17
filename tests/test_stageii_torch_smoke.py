import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "support_data" / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.prior.gmm_prior_torch import prepare_gmm_prior
from moshpp.tools.stageii_torch_smoke import (
    load_mocap_frame,
    load_stageii_frame_inputs,
    run_stageii_torch_smoke,
)
from moshpp.transformed_lm_torch import build_marker_attachment


class DummyBodyOutput:
    def __init__(self, vertices, joints):
        self.vertices = vertices
        self.joints = joints


class DummyBodyModel:
    def __init__(self, vertices):
        self.vertices = vertices

    def __call__(self, **kwargs):
        batch_size = kwargs["global_orient"].shape[0]
        vertices = self.vertices.unsqueeze(0).expand(batch_size, -1, -1).clone()
        joints = torch.zeros(batch_size, 3, 3)
        return DummyBodyOutput(vertices=vertices, joints=joints)


def test_stageii_torch_smoke_validates_required_inputs_before_running():
    fullpose = torch.zeros(1, 165)
    betas = torch.zeros(1, 10)
    transl = torch.zeros(1, 3)
    marker_observations = torch.zeros(1, 3)

    with pytest.raises(ValueError, match="marker_attachment"):
        run_stageii_torch_smoke(
            body_model=DummyBodyModel(torch.zeros(4, 3)),
            fullpose=fullpose,
            betas=betas,
            transl=transl,
            marker_attachment=None,
            marker_observations=marker_observations,
            pose_prior=None,
        )


def test_stageii_torch_smoke_runs_on_synthetic_single_frame():
    can_body = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    markers_latent = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)
    marker_attachment = build_marker_attachment(can_body, markers_latent)
    marker_observations = markers_latent.clone()

    means = torch.zeros(1, 63, dtype=torch.float32)
    covars = torch.eye(63, dtype=torch.float32).unsqueeze(0)
    weights = torch.ones(1, dtype=torch.float32)
    pose_prior = prepare_gmm_prior(means, covars, weights)

    result = run_stageii_torch_smoke(
        body_model=DummyBodyModel(can_body),
        fullpose=torch.zeros(1, 165),
        betas=torch.zeros(1, 10),
        transl=torch.zeros(1, 3),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=pose_prior,
    )

    assert result.vertices.shape == (1, 4, 3)
    assert result.joints.shape == (1, 3, 3)
    assert result.predicted_markers.shape == (1, 3)
    assert result.data_residual.shape == (1, 3)
    assert result.prior_residual.shape == (1, 64)
    assert torch.allclose(result.predicted_markers, marker_observations, atol=1e-5)


def test_load_stageii_frame_inputs_supports_real_legacy_stageii_sample():
    loaded = load_stageii_frame_inputs(SAMPLES / "mosh_stageii.pkl", frame_idx=0)

    assert loaded.source_format == "legacy_stageii_pkl"
    assert loaded.surface_model_type == "smplh"
    assert loaded.fullpose.shape == (1, 165)
    assert loaded.betas.ndim == 2 and loaded.betas.shape[0] == 1
    assert loaded.transl.shape == (1, 3)
    assert loaded.markers_latent.shape == (67, 3)
    assert loaded.marker_observations.shape == (67, 3)
    assert len(loaded.latent_labels) == 67
    assert len(loaded.marker_labels) == 67
    assert loaded.fullpose.dtype == torch.float32


def test_load_mocap_frame_supports_real_pickle_sample():
    loaded = load_mocap_frame(SAMPLES / "0006_normal_walk2.pkl", frame_idx=0)

    assert loaded.source_format == "mocap_pkl"
    assert loaded.markers.shape == (49, 3)
    assert len(loaded.labels) == 49
    assert loaded.frame_rate == pytest.approx(120.0)


def test_load_mocap_frame_rejects_c3d_until_parser_fix_lands():
    with pytest.raises(ValueError, match="Unsupported mocap file format"):
        load_mocap_frame(ROOT / "out1.c3d", frame_idx=0)


def test_stageii_torch_smoke_runs_on_real_legacy_stageii_frame_with_dummy_body():
    loaded = load_stageii_frame_inputs(SAMPLES / "mosh_stageii.pkl", frame_idx=0)

    means = torch.zeros(1, 63, dtype=torch.float32)
    covars = torch.eye(63, dtype=torch.float32).unsqueeze(0)
    weights = torch.ones(1, dtype=torch.float32)
    pose_prior = prepare_gmm_prior(means, covars, weights)

    marker_attachment = build_marker_attachment(loaded.markers_latent, loaded.markers_latent)

    result = run_stageii_torch_smoke(
        body_model=DummyBodyModel(loaded.markers_latent),
        fullpose=loaded.fullpose,
        betas=loaded.betas,
        transl=loaded.transl,
        marker_attachment=marker_attachment,
        marker_observations=loaded.marker_observations,
        pose_prior=pose_prior,
    )

    assert result.vertices.shape == (1, 67, 3)
    assert result.predicted_markers.shape == (67, 3)
    assert result.data_residual.shape == (67, 3)
    assert result.prior_residual.shape == (1, 64)
