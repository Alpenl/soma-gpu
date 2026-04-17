import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.optim.frame_fit_torch import make_stageii_latent_layout
from moshpp.transformed_lm_torch import build_marker_attachment


def _load_sequence_evaluator_module():
    try:
        spec = importlib.util.find_spec("moshpp.optim.sequence_evaluator_torch")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "moshpp.optim.sequence_evaluator_torch is not available on main yet"
    return importlib.import_module("moshpp.optim.sequence_evaluator_torch")


class DummyBodyOutput:
    def __init__(self, vertices, joints):
        self.vertices = vertices
        self.joints = joints


class TranslOnlyWrapper:
    def __init__(self, canonical_vertices):
        self.canonical_vertices = canonical_vertices

    def __call__(self, *, fullpose, betas, transl, expression=None):
        del fullpose, betas, expression
        vertices = self.canonical_vertices.unsqueeze(0) + transl[:, None, :]
        joints = torch.zeros(transl.shape[0], 3, 3, dtype=vertices.dtype, device=vertices.device)
        return DummyBodyOutput(vertices=vertices, joints=joints)


class ZeroPosePrior:
    def __init__(self, dim):
        self.means = torch.zeros(1, dim)

    def __call__(self, x):
        return torch.zeros(x.shape[0], x.shape[1] + 1, dtype=x.dtype, device=x.device)


def _make_problem(num_frames=6):
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
    layout = make_stageii_latent_layout(
        surface_model_type="smplx",
        dof_per_hand=24,
        optimize_fingers=False,
        optimize_face=False,
    )
    transl = torch.stack(
        [torch.tensor([0.1 * t, -0.05 * t, 0.03 * t], dtype=torch.float32) for t in range(num_frames)],
        dim=0,
    )
    marker_observations = canonical_markers.unsqueeze(0) + transl[:, None, :]
    return canonical_markers, marker_attachment, layout, transl, marker_observations


def test_evaluate_stageii_sequence_returns_expected_shapes_and_near_zero_data_term():
    module = _load_sequence_evaluator_module()
    canonical_markers, marker_attachment, layout, transl, marker_observations = _make_problem(num_frames=4)
    evaluator = module.build_stageii_sequence_evaluator(
        wrapper=TranslOnlyWrapper(canonical_markers),
        layout=layout,
        hand_pca=None,
        pose_prior=ZeroPosePrior(63),
        optimize_fingers=False,
        optimize_face=False,
    )

    weights = SimpleNamespace(
        data=1.0,
        pose_body=0.0,
        pose_hand=0.0,
        pose_face=0.0,
        expr=0.0,
        velocity=0.0,
        temporal_accel=0.0,
    )
    result = module.evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl=transl,
        expression=None,
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=torch.ones(marker_observations.shape[:2], dtype=torch.bool),
        marker_data_weights=None,
        weights=weights,
        velocity_reference=None,
    )

    assert result.predicted_markers.shape == marker_observations.shape
    assert result.body_output.vertices.shape == marker_observations.shape
    assert result.fullpose.shape == (marker_observations.shape[0], 165)
    assert set(result.loss_terms) == {"data", "poseB", "poseH", "poseF", "expr", "velo", "veloT", "accel"}
    for value in result.loss_terms.values():
        assert value.shape == (marker_observations.shape[0],)
    assert float(result.total) >= 0.0
    assert torch.max(result.loss_terms["data"]) < 1e-8


def test_evaluate_stageii_sequence_temporal_accel_penalizes_curved_translation_more_than_linear():
    module = _load_sequence_evaluator_module()
    canonical_markers, marker_attachment, layout, _, _ = _make_problem(num_frames=5)
    evaluator = module.build_stageii_sequence_evaluator(
        wrapper=TranslOnlyWrapper(canonical_markers),
        layout=layout,
        hand_pca=None,
        pose_prior=ZeroPosePrior(63),
        optimize_fingers=False,
        optimize_face=False,
    )

    linear_transl = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
        dtype=torch.float32,
    )
    curved_transl = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.25, 0.0, 0.0], [0.45, 0.0, 0.0], [0.7, 0.0, 0.0]],
        dtype=torch.float32,
    )
    weights = SimpleNamespace(
        data=1.0,
        pose_body=0.0,
        pose_hand=0.0,
        pose_face=0.0,
        expr=0.0,
        velocity=0.0,
        temporal_accel=5.0,
    )

    linear_eval = module.evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=torch.zeros(linear_transl.shape[0], layout.latent_dim),
        transl=linear_transl,
        expression=None,
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=canonical_markers.unsqueeze(0) + linear_transl[:, None, :],
        visible_mask=torch.ones((linear_transl.shape[0], canonical_markers.shape[0]), dtype=torch.bool),
        marker_data_weights=None,
        weights=weights,
        velocity_reference=None,
    )
    curved_eval = module.evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=torch.zeros(curved_transl.shape[0], layout.latent_dim),
        transl=curved_transl,
        expression=None,
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=canonical_markers.unsqueeze(0) + curved_transl[:, None, :],
        visible_mask=torch.ones((curved_transl.shape[0], canonical_markers.shape[0]), dtype=torch.bool),
        marker_data_weights=None,
        weights=weights,
        velocity_reference=None,
    )

    assert torch.sum(linear_eval.loss_terms["accel"]) < 1e-8
    assert torch.sum(curved_eval.loss_terms["accel"]) > torch.sum(linear_eval.loss_terms["accel"])


def test_evaluate_stageii_sequence_velocity_reference_adds_boundary_term_without_disabling_internal_diffs():
    module = _load_sequence_evaluator_module()
    canonical_markers, marker_attachment, layout, transl, marker_observations = _make_problem(num_frames=4)
    evaluator = module.build_stageii_sequence_evaluator(
        wrapper=TranslOnlyWrapper(canonical_markers),
        layout=layout,
        hand_pca=None,
        pose_prior=ZeroPosePrior(63),
        optimize_fingers=False,
        optimize_face=False,
    )

    latent_pose = torch.zeros(marker_observations.shape[0], layout.latent_dim)
    latent_pose[:, 0] = torch.tensor([1.0, 3.0, 6.0, 10.0], dtype=torch.float32)
    weights = SimpleNamespace(
        data=0.0,
        pose_body=0.0,
        pose_hand=0.0,
        pose_face=0.0,
        expr=0.0,
        velocity=2.0,
        temporal_accel=0.0,
    )

    result = module.evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=latent_pose,
        transl=transl,
        expression=None,
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=torch.ones(marker_observations.shape[:2], dtype=torch.bool),
        marker_data_weights=None,
        weights=weights,
        velocity_reference=torch.zeros(1, layout.latent_dim),
    )

    expected = torch.tensor(
        [
            (1.0 * weights.velocity) ** 2,
            ((3.0 - 1.0) * weights.velocity) ** 2,
            ((6.0 - 3.0) * weights.velocity) ** 2,
            ((10.0 - 6.0) * weights.velocity) ** 2,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(result.loss_terms["velo"], expected)


def test_evaluate_stageii_sequence_transl_velocity_reference_adds_boundary_term_without_disabling_internal_diffs():
    module = _load_sequence_evaluator_module()
    canonical_markers, marker_attachment, layout, _, _ = _make_problem(num_frames=4)
    evaluator = module.build_stageii_sequence_evaluator(
        wrapper=TranslOnlyWrapper(canonical_markers),
        layout=layout,
        hand_pca=None,
        pose_prior=ZeroPosePrior(63),
        optimize_fingers=False,
        optimize_face=False,
    )

    transl = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    marker_observations = canonical_markers.unsqueeze(0) + transl[:, None, :]
    weights = SimpleNamespace(
        data=0.0,
        pose_body=0.0,
        pose_hand=0.0,
        pose_face=0.0,
        expr=0.0,
        velocity=0.0,
        transl_velocity=2.0,
        temporal_accel=0.0,
    )

    result = module.evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl=transl,
        expression=None,
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=torch.ones(marker_observations.shape[:2], dtype=torch.bool),
        marker_data_weights=None,
        weights=weights,
        velocity_reference=None,
        transl_velocity_reference=torch.zeros(1, 3),
    )

    expected = torch.tensor(
        [
            (1.0 * weights.transl_velocity) ** 2,
            ((3.0 - 1.0) * weights.transl_velocity) ** 2,
            ((6.0 - 3.0) * weights.transl_velocity) ** 2,
            ((10.0 - 6.0) * weights.transl_velocity) ** 2,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(result.loss_terms["veloT"], expected)


def test_evaluate_stageii_sequence_transl_velocity_reference_can_target_overlap_keep_seam():
    module = _load_sequence_evaluator_module()
    canonical_markers, marker_attachment, layout, _, _ = _make_problem(num_frames=4)
    evaluator = module.build_stageii_sequence_evaluator(
        wrapper=TranslOnlyWrapper(canonical_markers),
        layout=layout,
        hand_pca=None,
        pose_prior=ZeroPosePrior(63),
        optimize_fingers=False,
        optimize_face=False,
    )

    transl = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    marker_observations = canonical_markers.unsqueeze(0) + transl[:, None, :]
    weights = SimpleNamespace(
        data=0.0,
        pose_body=0.0,
        pose_hand=0.0,
        pose_face=0.0,
        expr=0.0,
        velocity=0.0,
        transl_velocity=2.0,
        temporal_accel=0.0,
    )

    result = module.evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl=transl,
        expression=None,
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=torch.ones(marker_observations.shape[:2], dtype=torch.bool),
        marker_data_weights=None,
        weights=weights,
        velocity_reference=None,
        transl_velocity_reference=torch.tensor([[9.0, 0.0, 0.0]], dtype=torch.float32),
        transl_velocity_reference_index=2,
    )

    expected = torch.tensor(
        [
            0.0,
            ((3.0 - 1.0) * weights.transl_velocity) ** 2,
            ((6.0 - 9.0) * weights.transl_velocity) ** 2,
            ((10.0 - 6.0) * weights.transl_velocity) ** 2,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(result.loss_terms["veloT"], expected)


def test_evaluate_stageii_sequence_transl_velocity_reference_window_can_preserve_local_structure_past_keep_seam():
    module = _load_sequence_evaluator_module()
    canonical_markers, marker_attachment, layout, _, _ = _make_problem(num_frames=4)
    evaluator = module.build_stageii_sequence_evaluator(
        wrapper=TranslOnlyWrapper(canonical_markers),
        layout=layout,
        hand_pca=None,
        pose_prior=ZeroPosePrior(63),
        optimize_fingers=False,
        optimize_face=False,
    )

    transl = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    marker_observations = canonical_markers.unsqueeze(0) + transl[:, None, :]
    weights = SimpleNamespace(
        data=0.0,
        pose_body=0.0,
        pose_hand=0.0,
        pose_face=0.0,
        expr=0.0,
        velocity=0.0,
        transl_velocity=2.0,
        temporal_accel=0.0,
    )

    result = module.evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl=transl,
        expression=None,
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=torch.ones(marker_observations.shape[:2], dtype=torch.bool),
        marker_data_weights=None,
        weights=weights,
        velocity_reference=None,
        transl_velocity_reference=torch.tensor(
            [[9.0, 0.0, 0.0], [9.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        transl_velocity_reference_index=2,
    )

    expected = torch.tensor(
        [
            0.0,
            ((3.0 - 1.0) * weights.transl_velocity) ** 2,
            (((6.0 - 9.0) - (9.0 - 9.0)) * weights.transl_velocity) ** 2,
            (((10.0 - 6.0) - (10.0 - 9.0)) * weights.transl_velocity) ** 2,
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(result.loss_terms["veloT"], expected)


def test_evaluate_stageii_sequence_delta_terms_match_manual_l2_with_seed_broadcast():
    module = _load_sequence_evaluator_module()
    canonical_markers, marker_attachment, layout, transl, marker_observations = _make_problem(num_frames=3)
    evaluator = module.build_stageii_sequence_evaluator(
        wrapper=TranslOnlyWrapper(canonical_markers),
        layout=layout,
        hand_pca=None,
        pose_prior=ZeroPosePrior(63),
        optimize_fingers=False,
        optimize_face=False,
    )

    latent_pose = torch.zeros(marker_observations.shape[0], layout.latent_dim)
    latent_pose[:, 0] = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float32)
    expression = torch.tensor(
        [[0.1, -0.2], [0.3, -0.1], [0.0, 0.4]],
        dtype=torch.float32,
    )
    weights = SimpleNamespace(
        data=0.0,
        pose_body=0.0,
        pose_hand=0.0,
        pose_face=0.0,
        expr=0.0,
        velocity=0.0,
        temporal_accel=0.0,
        delta_pose=2.0,
        delta_trans=3.0,
        delta_expr=4.0,
    )

    result = module.evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=latent_pose,
        transl=transl,
        expression=expression,
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=torch.ones(marker_observations.shape[:2], dtype=torch.bool),
        marker_data_weights=None,
        weights=weights,
        velocity_reference=None,
        latent_pose_reference=torch.zeros(1, layout.latent_dim),
        transl_reference=torch.zeros(1, 3),
        expression_reference=torch.zeros(1, expression.shape[1]),
    )

    expected_pose = torch.sum((latent_pose * weights.delta_pose) ** 2, dim=1)
    expected_trans = torch.sum((transl * weights.delta_trans) ** 2, dim=1)
    expected_expr = torch.sum((expression * weights.delta_expr) ** 2, dim=1)

    assert torch.allclose(result.loss_terms["deltaP"], expected_pose)
    assert torch.allclose(result.loss_terms["deltaT"], expected_trans)
    assert torch.allclose(result.loss_terms["deltaE"], expected_expr)
