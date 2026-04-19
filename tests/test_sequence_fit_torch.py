import importlib
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moshpp.optim.frame_fit_torch import make_stageii_latent_layout
from moshpp.transformed_lm_torch import build_marker_attachment


def _load_sequence_fit_module():
    try:
        spec = importlib.util.find_spec("moshpp.optim.sequence_fit_torch")
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "moshpp.optim.sequence_fit_torch is not available on main yet"
    return importlib.import_module("moshpp.optim.sequence_fit_torch")


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


def _make_minimal_problem(num_frames=6):
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
    true_transl = torch.stack(
        [torch.tensor([0.05 * t, -0.03 * t, 0.02 * t], dtype=torch.float32) for t in range(num_frames)],
        dim=0,
    )
    marker_observations = canonical_markers.unsqueeze(0) + true_transl[:, None, :]
    return canonical_markers, marker_attachment, layout, true_transl, marker_observations


def test_fit_stageii_sequence_torch_recovers_translation_on_synthetic_sequence():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, true_transl, marker_observations = _make_minimal_problem(num_frames=8)

    result = module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        weights=module.TorchSequenceFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=250, lr=0.08, optimizer="adam"),
    )

    assert result.transl.shape == true_transl.shape
    assert result.predicted_markers.shape == marker_observations.shape
    assert torch.allclose(result.transl, true_transl, atol=2e-2)
    assert torch.allclose(result.predicted_markers, marker_observations, atol=2e-2)


def test_fit_stageii_sequence_torch_respects_visible_mask_and_marker_data_weights():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, true_transl, marker_observations = _make_minimal_problem(num_frames=2)

    outlier = torch.tensor([12.0, 12.0, 12.0], dtype=torch.float32)
    marker_observations = marker_observations.clone()
    marker_observations[0, -1] = outlier
    marker_observations[1, -2] = outlier

    visible_mask = torch.ones(marker_observations.shape[:2], dtype=torch.bool)
    visible_mask[0, -1] = False
    marker_data_weights = torch.ones(marker_observations.shape[:2], dtype=torch.float32)
    marker_data_weights[1, -2] = 0.0

    result = module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=visible_mask,
        marker_data_weights=marker_data_weights,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        weights=module.TorchSequenceFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=250, lr=0.08, optimizer="adam"),
    )

    assert torch.allclose(result.transl, true_transl, atol=2e-2)


def test_fit_stageii_sequence_torch_temporal_second_order_term_smooths_translation():
    module = _load_sequence_fit_module()
    torch.manual_seed(42)
    canonical_markers, marker_attachment, layout, true_transl, clean_observations = _make_minimal_problem(num_frames=12)
    noisy_observations = clean_observations + 0.03 * torch.randn_like(clean_observations)

    common_kwargs = dict(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=noisy_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(noisy_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(noisy_observations.shape[0], 3),
        options=module.TorchSequenceFitOptions(max_iters=300, lr=0.07, optimizer="adam"),
    )

    no_smooth = module.fit_stageii_sequence_torch(
        weights=module.TorchSequenceFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            temporal_accel=0.0,
        ),
        **common_kwargs,
    )
    with_smooth = module.fit_stageii_sequence_torch(
        weights=module.TorchSequenceFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            temporal_accel=5.0,
        ),
        **common_kwargs,
    )

    accel_no_smooth = no_smooth.transl[2:] - 2.0 * no_smooth.transl[1:-1] + no_smooth.transl[:-2]
    accel_with_smooth = with_smooth.transl[2:] - 2.0 * with_smooth.transl[1:-1] + with_smooth.transl[:-2]

    no_smooth_norm = torch.mean(torch.linalg.norm(accel_no_smooth, dim=1))
    with_smooth_norm = torch.mean(torch.linalg.norm(accel_with_smooth, dim=1))

    assert with_smooth_norm < no_smooth_norm
    assert torch.allclose(with_smooth.transl.mean(dim=0), true_transl.mean(dim=0), atol=8e-2)


def test_fit_stageii_sequence_torch_delta_trans_regularization_keeps_solution_near_seed_under_outlier():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, true_transl, marker_observations = _make_minimal_problem(num_frames=6)
    noisy = marker_observations.clone()
    noisy[3:] += torch.tensor([0.30, -0.20, 0.10], dtype=torch.float32)

    common_kwargs = dict(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=noisy,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(noisy.shape[0], layout.latent_dim),
        transl_init=true_transl.clone(),
        options=module.TorchSequenceFitOptions(max_iters=200, lr=0.06, optimizer="adam"),
    )

    unconstrained = module.fit_stageii_sequence_torch(
        weights=module.TorchSequenceFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            temporal_accel=0.0,
            delta_pose=0.0,
            delta_trans=0.0,
            delta_expr=0.0,
        ),
        **common_kwargs,
    )
    constrained = module.fit_stageii_sequence_torch(
        weights=module.TorchSequenceFitWeights(
            data=1.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            temporal_accel=0.0,
            delta_pose=0.0,
            delta_trans=4.0,
            delta_expr=0.0,
        ),
        **common_kwargs,
    )

    unconstrained_err = torch.mean(torch.linalg.norm(unconstrained.transl - true_transl, dim=1))
    constrained_err = torch.mean(torch.linalg.norm(constrained.transl - true_transl, dim=1))

    assert constrained_err < unconstrained_err


def test_fit_stageii_sequence_torch_uses_explicit_references_instead_of_init_defaults():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=3)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["latent_pose_reference"] = kwargs["latent_pose_reference"].detach().clone()
            seen["transl_reference"] = kwargs["transl_reference"].detach().clone()
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    latent_pose_reference = torch.full((marker_observations.shape[0], layout.latent_dim), 7.0, dtype=torch.float32)
    transl_reference = torch.full((marker_observations.shape[0], 3), 9.0, dtype=torch.float32)

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        latent_pose_reference=latent_pose_reference,
        transl_reference=transl_reference,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert torch.allclose(seen["latent_pose_reference"], latent_pose_reference)
    assert torch.allclose(seen["transl_reference"], transl_reference)


def test_fit_stageii_sequence_torch_passes_explicit_transl_velocity_reference():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=3)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["transl_velocity_reference"] = kwargs["transl_velocity_reference"].detach().clone()
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    transl_velocity_reference = torch.tensor([[9.0, 8.0, 7.0]], dtype=torch.float32)

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        transl_velocity_reference=transl_velocity_reference,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            transl_velocity=1.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert torch.allclose(seen["transl_velocity_reference"], transl_velocity_reference)


def test_fit_stageii_sequence_torch_passes_explicit_velocity_reference_window():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=4)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["velocity_reference"] = kwargs["velocity_reference"].detach().clone()
            seen["velocity_reference_index"] = kwargs["velocity_reference_index"]
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    velocity_reference = torch.zeros(3, layout.latent_dim, dtype=torch.float32)
    velocity_reference[:, 0] = torch.tensor([9.0, 9.0, 10.0], dtype=torch.float32)

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        velocity_reference=velocity_reference,
        velocity_reference_index=2,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=1.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert torch.allclose(seen["velocity_reference"], velocity_reference)
    assert seen["velocity_reference_index"] == 2


def test_fit_stageii_sequence_torch_passes_explicit_transl_velocity_reference_index():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=4)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["transl_velocity_reference_index"] = kwargs["transl_velocity_reference_index"]
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        transl_velocity_reference=torch.tensor([[9.0, 8.0, 7.0]], dtype=torch.float32),
        transl_velocity_reference_index=2,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            transl_velocity=1.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert seen["transl_velocity_reference_index"] == 2


def test_fit_stageii_sequence_torch_passes_explicit_transl_boundary_reference_and_index():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=4)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["transl_boundary_reference"] = kwargs["transl_boundary_reference"].detach().clone()
            seen["transl_boundary_reference_index"] = kwargs["transl_boundary_reference_index"]
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "seamT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    transl_boundary_reference = torch.tensor([[9.0, 8.0, 7.0]], dtype=torch.float32)

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        transl_boundary_reference=transl_boundary_reference,
        transl_boundary_reference_index=2,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            boundary_transl_seam=1.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert torch.allclose(seen["transl_boundary_reference"], transl_boundary_reference)
    assert seen["transl_boundary_reference_index"] == 2


def test_fit_stageii_sequence_torch_passes_explicit_transl_velocity_reference_window():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=4)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["transl_velocity_reference"] = kwargs["transl_velocity_reference"].detach().clone()
            seen["transl_velocity_reference_index"] = kwargs["transl_velocity_reference_index"]
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    transl_velocity_reference = torch.tensor(
        [[9.0, 8.0, 7.0], [9.0, 8.0, 7.0], [10.0, 8.0, 7.0]],
        dtype=torch.float32,
    )

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        transl_velocity_reference=transl_velocity_reference,
        transl_velocity_reference_index=2,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            transl_velocity=1.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert torch.allclose(seen["transl_velocity_reference"], transl_velocity_reference)
    assert seen["transl_velocity_reference_index"] == 2


def test_fit_stageii_sequence_torch_passes_explicit_transl_boundary_reference_window():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=5)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["transl_boundary_reference"] = kwargs["transl_boundary_reference"].detach().clone()
            seen["transl_boundary_reference_index"] = kwargs["transl_boundary_reference_index"]
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "seamT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    transl_boundary_reference = torch.tensor(
        [[9.0, 8.0, 7.0], [10.0, 8.0, 7.0], [14.0, 8.0, 7.0]],
        dtype=torch.float32,
    )

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        transl_boundary_reference=transl_boundary_reference,
        transl_boundary_reference_index=2,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            boundary_transl_seam=1.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert torch.allclose(seen["transl_boundary_reference"], transl_boundary_reference)
    assert seen["transl_boundary_reference_index"] == 2


def test_fit_stageii_sequence_torch_passes_explicit_full_transl_velocity_reference():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=4)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["transl_velocity_reference"] = kwargs["transl_velocity_reference"].detach().clone()
            seen["transl_velocity_reference_index"] = kwargs["transl_velocity_reference_index"]
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    transl_velocity_reference = torch.tensor(
        [
            [9.0, 8.0, 7.0],
            [10.0, 8.0, 7.0],
            [12.0, 8.0, 7.0],
            [15.0, 8.0, 7.0],
        ],
        dtype=torch.float32,
    )

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        transl_velocity_reference=transl_velocity_reference,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            transl_velocity=1.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert torch.allclose(seen["transl_velocity_reference"], transl_velocity_reference)
    assert seen["transl_velocity_reference_index"] is None


def test_fit_stageii_sequence_torch_passes_explicit_full_velocity_reference():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=4)

    seen = {}

    class RecordingEvaluator:
        def __call__(self, **kwargs):
            seen["velocity_reference"] = kwargs["velocity_reference"].detach().clone()
            seen["velocity_reference_index"] = kwargs["velocity_reference_index"]
            total = kwargs["latent_pose"].sum() * 0.0 + kwargs["transl"].sum() * 0.0
            num_frames = kwargs["latent_pose"].shape[0]
            predicted_markers = kwargs["marker_observations"]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    velocity_reference = torch.zeros((4, layout.latent_dim), dtype=torch.float32)
    velocity_reference[:, 0] = torch.tensor([9.0, 11.0, 14.0, 18.0], dtype=torch.float32)

    module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        velocity_reference=velocity_reference,
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=1.0,
            transl_velocity=0.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=0, lr=0.01, optimizer="adam"),
        evaluator=RecordingEvaluator(),
    )

    assert torch.allclose(seen["velocity_reference"], velocity_reference)
    assert seen["velocity_reference_index"] is None


def test_fit_stageii_sequence_torch_masks_inactive_face_and_hand_latents():
    module = _load_sequence_fit_module()
    canonical_markers, marker_attachment, layout, _, marker_observations = _make_minimal_problem(num_frames=3)

    class GradientEvaluator:
        def __call__(self, **kwargs):
            latent_pose = kwargs["latent_pose"]
            transl = kwargs["transl"]
            predicted_markers = kwargs["marker_observations"]
            total = (
                latent_pose[:, 0].sum()
                + latent_pose[:, layout.jaw_slice].sum()
                + latent_pose[:, layout.left_hand_coeff_slice].sum()
                + transl.sum() * 0.0
            )
            num_frames = latent_pose.shape[0]
            body_output = DummyBodyOutput(
                vertices=predicted_markers.detach().clone(),
                joints=torch.zeros(num_frames, 3, 3, dtype=predicted_markers.dtype),
            )
            terms = {
                "data": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseB": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseH": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "poseF": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "expr": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "velo": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "veloT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "seamT": torch.zeros(num_frames, dtype=predicted_markers.dtype),
                "accel": torch.zeros(num_frames, dtype=predicted_markers.dtype),
            }
            fullpose = torch.zeros(num_frames, 165, dtype=predicted_markers.dtype)
            return total, terms, fullpose, body_output, predicted_markers

    result = module.fit_stageii_sequence_torch(
        body_model=TranslOnlyBodyModel(canonical_markers),
        betas=torch.zeros(1, 10),
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=ZeroPosePrior(63),
        layout=layout,
        latent_pose_init=torch.zeros(marker_observations.shape[0], layout.latent_dim),
        transl_init=torch.zeros(marker_observations.shape[0], 3),
        weights=module.TorchSequenceFitWeights(
            data=0.0,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
            temporal_accel=0.0,
        ),
        options=module.TorchSequenceFitOptions(max_iters=1, lr=0.1, optimizer="adam"),
        optimize_fingers=False,
        optimize_face=False,
        evaluator=GradientEvaluator(),
    )

    assert torch.all(result.latent_pose[:, 0] < 0.0)
    assert torch.allclose(result.latent_pose[:, layout.jaw_slice], torch.zeros_like(result.latent_pose[:, layout.jaw_slice]))
    assert torch.allclose(
        result.latent_pose[:, layout.left_hand_coeff_slice],
        torch.zeros_like(result.latent_pose[:, layout.left_hand_coeff_slice]),
    )
