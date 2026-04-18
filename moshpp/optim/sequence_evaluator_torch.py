from types import SimpleNamespace

import torch

from moshpp.optim.frame_fit_torch import decode_stageii_latent_pose
from moshpp.transformed_lm_torch import decode_marker_attachment_batched


def _select_prior_input(fullpose, pose_prior):
    prior_dim = pose_prior.means.shape[-1]
    if prior_dim == fullpose.shape[1]:
        return fullpose
    if prior_dim == 63:
        return fullpose[:, 3:66]
    if prior_dim == 69:
        return fullpose[:, 3:72]
    raise ValueError(f"Unsupported prior dimension {prior_dim} for fullpose shape {tuple(fullpose.shape)}")


def _coerce_sequence_marker_data_weights(marker_data_weights, reference):
    if marker_data_weights is None:
        return None
    marker_data_weights = torch.as_tensor(
        marker_data_weights,
        dtype=reference.dtype,
        device=reference.device,
    )
    num_frames, num_markers = reference.shape[:2]
    if marker_data_weights.ndim == 1:
        if marker_data_weights.shape[0] != num_markers:
            raise ValueError(
                f"marker_data_weights must have shape ({num_markers},) or broadcastable time-major shape, got {tuple(marker_data_weights.shape)}"
            )
        return marker_data_weights[None, :, None]
    if marker_data_weights.ndim == 2:
        if marker_data_weights.shape == (num_frames, num_markers):
            return marker_data_weights[:, :, None]
        if marker_data_weights.shape[0] == num_markers and marker_data_weights.shape[1] in (1, 3):
            return marker_data_weights[None]
        raise ValueError(
            f"marker_data_weights 2D shape must be ({num_frames}, {num_markers}) or ({num_markers}, 1/3), got {tuple(marker_data_weights.shape)}"
        )
    if marker_data_weights.ndim == 3:
        if marker_data_weights.shape[:2] != (num_frames, num_markers) or marker_data_weights.shape[2] not in (1, 3):
            raise ValueError(
                f"marker_data_weights 3D shape must be ({num_frames}, {num_markers}, 1/3), got {tuple(marker_data_weights.shape)}"
            )
        return marker_data_weights
    raise ValueError(f"Unsupported marker_data_weights ndim={marker_data_weights.ndim}")


def _coerce_sequence_reference(reference, like):
    if reference is None:
        return None
    reference = torch.as_tensor(reference, dtype=like.dtype, device=like.device)
    if reference.shape == like.shape:
        return reference
    if reference.ndim == like.ndim and reference.shape[0] == 1 and reference.shape[1:] == like.shape[1:]:
        return reference.expand(like.shape[0], *reference.shape[1:])
    raise ValueError(
        f"reference must match {tuple(like.shape)} or broadcast from leading dim 1, got {tuple(reference.shape)}"
    )


def _coerce_velocity_reference(reference, like):
    if reference is None:
        return None
    reference = torch.as_tensor(reference, dtype=like.dtype, device=like.device)
    if reference.ndim == like.ndim - 1 and reference.shape == like.shape[1:]:
        return reference[None]
    if reference.ndim == like.ndim and 1 <= reference.shape[0] <= like.shape[0] and reference.shape[1:] == like.shape[1:]:
        return reference
    raise ValueError(
        f"velocity_reference must have trailing shape {tuple(like.shape[1:])} with leading dim in [1, {like.shape[0]}], got {tuple(reference.shape)}"
    )


def _coerce_reference_index(reference_index, *, num_frames, name):
    if reference_index is None:
        return None
    reference_index = int(reference_index)
    if reference_index < 0 or reference_index >= num_frames:
        raise ValueError(f"{name} must be in [0, {num_frames}), got {reference_index}")
    return reference_index


def _velocity_term(sequence, *, weight, reference=None, reference_index=None, index_name):
    term = sequence.new_zeros((sequence.shape[0],))
    if weight == 0.0:
        return term

    reduce_dims = tuple(range(1, sequence.ndim))
    if sequence.shape[0] >= 2:
        diffs = sequence[1:] - sequence[:-1]
        term[1:] = torch.sum((diffs * weight) ** 2, dim=reduce_dims)

    if reference is None:
        return term

    reference = _coerce_velocity_reference(reference, sequence)
    if reference.shape[0] == 1:
        seam_idx = _coerce_reference_index(
            reference_index,
            num_frames=sequence.shape[0],
            name=index_name,
        )
        if seam_idx is None:
            seam_idx = 0
        term[seam_idx] = torch.sum(((sequence[seam_idx] - reference[0]) * weight) ** 2)
        return term

    if reference.shape[0] == sequence.shape[0]:
        term[0] = torch.sum(((sequence[0] - reference[0]) * weight) ** 2)
        if sequence.shape[0] > 1:
            reference_diffs = reference[1:] - reference[:-1]
            term[1:] = torch.sum(((diffs - reference_diffs) * weight) ** 2, dim=reduce_dims)
        return term

    seam_idx = _coerce_reference_index(
        reference_index,
        num_frames=sequence.shape[0],
        name=index_name,
    )
    if seam_idx is None:
        seam_idx = 0
    local_frame_count = reference.shape[0] - 1
    window_end = seam_idx + local_frame_count
    if window_end > sequence.shape[0]:
        raise ValueError(
            f"{index_name.rsplit('_', 1)[0]} local window exceeds sequence length: "
            f"start={seam_idx}, window={local_frame_count}, num_frames={sequence.shape[0]}"
        )
    reference_diffs = reference[1:] - reference[:-1]
    term[seam_idx] = torch.sum((((sequence[seam_idx] - reference[0]) - reference_diffs[0]) * weight) ** 2)
    if local_frame_count > 1:
        local_diffs = sequence[seam_idx + 1 : window_end] - sequence[seam_idx : window_end - 1]
        term[seam_idx + 1 : window_end] = torch.sum(
            ((local_diffs - reference_diffs[1:]) * weight) ** 2,
            dim=reduce_dims,
        )
    return term


def _boundary_term(sequence, *, weight, reference=None, reference_index=None, index_name):
    term = sequence.new_zeros((sequence.shape[0],))
    if weight == 0.0 or reference is None:
        return term

    reference = _coerce_velocity_reference(reference, sequence)
    seam_idx = _coerce_reference_index(
        reference_index,
        num_frames=sequence.shape[0],
        name=index_name,
    )
    if seam_idx is None:
        seam_idx = 0
    reduce_dims = tuple(range(1, sequence.ndim))
    if reference.shape[0] == 1:
        term[seam_idx] = torch.sum(((sequence[seam_idx] - reference[0]) * weight) ** 2)
        return term

    window_end = seam_idx + reference.shape[0]
    if window_end > sequence.shape[0]:
        raise ValueError(
            f"{index_name.rsplit('_', 1)[0]} local window exceeds sequence length: "
            f"start={seam_idx}, window={reference.shape[0]}, num_frames={sequence.shape[0]}"
        )
    term[seam_idx:window_end] = torch.sum(
        ((sequence[seam_idx:window_end] - reference) * weight) ** 2,
        dim=reduce_dims,
    )
    return term


class StageIISequenceEvaluator(torch.nn.Module):
    def __init__(
        self,
        *,
        wrapper,
        layout,
        hand_pca,
        pose_prior,
        optimize_fingers,
        optimize_face,
    ):
        super().__init__()
        self.wrapper = wrapper
        self.layout = layout
        self.hand_pca = hand_pca
        self.pose_prior = pose_prior
        self.optimize_fingers = optimize_fingers
        self.optimize_face = optimize_face

    def forward(
        self,
        *,
        latent_pose,
        transl,
        expression,
        betas,
        marker_attachment,
        marker_observations,
        visible_mask,
        marker_data_weights,
        weights,
        velocity_reference,
        velocity_reference_index=None,
        transl_velocity_reference=None,
        transl_velocity_reference_index=None,
        transl_boundary_reference=None,
        transl_boundary_reference_index=None,
        latent_pose_reference=None,
        transl_reference=None,
        expression_reference=None,
    ):
        fullpose = decode_stageii_latent_pose(latent_pose, self.layout, hand_pca=self.hand_pca)
        body_output = self.wrapper(fullpose=fullpose, betas=betas, transl=transl, expression=expression)
        predicted_markers = decode_marker_attachment_batched(marker_attachment, body_output.vertices)

        residual = predicted_markers - marker_observations
        if visible_mask is not None:
            residual = residual * visible_mask[..., None]
        marker_data_weights = _coerce_sequence_marker_data_weights(marker_data_weights, residual)
        if marker_data_weights is not None:
            residual = residual * marker_data_weights

        data_term = torch.sum((residual * weights.data) ** 2, dim=(1, 2))
        pose_term = torch.sum((self.pose_prior(_select_prior_input(fullpose, self.pose_prior)) * weights.pose_body) ** 2, dim=1)

        hand_term = latent_pose.new_zeros((latent_pose.shape[0],))
        if self.optimize_fingers and self.layout.left_hand_coeff_slice is not None:
            hand_ids = self.layout.hand_ids()
            hand_term = torch.sum((latent_pose[:, hand_ids] * weights.pose_hand) ** 2, dim=1)

        face_term = latent_pose.new_zeros((latent_pose.shape[0],))
        expr_term = latent_pose.new_zeros((latent_pose.shape[0],))
        if self.optimize_face and self.layout.jaw_slice is not None:
            face_ids = self.layout.face_ids()
            face_term = torch.sum((latent_pose[:, face_ids] * weights.pose_face) ** 2, dim=1)
            if expression is not None:
                expr_term = torch.sum((expression * weights.expr) ** 2, dim=1)

        velocity_term = latent_pose.new_zeros((latent_pose.shape[0],))
        velocity_weight = float(getattr(weights, "velocity", 0.0))
        if velocity_weight != 0.0:
            velocity_term = _velocity_term(
                latent_pose,
                weight=velocity_weight,
                reference=velocity_reference,
                reference_index=velocity_reference_index,
                index_name="velocity_reference_index",
            )

        transl_velocity_term = latent_pose.new_zeros((latent_pose.shape[0],))
        transl_velocity_weight = float(getattr(weights, "transl_velocity", 0.0))
        if transl_velocity_weight != 0.0:
            transl_velocity_term = _velocity_term(
                transl,
                weight=transl_velocity_weight,
                reference=transl_velocity_reference,
                reference_index=transl_velocity_reference_index,
                index_name="transl_velocity_reference_index",
            )

        boundary_transl_seam_term = latent_pose.new_zeros((latent_pose.shape[0],))
        boundary_transl_seam_weight = float(getattr(weights, "boundary_transl_seam", 0.0))
        if boundary_transl_seam_weight != 0.0:
            boundary_transl_seam_term = _boundary_term(
                transl,
                weight=boundary_transl_seam_weight,
                reference=transl_boundary_reference,
                reference_index=transl_boundary_reference_index,
                index_name="transl_boundary_reference_index",
            )

        accel_term = latent_pose.new_zeros((latent_pose.shape[0],))
        temporal_accel = float(getattr(weights, "temporal_accel", 0.0))
        if temporal_accel != 0.0 and transl.shape[0] >= 3:
            accel = transl[2:] - (2.0 * transl[1:-1]) + transl[:-2]
            accel_term[2:] = torch.sum((accel * temporal_accel) ** 2, dim=1)

        terms = {
            "data": data_term,
            "poseB": pose_term,
            "poseH": hand_term,
            "poseF": face_term,
            "expr": expr_term,
            "velo": velocity_term,
            "veloT": transl_velocity_term,
            "seamT": boundary_transl_seam_term,
            "accel": accel_term,
        }

        delta_pose_weight = float(getattr(weights, "delta_pose", 0.0))
        if delta_pose_weight != 0.0:
            latent_pose_reference = _coerce_sequence_reference(latent_pose_reference, latent_pose)
            terms["deltaP"] = torch.sum(((latent_pose - latent_pose_reference) * delta_pose_weight) ** 2, dim=1)

        delta_trans_weight = float(getattr(weights, "delta_trans", 0.0))
        if delta_trans_weight != 0.0:
            transl_reference = _coerce_sequence_reference(transl_reference, transl)
            terms["deltaT"] = torch.sum(((transl - transl_reference) * delta_trans_weight) ** 2, dim=1)

        delta_expr_weight = float(getattr(weights, "delta_expr", 0.0))
        if delta_expr_weight != 0.0 and expression is not None:
            expression_reference = _coerce_sequence_reference(expression_reference, expression)
            terms["deltaE"] = torch.sum(((expression - expression_reference) * delta_expr_weight) ** 2, dim=1)

        total = sum(value.sum() for value in terms.values())
        return total, terms, fullpose, body_output, predicted_markers


def build_stageii_sequence_evaluator(
    *,
    wrapper,
    layout,
    hand_pca,
    pose_prior,
    optimize_fingers,
    optimize_face,
    compile_module=False,
    compile_mode="reduce-overhead",
    compile_fullgraph=False,
):
    evaluator = StageIISequenceEvaluator(
        wrapper=wrapper,
        layout=layout,
        hand_pca=hand_pca,
        pose_prior=pose_prior,
        optimize_fingers=optimize_fingers,
        optimize_face=optimize_face,
    )
    if compile_module and hasattr(torch, "compile"):
        return torch.compile(evaluator, mode=compile_mode, fullgraph=compile_fullgraph)
    return evaluator


def evaluate_stageii_sequence(
    *,
    evaluator,
    latent_pose,
    transl,
    expression,
    betas,
    marker_attachment,
    marker_observations,
    visible_mask,
    marker_data_weights,
    weights,
    velocity_reference,
    velocity_reference_index=None,
    transl_velocity_reference=None,
    transl_velocity_reference_index=None,
    transl_boundary_reference=None,
    transl_boundary_reference_index=None,
    latent_pose_reference=None,
    transl_reference=None,
    expression_reference=None,
):
    total, terms, fullpose, body_output, predicted_markers = evaluator(
        latent_pose=latent_pose,
        transl=transl,
        expression=expression,
        betas=betas,
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=visible_mask,
        marker_data_weights=marker_data_weights,
        weights=weights,
        velocity_reference=velocity_reference,
        velocity_reference_index=velocity_reference_index,
        transl_velocity_reference=transl_velocity_reference,
        transl_velocity_reference_index=transl_velocity_reference_index,
        transl_boundary_reference=transl_boundary_reference,
        transl_boundary_reference_index=transl_boundary_reference_index,
        latent_pose_reference=latent_pose_reference,
        transl_reference=transl_reference,
        expression_reference=expression_reference,
    )
    return SimpleNamespace(
        total=total,
        loss_terms=terms,
        fullpose=fullpose,
        body_output=body_output,
        predicted_markers=predicted_markers,
    )
