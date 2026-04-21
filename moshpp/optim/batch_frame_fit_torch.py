from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from moshpp.models.smplx_torch_wrapper import SmplxTorchWrapper
from moshpp.optim.frame_fit_torch import (
    TorchFrameFitOptions,
    TorchFrameFitWeights,
    decode_stageii_latent_pose,
)
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


def _coerce_marker_data_weights(marker_data_weights, *, reference):
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
                f"marker_data_weights must have shape ({num_markers},) or a time-major broadcastable shape, got {tuple(marker_data_weights.shape)}"
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


@dataclass
class BatchedTorchFrameFitWeights:
    data: torch.Tensor
    pose_body: torch.Tensor
    pose_hand: torch.Tensor
    pose_face: torch.Tensor
    expr: torch.Tensor
    velocity: torch.Tensor

    def replace(self, **updates):
        data = {
            "data": self.data,
            "pose_body": self.pose_body,
            "pose_hand": self.pose_hand,
            "pose_face": self.pose_face,
            "expr": self.expr,
            "velocity": self.velocity,
        }
        data.update(updates)
        return BatchedTorchFrameFitWeights(**data)


@dataclass
class TorchBatchFrameFitResult:
    latent_pose: torch.Tensor
    fullpose: torch.Tensor
    transl: torch.Tensor
    expression: Optional[torch.Tensor]
    predicted_markers: torch.Tensor
    vertices: torch.Tensor
    joints: torch.Tensor
    loss_terms: dict
    fallback_mask: torch.Tensor
    solver_diagnostics: dict


class StageIIBatchFrameEvaluator(torch.nn.Module):
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
    ):
        fullpose = decode_stageii_latent_pose(latent_pose, self.layout, hand_pca=self.hand_pca)
        body_output = self.wrapper(fullpose=fullpose, betas=betas, transl=transl, expression=expression)
        predicted_markers = decode_marker_attachment_batched(marker_attachment, body_output.vertices)

        residual = predicted_markers - marker_observations
        if visible_mask is not None:
            residual = residual * visible_mask[..., None]
        marker_data_weights = _coerce_marker_data_weights(marker_data_weights, reference=residual)
        if marker_data_weights is not None:
            residual = residual * marker_data_weights

        data_term = torch.sum((residual * weights.data[:, None, None]) ** 2, dim=(1, 2))
        pose_term = torch.sum(
            (self.pose_prior(_select_prior_input(fullpose, self.pose_prior)) * weights.pose_body[:, None]) ** 2,
            dim=1,
        )

        hand_term = latent_pose.new_zeros((latent_pose.shape[0],))
        if self.optimize_fingers and self.layout.left_hand_coeff_slice is not None:
            hand_ids = self.layout.hand_ids()
            hand_term = torch.sum((latent_pose[:, hand_ids] * weights.pose_hand[:, None]) ** 2, dim=1)

        face_term = latent_pose.new_zeros((latent_pose.shape[0],))
        expr_term = latent_pose.new_zeros((latent_pose.shape[0],))
        if self.optimize_face and self.layout.jaw_slice is not None:
            face_ids = self.layout.face_ids()
            face_term = torch.sum((latent_pose[:, face_ids] * weights.pose_face[:, None]) ** 2, dim=1)
            if expression is not None:
                expr_term = torch.sum((expression * weights.expr[:, None]) ** 2, dim=1)

        velocity_term = latent_pose.new_zeros((latent_pose.shape[0],))
        if velocity_reference is not None:
            velocity_reference = torch.as_tensor(
                velocity_reference,
                dtype=latent_pose.dtype,
                device=latent_pose.device,
            )
            if velocity_reference.shape != latent_pose.shape:
                raise ValueError(
                    f"velocity_reference must have shape {tuple(latent_pose.shape)}, got {tuple(velocity_reference.shape)}"
                )
            velocity_term = torch.sum(
                ((latent_pose - velocity_reference) * weights.velocity[:, None]) ** 2,
                dim=1,
            )

        terms = {
            "data": data_term,
            "poseB": pose_term,
            "poseH": hand_term,
            "poseF": face_term,
            "expr": expr_term,
            "velo": velocity_term,
        }
        total = sum(value.sum() for value in terms.values())
        return total, terms, fullpose, body_output, predicted_markers


def build_stageii_batch_frame_evaluator(
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
    evaluator = StageIIBatchFrameEvaluator(
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


def _as_hand_pca_spec(hand_pca, *, device, dtype):
    if hand_pca is None:
        return None
    return hand_pca.to(device=device, dtype=dtype)


def _coerce_batched_weights(weights, *, batch_size, device, dtype):
    if isinstance(weights, TorchFrameFitWeights):
        weights = [weights] * batch_size
    if not isinstance(weights, Sequence) or len(weights) != batch_size:
        raise ValueError(f"weights must be a TorchFrameFitWeights or a sequence of length {batch_size}")

    def _field(name):
        return torch.tensor(
            [float(getattr(weight, name)) for weight in weights],
            dtype=dtype,
            device=device,
        )

    return BatchedTorchFrameFitWeights(
        data=_field("data"),
        pose_body=_field("pose_body"),
        pose_hand=_field("pose_hand"),
        pose_face=_field("pose_face"),
        expr=_field("expr"),
        velocity=_field("velocity"),
    )


def _pack_values(latent_pose, transl, expression):
    parts = [latent_pose.reshape(latent_pose.shape[0], -1), transl.reshape(transl.shape[0], -1)]
    if expression is not None:
        parts.append(expression.reshape(expression.shape[0], -1))
    return torch.cat(parts, dim=1)


def _unpack_values(flat, *, latent_dim, expression_dim):
    latent_pose = flat[:, :latent_dim]
    transl = flat[:, latent_dim : latent_dim + 3]
    expression = None
    if expression_dim > 0:
        expression = flat[:, latent_dim + 3 : latent_dim + 3 + expression_dim]
    return latent_pose, transl, expression


def _pack_grads(latent_pose, transl, expression):
    latent_grad = latent_pose.grad if latent_pose.grad is not None else torch.zeros_like(latent_pose)
    transl_grad = transl.grad if transl.grad is not None else torch.zeros_like(transl)
    parts = [latent_grad.reshape(latent_grad.shape[0], -1), transl_grad.reshape(transl_grad.shape[0], -1)]
    if expression is not None:
        expr_grad = expression.grad if expression.grad is not None else torch.zeros_like(expression)
        parts.append(expr_grad.reshape(expr_grad.shape[0], -1))
    return torch.cat(parts, dim=1)


def _flat_active_mask(*, latent_dim, expression_dim, active_pose_ids, optimize_expression, device, dtype):
    mask = torch.zeros(latent_dim + 3 + expression_dim, dtype=dtype, device=device)
    if active_pose_ids:
        mask[active_pose_ids] = 1.0
    mask[latent_dim : latent_dim + 3] = 1.0
    if optimize_expression and expression_dim > 0:
        mask[latent_dim + 3 :] = 1.0
    return mask


def _copy_flat_params_(flat, *, latent_pose_param, transl_param, expression_param):
    latent_dim = latent_pose_param.shape[1]
    expression_dim = 0 if expression_param is None else expression_param.shape[1]
    latent_pose, transl, expression = _unpack_values(flat, latent_dim=latent_dim, expression_dim=expression_dim)
    with torch.no_grad():
        latent_pose_param.copy_(latent_pose)
        transl_param.copy_(transl)
        if expression_param is not None:
            expression_param.copy_(expression)


def _evaluate_state(
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
    compute_grad,
):
    context = torch.enable_grad() if compute_grad else torch.no_grad()
    with context:
        if compute_grad:
            for tensor in (latent_pose, transl, expression):
                if tensor is not None and getattr(tensor, "grad", None) is not None:
                    tensor.grad.zero_()
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
        )
        per_frame_total = sum(value for value in terms.values())
        grad_flat = None
        if compute_grad:
            total.backward()
            grad_flat = _pack_grads(latent_pose, transl, expression)
    return {
        "per_frame_total": per_frame_total.detach(),
        "loss_terms": {key: value.detach() for key, value in terms.items()},
        "fullpose": fullpose.detach(),
        "vertices": body_output.vertices.detach(),
        "joints": body_output.joints.detach(),
        "predicted_markers": predicted_markers.detach(),
        "grad_flat": grad_flat.detach() if grad_flat is not None else None,
    }


def _evaluate_candidate_losses(
    *,
    evaluator,
    candidate_flat,
    latent_dim,
    expression_dim,
    betas,
    marker_attachment,
    marker_observations,
    visible_mask,
    marker_data_weights,
    weights,
    velocity_reference,
):
    latent_pose, transl, expression = _unpack_values(
        candidate_flat,
        latent_dim=latent_dim,
        expression_dim=expression_dim,
    )
    result = _evaluate_state(
        evaluator=evaluator,
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
        compute_grad=False,
    )
    return result["per_frame_total"]


def _two_loop_direction(*, grad_flat, s_hist, y_hist, rho_hist, hist_count, active_mask, tolerance_change):
    direction = torch.zeros_like(grad_flat)
    for frame_idx in range(grad_flat.shape[0]):
        g = grad_flat[frame_idx]
        if g.abs().max().item() <= tolerance_change:
            continue
        count = int(hist_count[frame_idx].item())
        if count <= 0:
            direction[frame_idx] = -g
            continue

        q = g.clone()
        alpha_values = []
        for hist_idx in range(count - 1, -1, -1):
            s_vec = s_hist[frame_idx, hist_idx]
            y_vec = y_hist[frame_idx, hist_idx]
            rho = rho_hist[frame_idx, hist_idx]
            alpha = rho * torch.dot(s_vec, q)
            q = q - alpha * y_vec
            alpha_values.append(alpha)

        last_s = s_hist[frame_idx, count - 1]
        last_y = y_hist[frame_idx, count - 1]
        yy = torch.dot(last_y, last_y)
        if yy.abs().item() > 1e-12:
            gamma = torch.dot(last_s, last_y) / yy
        else:
            gamma = g.new_tensor(1.0)
        r = gamma * q

        for hist_idx in range(count):
            s_vec = s_hist[frame_idx, hist_idx]
            y_vec = y_hist[frame_idx, hist_idx]
            rho = rho_hist[frame_idx, hist_idx]
            beta = rho * torch.dot(y_vec, r)
            alpha = alpha_values[count - 1 - hist_idx]
            r = r + s_vec * (alpha - beta)

        direction[frame_idx] = -r

    direction = direction * active_mask[None, :]
    directional = torch.sum(direction * grad_flat, dim=1)
    bad_direction = ~torch.isfinite(directional) | (directional >= 0)
    if bool(bad_direction.any()):
        direction[bad_direction] = -grad_flat[bad_direction]
    return direction * active_mask[None, :]


def _append_history(*, s_hist, y_hist, rho_hist, hist_count, s_new, y_new, accepted_mask):
    history_size = s_hist.shape[1]
    accepted_ids = torch.nonzero(accepted_mask, as_tuple=False).flatten().tolist()
    for frame_idx in accepted_ids:
        s_vec = s_new[frame_idx]
        y_vec = y_new[frame_idx]
        curvature = torch.dot(s_vec, y_vec)
        if not torch.isfinite(curvature) or curvature.item() <= 1e-12:
            continue
        rho = 1.0 / curvature
        count = int(hist_count[frame_idx].item())
        if count < history_size:
            insert_idx = count
            hist_count[frame_idx] = count + 1
        else:
            s_hist[frame_idx, :-1] = s_hist[frame_idx, 1:].clone()
            y_hist[frame_idx, :-1] = y_hist[frame_idx, 1:].clone()
            rho_hist[frame_idx, :-1] = rho_hist[frame_idx, 1:].clone()
            insert_idx = history_size - 1
        s_hist[frame_idx, insert_idx] = s_vec
        y_hist[frame_idx, insert_idx] = y_vec
        rho_hist[frame_idx, insert_idx] = rho


def _run_batched_stage(
    *,
    latent_pose_param,
    transl_param,
    expression_param,
    evaluator,
    betas,
    marker_attachment,
    marker_observations,
    visible_mask,
    marker_data_weights,
    weights,
    velocity_reference,
    active_mask,
    max_iters,
    lr,
    history_size,
    tolerance_grad,
    tolerance_change,
    armijo_c1,
    armijo_beta,
    max_backtracks,
    fallback_mask,
    fallback_reasons,
):
    if max_iters <= 0:
        return fallback_mask, fallback_reasons

    batch_size = latent_pose_param.shape[0]
    latent_dim = latent_pose_param.shape[1]
    expression_dim = 0 if expression_param is None else expression_param.shape[1]
    active_mask = active_mask.to(device=latent_pose_param.device, dtype=latent_pose_param.dtype)

    s_hist = torch.zeros(
        batch_size,
        history_size,
        latent_dim + 3 + expression_dim,
        dtype=latent_pose_param.dtype,
        device=latent_pose_param.device,
    )
    y_hist = torch.zeros_like(s_hist)
    rho_hist = torch.zeros(batch_size, history_size, dtype=latent_pose_param.dtype, device=latent_pose_param.device)
    hist_count = torch.zeros(batch_size, dtype=torch.long, device=latent_pose_param.device)

    current_eval = _evaluate_state(
        evaluator=evaluator,
        latent_pose=latent_pose_param,
        transl=transl_param,
        expression=expression_param,
        betas=betas,
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=visible_mask,
        marker_data_weights=marker_data_weights,
        weights=weights,
        velocity_reference=velocity_reference,
        compute_grad=True,
    )
    best_flat = _pack_values(latent_pose_param, transl_param, expression_param).detach().clone()
    best_losses = current_eval["per_frame_total"].clone()

    done_mask = torch.zeros(batch_size, dtype=torch.bool, device=latent_pose_param.device)
    for _ in range(max_iters):
        grad_flat = current_eval["grad_flat"] * active_mask[None, :]
        grad_inf = torch.max(torch.abs(grad_flat), dim=1).values
        done_mask |= grad_inf <= tolerance_grad
        active = ~(done_mask | fallback_mask)
        if not bool(active.any()):
            break

        direction = _two_loop_direction(
            grad_flat=grad_flat,
            s_hist=s_hist,
            y_hist=y_hist,
            rho_hist=rho_hist,
            hist_count=hist_count,
            active_mask=active_mask,
            tolerance_change=tolerance_change,
        )
        direction_inf = torch.max(torch.abs(direction), dim=1).values
        done_mask |= active & (direction_inf <= tolerance_change)
        active = ~(done_mask | fallback_mask)
        if not bool(active.any()):
            break

        current_flat = _pack_values(latent_pose_param, transl_param, expression_param).detach()
        directional = torch.sum(direction * grad_flat, dim=1)

        trial_alpha = torch.full(
            (batch_size,),
            float(lr),
            dtype=latent_pose_param.dtype,
            device=latent_pose_param.device,
        )
        next_flat = current_flat.clone()
        accepted = torch.zeros(batch_size, dtype=torch.bool, device=latent_pose_param.device)
        remaining = active.clone()

        for _ in range(max_backtracks):
            trial_flat = current_flat + trial_alpha[:, None] * direction
            trial_losses = _evaluate_candidate_losses(
                evaluator=evaluator,
                candidate_flat=trial_flat,
                latent_dim=latent_dim,
                expression_dim=expression_dim,
                betas=betas,
                marker_attachment=marker_attachment,
                marker_observations=marker_observations,
                visible_mask=visible_mask,
                marker_data_weights=marker_data_weights,
                weights=weights,
                velocity_reference=velocity_reference,
            )
            rhs = current_eval["per_frame_total"] + armijo_c1 * trial_alpha * directional
            can_accept = remaining & torch.isfinite(trial_losses) & (trial_losses <= rhs)
            if bool(can_accept.any()):
                next_flat[can_accept] = trial_flat[can_accept]
                accepted |= can_accept
                remaining &= ~can_accept
            if not bool(remaining.any()):
                break
            trial_alpha[remaining] *= armijo_beta

        if bool(remaining.any()):
            fallback_mask |= remaining
            for frame_idx in torch.nonzero(remaining, as_tuple=False).flatten().tolist():
                if fallback_reasons[frame_idx] is None:
                    fallback_reasons[frame_idx] = "line_search_failed"

        if not bool(accepted.any()):
            break

        prev_flat = current_flat
        prev_grad = grad_flat
        previous_losses = current_eval["per_frame_total"]
        _copy_flat_params_(
            next_flat,
            latent_pose_param=latent_pose_param,
            transl_param=transl_param,
            expression_param=expression_param,
        )
        current_eval = _evaluate_state(
            evaluator=evaluator,
            latent_pose=latent_pose_param,
            transl=transl_param,
            expression=expression_param,
            betas=betas,
            marker_attachment=marker_attachment,
            marker_observations=marker_observations,
            visible_mask=visible_mask,
            marker_data_weights=marker_data_weights,
            weights=weights,
            velocity_reference=velocity_reference,
            compute_grad=True,
        )
        current_flat = _pack_values(latent_pose_param, transl_param, expression_param).detach()
        improved = current_eval["per_frame_total"] < best_losses
        if bool(improved.any()):
            best_losses[improved] = current_eval["per_frame_total"][improved]
            best_flat[improved] = current_flat[improved]

        s_new = (current_flat - prev_flat) * active_mask[None, :]
        y_new = (current_eval["grad_flat"] * active_mask[None, :]) - prev_grad
        _append_history(
            s_hist=s_hist,
            y_hist=y_hist,
            rho_hist=rho_hist,
            hist_count=hist_count,
            s_new=s_new,
            y_new=y_new,
            accepted_mask=accepted,
        )

        step_inf = torch.max(torch.abs(s_new), dim=1).values
        loss_delta = torch.abs(previous_losses - current_eval["per_frame_total"])
        done_mask |= accepted & ((step_inf <= tolerance_change) | (loss_delta <= tolerance_change))

    _copy_flat_params_(
        best_flat,
        latent_pose_param=latent_pose_param,
        transl_param=transl_param,
        expression_param=expression_param,
    )
    return fallback_mask, fallback_reasons


def fit_stageii_frames_batched_torch(
    *,
    body_model,
    wrapper=None,
    betas,
    marker_attachment,
    marker_observations,
    pose_prior,
    layout,
    latent_pose_init,
    transl_init,
    weights,
    options,
    hand_pca=None,
    expression_init=None,
    optimize_fingers=False,
    optimize_face=False,
    optimize_toes=False,
    velocity_reference=None,
    rigid_init=True,
    warmup_pose_scales=(1.0,),
    visible_mask=None,
    marker_data_weights=None,
    evaluator=None,
    compile_module=False,
    compile_mode="reduce-overhead",
    compile_fullgraph=False,
    armijo_c1=1e-4,
    armijo_beta=0.5,
    max_backtracks=8,
):
    marker_observations = torch.as_tensor(marker_observations, dtype=torch.float32)
    if marker_observations.ndim != 3 or marker_observations.shape[-1] != 3:
        raise ValueError(
            f"marker_observations must have shape (B, M, 3), got {tuple(marker_observations.shape)}"
        )
    batch_size, num_markers = marker_observations.shape[:2]
    device = marker_observations.device

    if visible_mask is None:
        visible_mask = torch.ones((batch_size, num_markers), dtype=torch.bool, device=device)
    else:
        visible_mask = torch.as_tensor(visible_mask, dtype=torch.bool, device=device)
        if visible_mask.shape != (batch_size, num_markers):
            raise ValueError(
                f"visible_mask must have shape {(batch_size, num_markers)}, got {tuple(visible_mask.shape)}"
            )

    latent_pose_init = torch.as_tensor(latent_pose_init, dtype=torch.float32, device=device)
    transl_init = torch.as_tensor(transl_init, dtype=torch.float32, device=device)
    if latent_pose_init.shape != (batch_size, layout.latent_dim):
        raise ValueError(
            f"latent_pose_init must have shape {(batch_size, layout.latent_dim)}, got {tuple(latent_pose_init.shape)}"
        )
    if transl_init.shape != (batch_size, 3):
        raise ValueError(f"transl_init must have shape {(batch_size, 3)}, got {tuple(transl_init.shape)}")

    expression_param = None
    if expression_init is not None:
        expression_init = torch.as_tensor(expression_init, dtype=torch.float32, device=device)
        if expression_init.shape[0] != batch_size:
            raise ValueError(
                f"expression_init must have leading dimension {batch_size}, got {tuple(expression_init.shape)}"
            )
        expression_param = expression_init.clone().detach().requires_grad_(True)

    latent_pose_param = latent_pose_init.clone().detach().requires_grad_(True)
    transl_param = transl_init.clone().detach().requires_grad_(True)
    betas = torch.as_tensor(betas, dtype=torch.float32, device=device)
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)

    if wrapper is None:
        wrapper = SmplxTorchWrapper(body_model=body_model, surface_model_type=layout.surface_model_type)
    hand_pca = _as_hand_pca_spec(hand_pca, device=device, dtype=latent_pose_param.dtype)
    marker_attachment = marker_attachment.to(device=device, dtype=torch.float32)
    batched_weights = _coerce_batched_weights(
        weights,
        batch_size=batch_size,
        device=device,
        dtype=latent_pose_param.dtype,
    )
    if velocity_reference is not None:
        velocity_reference = torch.as_tensor(velocity_reference, dtype=torch.float32, device=device)
        if velocity_reference.shape != latent_pose_param.shape:
            raise ValueError(
                f"velocity_reference must have shape {tuple(latent_pose_param.shape)}, got {tuple(velocity_reference.shape)}"
            )

    if evaluator is None:
        evaluator = build_stageii_batch_frame_evaluator(
            wrapper=wrapper,
            layout=layout,
            hand_pca=hand_pca,
            pose_prior=pose_prior,
            optimize_fingers=optimize_fingers,
            optimize_face=optimize_face,
            compile_module=compile_module,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
        )

    fallback_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    fallback_reasons = [None] * batch_size

    latent_dim = layout.latent_dim
    expression_dim = 0 if expression_param is None else expression_param.shape[1]

    if rigid_init:
        rigid_weights = batched_weights.replace(
            pose_body=torch.zeros_like(batched_weights.pose_body),
            pose_hand=torch.zeros_like(batched_weights.pose_hand),
            pose_face=torch.zeros_like(batched_weights.pose_face),
            expr=torch.zeros_like(batched_weights.expr),
            velocity=torch.zeros_like(batched_weights.velocity),
        )
        rigid_mask = _flat_active_mask(
            latent_dim=latent_dim,
            expression_dim=expression_dim,
            active_pose_ids=list(range(layout.root_slice.start, layout.root_slice.stop)),
            optimize_expression=False,
            device=device,
            dtype=latent_pose_param.dtype,
        )
        fallback_mask, fallback_reasons = _run_batched_stage(
            latent_pose_param=latent_pose_param,
            transl_param=transl_param,
            expression_param=expression_param,
            evaluator=evaluator,
            betas=betas,
            marker_attachment=marker_attachment,
            marker_observations=marker_observations,
            visible_mask=visible_mask,
            marker_data_weights=marker_data_weights,
            weights=rigid_weights,
            velocity_reference=None,
            active_mask=rigid_mask,
            max_iters=options.rigid_iters,
            lr=options.rigid_lr,
            history_size=options.history_size,
            tolerance_grad=options.tolerance_grad,
            tolerance_change=options.tolerance_change,
            armijo_c1=armijo_c1,
            armijo_beta=armijo_beta,
            max_backtracks=max_backtracks,
            fallback_mask=fallback_mask,
            fallback_reasons=fallback_reasons,
        )

    for scale in warmup_pose_scales:
        warmup_weights = batched_weights.replace(
            pose_body=batched_weights.pose_body * float(scale),
            pose_hand=torch.zeros_like(batched_weights.pose_hand),
            pose_face=torch.zeros_like(batched_weights.pose_face),
            expr=torch.zeros_like(batched_weights.expr),
        )
        warmup_mask = _flat_active_mask(
            latent_dim=latent_dim,
            expression_dim=expression_dim,
            active_pose_ids=layout.warmup_ids(optimize_toes=optimize_toes),
            optimize_expression=False,
            device=device,
            dtype=latent_pose_param.dtype,
        )
        fallback_mask, fallback_reasons = _run_batched_stage(
            latent_pose_param=latent_pose_param,
            transl_param=transl_param,
            expression_param=expression_param,
            evaluator=evaluator,
            betas=betas,
            marker_attachment=marker_attachment,
            marker_observations=marker_observations,
            visible_mask=visible_mask,
            marker_data_weights=marker_data_weights,
            weights=warmup_weights,
            velocity_reference=velocity_reference,
            active_mask=warmup_mask,
            max_iters=options.warmup_iters,
            lr=options.warmup_lr,
            history_size=options.history_size,
            tolerance_grad=options.tolerance_grad,
            tolerance_change=options.tolerance_change,
            armijo_c1=armijo_c1,
            armijo_beta=armijo_beta,
            max_backtracks=max_backtracks,
            fallback_mask=fallback_mask,
            fallback_reasons=fallback_reasons,
        )

    refine_mask = _flat_active_mask(
        latent_dim=latent_dim,
        expression_dim=expression_dim,
        active_pose_ids=layout.refine_ids(
            optimize_fingers=optimize_fingers,
            optimize_face=optimize_face,
            optimize_toes=optimize_toes,
        ),
        optimize_expression=optimize_face and expression_param is not None,
        device=device,
        dtype=latent_pose_param.dtype,
    )
    fallback_mask, fallback_reasons = _run_batched_stage(
        latent_pose_param=latent_pose_param,
        transl_param=transl_param,
        expression_param=expression_param,
        evaluator=evaluator,
        betas=betas,
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=visible_mask,
        marker_data_weights=marker_data_weights,
        weights=batched_weights,
        velocity_reference=velocity_reference,
        active_mask=refine_mask,
        max_iters=options.refine_iters,
        lr=options.refine_lr,
        history_size=options.history_size,
        tolerance_grad=options.tolerance_grad,
        tolerance_change=options.tolerance_change,
        armijo_c1=armijo_c1,
        armijo_beta=armijo_beta,
        max_backtracks=max_backtracks,
        fallback_mask=fallback_mask,
        fallback_reasons=fallback_reasons,
    )

    final_eval = _evaluate_state(
        evaluator=evaluator,
        latent_pose=latent_pose_param.detach(),
        transl=transl_param.detach(),
        expression=expression_param.detach() if expression_param is not None else None,
        betas=betas,
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=visible_mask,
        marker_data_weights=marker_data_weights,
        weights=batched_weights,
        velocity_reference=velocity_reference,
        compute_grad=False,
    )
    return TorchBatchFrameFitResult(
        latent_pose=latent_pose_param.detach(),
        fullpose=final_eval["fullpose"],
        transl=transl_param.detach(),
        expression=expression_param.detach() if expression_param is not None else None,
        predicted_markers=final_eval["predicted_markers"],
        vertices=final_eval["vertices"],
        joints=final_eval["joints"],
        loss_terms=final_eval["loss_terms"],
        fallback_mask=fallback_mask.detach(),
        solver_diagnostics={"fallback_reasons": fallback_reasons},
    )
