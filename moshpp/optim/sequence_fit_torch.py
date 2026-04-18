from dataclasses import dataclass
from typing import Optional

import torch

from moshpp.models.smplx_torch_wrapper import SmplxTorchWrapper
from moshpp.optim.frame_fit_torch import encode_stageii_fullpose
from moshpp.optim.sequence_evaluator_torch import (
    build_stageii_sequence_evaluator,
    evaluate_stageii_sequence,
)


@dataclass
class TorchSequenceFitWeights:
    data: float
    pose_body: float
    pose_hand: float
    pose_face: float
    expr: float
    velocity: float
    transl_velocity: float = 0.0
    boundary_transl_seam: float = 0.0
    temporal_accel: float = 0.0
    delta_pose: float = 0.0
    delta_trans: float = 0.0
    delta_expr: float = 0.0


@dataclass
class TorchSequenceFitOptions:
    max_iters: int = 200
    lr: float = 1e-1
    optimizer: str = "adam"
    history_size: int = 100
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    max_eval: Optional[int] = None


@dataclass
class TorchSequenceFitResult:
    latent_pose: torch.Tensor
    fullpose: torch.Tensor
    transl: torch.Tensor
    expression: Optional[torch.Tensor]
    predicted_markers: torch.Tensor
    vertices: torch.Tensor
    joints: torch.Tensor
    loss_terms: dict


def _as_hand_pca_spec(hand_pca, *, device, dtype):
    if hand_pca is None:
        return None
    return hand_pca.to(device=device, dtype=dtype)


def _coerce_sequence_options(options):
    if isinstance(options, TorchSequenceFitOptions):
        return options
    return TorchSequenceFitOptions(
        max_iters=int(getattr(options, "max_iters", getattr(options, "refine_iters", 200))),
        lr=float(getattr(options, "lr", getattr(options, "refine_lr", 1e-1))),
        optimizer=str(getattr(options, "optimizer", getattr(options, "refine_optimizer", "adam"))),
        history_size=int(getattr(options, "history_size", 100)),
        tolerance_grad=float(getattr(options, "tolerance_grad", 1e-7)),
        tolerance_change=float(getattr(options, "tolerance_change", 1e-9)),
        max_eval=getattr(options, "max_eval", getattr(options, "refine_max_eval", None)),
    )


def _coerce_sequence_weights(weights):
    if isinstance(weights, TorchSequenceFitWeights):
        return weights
    return TorchSequenceFitWeights(
        data=float(weights.data),
        pose_body=float(weights.pose_body),
        pose_hand=float(weights.pose_hand),
        pose_face=float(weights.pose_face),
        expr=float(weights.expr),
        velocity=float(weights.velocity),
        transl_velocity=float(getattr(weights, "transl_velocity", 0.0)),
        boundary_transl_seam=float(getattr(weights, "boundary_transl_seam", 0.0)),
        temporal_accel=float(getattr(weights, "temporal_accel", 0.0)),
        delta_pose=float(getattr(weights, "delta_pose", 0.0)),
        delta_trans=float(getattr(weights, "delta_trans", 0.0)),
        delta_expr=float(getattr(weights, "delta_expr", 0.0)),
    )


def _coerce_optional_reference(reference, *, device, dtype, feature_shape, num_frames, name):
    if reference is None:
        return None
    reference = torch.as_tensor(reference, dtype=dtype, device=device)
    if len(feature_shape) == 0:
        if reference.ndim == 1 and reference.shape[0] in (1, num_frames):
            return reference
        if reference.ndim == 2 and reference.shape[0] in (1, num_frames) and reference.shape[1] == 0:
            return reference
    elif reference.ndim == len(feature_shape) and reference.shape == feature_shape:
        return reference.reshape(1, *feature_shape)
    elif reference.ndim == len(feature_shape) + 1 and reference.shape[0] in (1, num_frames) and reference.shape[1:] == feature_shape:
        return reference
    raise ValueError(
        f"{name} must have shape {feature_shape}, (1, *{feature_shape}), or ({num_frames}, *{feature_shape}), got {tuple(reference.shape)}"
    )


def _coerce_optional_velocity_reference(reference, *, device, dtype, feature_shape, num_frames, name):
    if reference is None:
        return None
    reference = torch.as_tensor(reference, dtype=dtype, device=device)
    if len(feature_shape) == 0:
        if reference.ndim == 1 and 1 <= reference.shape[0] <= num_frames:
            return reference
        if reference.ndim == 2 and 1 <= reference.shape[0] <= num_frames and reference.shape[1] == 0:
            return reference
    elif reference.ndim == len(feature_shape) and reference.shape == feature_shape:
        return reference.reshape(1, *feature_shape)
    elif (
        reference.ndim == len(feature_shape) + 1
        and 1 <= reference.shape[0] <= num_frames
        and reference.shape[1:] == feature_shape
    ):
        return reference
    raise ValueError(
        f"{name} must have shape {feature_shape}, (1..{num_frames}, *{feature_shape}), got {tuple(reference.shape)}"
    )


def _coerce_optional_index(index, *, num_frames, name):
    if index is None:
        return None
    index = int(index)
    if index < 0 or index >= num_frames:
        raise ValueError(f"{name} must be in [0, {num_frames}), got {index}")
    return index


def _run_adam(*, params, closure_fn, max_iters, lr):
    if max_iters <= 0:
        return
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(max_iters):
        optimizer.zero_grad()
        loss = closure_fn()
        loss.backward()
        optimizer.step()


def _run_lbfgs(
    *,
    params,
    closure_fn,
    max_iters,
    lr,
    history_size,
    tolerance_grad,
    tolerance_change,
    max_eval,
):
    if max_iters <= 0:
        return
    optimizer = torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=max_iters,
        max_eval=max_eval,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = closure_fn()
        loss.backward()
        return loss

    optimizer.step(closure)


def _run_solver(*, options, params, closure_fn):
    optimizer_type = options.optimizer.lower()
    if optimizer_type == "adam":
        _run_adam(params=params, closure_fn=closure_fn, max_iters=options.max_iters, lr=options.lr)
        return
    if optimizer_type == "lbfgs":
        _run_lbfgs(
            params=params,
            closure_fn=closure_fn,
            max_iters=options.max_iters,
            lr=options.lr,
            history_size=options.history_size,
            tolerance_grad=options.tolerance_grad,
            tolerance_change=options.tolerance_change,
            max_eval=options.max_eval,
        )
        return
    raise ValueError(f"Unsupported optimizer: {options.optimizer}")


def fit_stageii_sequence_torch(
    *,
    body_model,
    wrapper=None,
    betas,
    marker_attachment,
    marker_observations,
    pose_prior,
    layout,
    weights,
    options,
    latent_pose_init=None,
    transl_init=None,
    hand_pca=None,
    fullpose_init=None,
    expression_init=None,
    latent_pose_reference=None,
    transl_reference=None,
    expression_reference=None,
    optimize_fingers=False,
    optimize_face=False,
    optimize_toes=False,
    velocity_reference=None,
    velocity_reference_index=None,
    transl_velocity_reference=None,
    transl_velocity_reference_index=None,
    transl_boundary_reference=None,
    transl_boundary_reference_index=None,
    visible_mask=None,
    marker_data_weights=None,
    evaluator=None,
):
    del optimize_toes
    marker_observations = torch.as_tensor(marker_observations, dtype=torch.float32)
    if marker_observations.ndim != 3 or marker_observations.shape[-1] != 3:
        raise ValueError(f"marker_observations must have shape (T, M, 3), got {tuple(marker_observations.shape)}")
    device = marker_observations.device
    num_frames, num_markers = marker_observations.shape[:2]

    if visible_mask is None:
        visible_mask = torch.ones((num_frames, num_markers), dtype=torch.bool, device=device)
    else:
        visible_mask = torch.as_tensor(visible_mask, dtype=torch.bool, device=device)
        if visible_mask.shape != (num_frames, num_markers):
            raise ValueError(f"visible_mask must have shape {(num_frames, num_markers)}, got {tuple(visible_mask.shape)}")
    visible_mask_float = visible_mask.to(dtype=marker_observations.dtype)
    marker_observations = torch.where(visible_mask[..., None], marker_observations, torch.zeros_like(marker_observations))

    if latent_pose_init is not None and fullpose_init is not None:
        raise ValueError("Specify either latent_pose_init or fullpose_init, not both.")

    hand_pca = _as_hand_pca_spec(hand_pca, device=device, dtype=marker_observations.dtype)
    if fullpose_init is not None:
        latent_pose_init = encode_stageii_fullpose(
            torch.as_tensor(fullpose_init, dtype=torch.float32, device=device),
            layout,
            hand_pca=hand_pca,
        )
    elif latent_pose_init is None:
        latent_pose_init = torch.zeros(num_frames, layout.latent_dim, dtype=torch.float32, device=device)
    else:
        latent_pose_init = torch.as_tensor(latent_pose_init, dtype=torch.float32, device=device)
    if latent_pose_init.shape != (num_frames, layout.latent_dim):
        raise ValueError(f"latent_pose_init must have shape {(num_frames, layout.latent_dim)}, got {tuple(latent_pose_init.shape)}")

    if transl_init is None:
        transl_init = torch.zeros(num_frames, 3, dtype=torch.float32, device=device)
    else:
        transl_init = torch.as_tensor(transl_init, dtype=torch.float32, device=device)
    if transl_init.shape != (num_frames, 3):
        raise ValueError(f"transl_init must have shape {(num_frames, 3)}, got {tuple(transl_init.shape)}")

    expression_param = None
    if expression_init is not None:
        expression_init = torch.as_tensor(expression_init, dtype=torch.float32, device=device)
        if expression_init.shape[0] != num_frames:
            raise ValueError(f"expression_init must have leading dimension {num_frames}, got {tuple(expression_init.shape)}")
        expression_param = expression_init.detach().clone().requires_grad_(True)

    options = _coerce_sequence_options(options)
    weights = _coerce_sequence_weights(weights)

    betas = torch.as_tensor(betas, dtype=torch.float32, device=device)
    marker_attachment = marker_attachment.to(device=device, dtype=torch.float32)
    if marker_data_weights is not None:
        marker_data_weights = torch.as_tensor(marker_data_weights, dtype=torch.float32, device=device)
    if velocity_reference is not None:
        velocity_reference = _coerce_optional_velocity_reference(
            velocity_reference,
            device=device,
            dtype=torch.float32,
            feature_shape=(layout.latent_dim,),
            num_frames=num_frames,
            name="velocity_reference",
        )
        velocity_reference = velocity_reference.detach().clone()
    velocity_reference_index = _coerce_optional_index(
        velocity_reference_index,
        num_frames=num_frames,
        name="velocity_reference_index",
    )
    transl_velocity_reference = _coerce_optional_velocity_reference(
        transl_velocity_reference,
        device=device,
        dtype=torch.float32,
        feature_shape=(3,),
        num_frames=num_frames,
        name="transl_velocity_reference",
    )
    if transl_velocity_reference is not None:
        transl_velocity_reference = transl_velocity_reference.detach().clone()
    transl_velocity_reference_index = _coerce_optional_index(
        transl_velocity_reference_index,
        num_frames=num_frames,
        name="transl_velocity_reference_index",
    )
    transl_boundary_reference = _coerce_optional_velocity_reference(
        transl_boundary_reference,
        device=device,
        dtype=torch.float32,
        feature_shape=(3,),
        num_frames=num_frames,
        name="transl_boundary_reference",
    )
    if transl_boundary_reference is not None:
        transl_boundary_reference = transl_boundary_reference.detach().clone()
    transl_boundary_reference_index = _coerce_optional_index(
        transl_boundary_reference_index,
        num_frames=num_frames,
        name="transl_boundary_reference_index",
    )

    latent_pose_reference = _coerce_optional_reference(
        latent_pose_reference,
        device=device,
        dtype=torch.float32,
        feature_shape=(layout.latent_dim,),
        num_frames=num_frames,
        name="latent_pose_reference",
    )
    if latent_pose_reference is None:
        latent_pose_reference = latent_pose_init.detach().clone()
    else:
        latent_pose_reference = latent_pose_reference.detach().clone()

    transl_reference = _coerce_optional_reference(
        transl_reference,
        device=device,
        dtype=torch.float32,
        feature_shape=(3,),
        num_frames=num_frames,
        name="transl_reference",
    )
    if transl_reference is None:
        transl_reference = transl_init.detach().clone()
    else:
        transl_reference = transl_reference.detach().clone()

    if expression_param is not None:
        expression_reference = _coerce_optional_reference(
            expression_reference,
            device=device,
            dtype=torch.float32,
            feature_shape=tuple(expression_param.shape[1:]),
            num_frames=num_frames,
            name="expression_reference",
        )
        if expression_reference is None:
            expression_reference = expression_init.detach().clone()
        else:
            expression_reference = expression_reference.detach().clone()
    else:
        expression_reference = None

    latent_pose_param = latent_pose_init.detach().clone().requires_grad_(True)
    transl_param = transl_init.detach().clone().requires_grad_(True)

    if wrapper is None:
        wrapper = SmplxTorchWrapper(body_model=body_model, surface_model_type=layout.surface_model_type)
    if evaluator is None:
        evaluator = build_stageii_sequence_evaluator(
            wrapper=wrapper,
            layout=layout,
            hand_pca=hand_pca,
            pose_prior=pose_prior,
            optimize_fingers=optimize_fingers,
            optimize_face=optimize_face,
        )

    params = [latent_pose_param, transl_param]
    if expression_param is not None:
        params.append(expression_param)

    def closure():
        evaluation = evaluate_stageii_sequence(
            evaluator=evaluator,
            latent_pose=latent_pose_param,
            transl=transl_param,
            expression=expression_param,
            betas=betas,
            marker_attachment=marker_attachment,
            marker_observations=marker_observations,
            visible_mask=visible_mask_float,
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
        return evaluation.total

    _run_solver(options=options, params=params, closure_fn=closure)

    evaluation = evaluate_stageii_sequence(
        evaluator=evaluator,
        latent_pose=latent_pose_param,
        transl=transl_param,
        expression=expression_param,
        betas=betas,
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        visible_mask=visible_mask_float,
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
    return TorchSequenceFitResult(
        latent_pose=latent_pose_param.detach(),
        fullpose=evaluation.fullpose.detach(),
        transl=transl_param.detach(),
        expression=expression_param.detach() if expression_param is not None else None,
        predicted_markers=evaluation.predicted_markers.detach(),
        vertices=evaluation.body_output.vertices.detach(),
        joints=evaluation.body_output.joints.detach(),
        loss_terms={k: v.detach() for k, v in evaluation.loss_terms.items()},
    )
