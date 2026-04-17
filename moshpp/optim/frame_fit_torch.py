from dataclasses import dataclass
from typing import Optional

import torch

from moshpp.models.smplx_torch_wrapper import SmplxTorchWrapper
from moshpp.transformed_lm_torch import decode_marker_attachment


@dataclass
class HandPcaSpec:
    left_components: torch.Tensor
    right_components: torch.Tensor
    left_mean: torch.Tensor
    right_mean: torch.Tensor

    def to(self, *, device=None, dtype=None):
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return HandPcaSpec(
            left_components=self.left_components.to(**kwargs),
            right_components=self.right_components.to(**kwargs),
            left_mean=self.left_mean.to(**kwargs),
            right_mean=self.right_mean.to(**kwargs),
        )


@dataclass
class StageIILatentPoseLayout:
    surface_model_type: str
    dof_per_hand: int
    latent_dim: int
    fullpose_dim: int
    root_slice: slice
    body_slice: slice
    jaw_slice: Optional[slice]
    leye_slice: Optional[slice]
    reye_slice: Optional[slice]
    left_hand_coeff_slice: Optional[slice]
    right_hand_coeff_slice: Optional[slice]
    left_hand_full_slice: Optional[slice]
    right_hand_full_slice: Optional[slice]

    def body_ids(self):
        return list(range(self.body_slice.start, self.body_slice.stop))

    def warmup_ids(self, *, optimize_toes):
        ids = list(range(self.root_slice.start, self.root_slice.stop))
        ids.extend(self.body_ids())
        if not optimize_toes:
            ids = [idx for idx in ids if idx < 30 or idx >= 36]
        return sorted(set(ids))

    def refine_ids(self, *, optimize_fingers, optimize_face, optimize_toes):
        ids = self.warmup_ids(optimize_toes=optimize_toes)
        if optimize_face and self.jaw_slice is not None:
            ids.extend(range(self.jaw_slice.start, self.jaw_slice.stop))
        if optimize_fingers and self.left_hand_coeff_slice is not None:
            ids.extend(range(self.left_hand_coeff_slice.start, self.left_hand_coeff_slice.stop))
            ids.extend(range(self.right_hand_coeff_slice.start, self.right_hand_coeff_slice.stop))
        return sorted(set(ids))

    def face_ids(self):
        if self.jaw_slice is None:
            return []
        return list(range(self.jaw_slice.start, self.jaw_slice.stop))

    def hand_ids(self):
        if self.left_hand_coeff_slice is None:
            return []
        ids = list(range(self.left_hand_coeff_slice.start, self.left_hand_coeff_slice.stop))
        ids.extend(range(self.right_hand_coeff_slice.start, self.right_hand_coeff_slice.stop))
        return ids


@dataclass
class TorchFrameFitWeights:
    data: float
    pose_body: float
    pose_hand: float
    pose_face: float
    expr: float
    velocity: float


@dataclass
class TorchFrameFitOptions:
    rigid_iters: int = 20
    warmup_iters: int = 20
    refine_iters: int = 20
    rigid_lr: float = 1.0
    warmup_lr: float = 1.0
    refine_lr: float = 1.0


@dataclass
class TorchFrameFitResult:
    latent_pose: torch.Tensor
    fullpose: torch.Tensor
    transl: torch.Tensor
    expression: Optional[torch.Tensor]
    predicted_markers: torch.Tensor
    vertices: torch.Tensor
    joints: torch.Tensor
    loss_terms: dict


def make_stageii_latent_layout(surface_model_type, dof_per_hand, optimize_fingers, optimize_face):
    del optimize_fingers, optimize_face
    if surface_model_type != "smplx":
        raise NotImplementedError(f"Unsupported surface_model_type: {surface_model_type}")

    latent_dim = 75 + (2 * dof_per_hand)
    return StageIILatentPoseLayout(
        surface_model_type=surface_model_type,
        dof_per_hand=dof_per_hand,
        latent_dim=latent_dim,
        fullpose_dim=165,
        root_slice=slice(0, 3),
        body_slice=slice(3, 66),
        jaw_slice=slice(66, 69),
        leye_slice=slice(69, 72),
        reye_slice=slice(72, 75),
        left_hand_coeff_slice=slice(75, 75 + dof_per_hand),
        right_hand_coeff_slice=slice(75 + dof_per_hand, 75 + 2 * dof_per_hand),
        left_hand_full_slice=slice(75, 120),
        right_hand_full_slice=slice(120, 165),
    )


def _as_hand_pca_spec(hand_pca, *, device, dtype):
    if hand_pca is None:
        return None
    return hand_pca.to(device=device, dtype=dtype)


def decode_stageii_latent_pose(latent_pose, layout, hand_pca=None):
    latent_pose = torch.as_tensor(latent_pose)
    batch_size = latent_pose.shape[0]
    fullpose = torch.zeros(
        batch_size,
        layout.fullpose_dim,
        dtype=latent_pose.dtype,
        device=latent_pose.device,
    )
    fullpose[:, layout.root_slice] = latent_pose[:, layout.root_slice]
    fullpose[:, layout.body_slice] = latent_pose[:, layout.body_slice]

    if layout.jaw_slice is not None:
        fullpose[:, layout.jaw_slice] = latent_pose[:, layout.jaw_slice]
    if layout.leye_slice is not None:
        fullpose[:, layout.leye_slice] = latent_pose[:, layout.leye_slice]
    if layout.reye_slice is not None:
        fullpose[:, layout.reye_slice] = latent_pose[:, layout.reye_slice]

    if hand_pca is not None:
        left_coeffs = latent_pose[:, layout.left_hand_coeff_slice]
        right_coeffs = latent_pose[:, layout.right_hand_coeff_slice]
        fullpose[:, layout.left_hand_full_slice] = hand_pca.left_mean + left_coeffs @ hand_pca.left_components
        fullpose[:, layout.right_hand_full_slice] = hand_pca.right_mean + right_coeffs @ hand_pca.right_components

    return fullpose


def encode_stageii_fullpose(fullpose, layout, hand_pca=None):
    fullpose = torch.as_tensor(fullpose)
    if fullpose.ndim == 1:
        fullpose = fullpose.unsqueeze(0)

    batch_size = fullpose.shape[0]
    latent_pose = torch.zeros(
        batch_size,
        layout.latent_dim,
        dtype=fullpose.dtype,
        device=fullpose.device,
    )
    latent_pose[:, layout.root_slice] = fullpose[:, layout.root_slice]
    latent_pose[:, layout.body_slice] = fullpose[:, layout.body_slice]

    if layout.jaw_slice is not None:
        latent_pose[:, layout.jaw_slice] = fullpose[:, layout.jaw_slice]
    if layout.leye_slice is not None:
        latent_pose[:, layout.leye_slice] = fullpose[:, layout.leye_slice]
    if layout.reye_slice is not None:
        latent_pose[:, layout.reye_slice] = fullpose[:, layout.reye_slice]

    if hand_pca is not None and layout.left_hand_coeff_slice is not None:
        left_delta = fullpose[:, layout.left_hand_full_slice] - hand_pca.left_mean
        right_delta = fullpose[:, layout.right_hand_full_slice] - hand_pca.right_mean
        left_pinv = torch.linalg.pinv(hand_pca.left_components)
        right_pinv = torch.linalg.pinv(hand_pca.right_components)
        latent_pose[:, layout.left_hand_coeff_slice] = left_delta @ left_pinv
        latent_pose[:, layout.right_hand_coeff_slice] = right_delta @ right_pinv

    return latent_pose


def _select_prior_input(fullpose, pose_prior):
    prior_dim = pose_prior.means.shape[-1]
    if prior_dim == fullpose.shape[1]:
        return fullpose
    if prior_dim == 63:
        return fullpose[:, 3:66]
    if prior_dim == 69:
        return fullpose[:, 3:72]
    raise ValueError(f"Unsupported prior dimension {prior_dim} for fullpose shape {tuple(fullpose.shape)}")


def _slice_norm_square(values, slice_or_ids, weight):
    if not slice_or_ids:
        return values.new_zeros(())
    subset = values[:, slice_or_ids]
    return torch.sum((subset * weight) ** 2)


def _compute_weighted_terms(
    *,
    wrapper,
    layout,
    hand_pca,
    latent_pose,
    transl,
    expression,
    betas,
    marker_attachment,
    marker_observations,
    pose_prior,
    weights,
    optimize_fingers,
    optimize_face,
    velocity_reference,
):
    fullpose = decode_stageii_latent_pose(latent_pose, layout, hand_pca=hand_pca)
    body_output = wrapper(fullpose=fullpose, betas=betas, transl=transl, expression=expression)
    predicted_markers = decode_marker_attachment(marker_attachment, body_output.vertices[0])
    marker_observations = marker_observations.to(device=predicted_markers.device, dtype=predicted_markers.dtype)

    data_term = torch.sum(((predicted_markers - marker_observations) * weights.data) ** 2)
    pose_term = torch.sum((pose_prior(_select_prior_input(fullpose, pose_prior)) * weights.pose_body) ** 2)

    hand_term = latent_pose.new_zeros(())
    if optimize_fingers:
        hand_term = _slice_norm_square(latent_pose, layout.hand_ids(), weights.pose_hand)

    face_term = latent_pose.new_zeros(())
    expr_term = latent_pose.new_zeros(())
    if optimize_face:
        face_term = _slice_norm_square(latent_pose, layout.face_ids(), weights.pose_face)
        if expression is not None:
            expr_term = torch.sum((expression * weights.expr) ** 2)

    velocity_term = latent_pose.new_zeros(())
    if velocity_reference is not None:
        velocity_reference = velocity_reference.to(device=latent_pose.device, dtype=latent_pose.dtype)
        velocity_term = torch.sum(((latent_pose - velocity_reference) * weights.velocity) ** 2)

    terms = {
        "data": data_term,
        "poseB": pose_term,
        "poseH": hand_term,
        "poseF": face_term,
        "expr": expr_term,
        "velo": velocity_term,
    }
    total = sum(terms.values())
    return total, terms, fullpose, body_output, predicted_markers


def _run_lbfgs(
    *,
    params,
    closure_fn,
    active_pose_ids,
    latent_pose_param,
    max_iters,
    lr,
):
    if max_iters <= 0:
        return
    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iters, line_search_fn="strong_wolfe")
    active_mask = torch.zeros_like(latent_pose_param)
    if active_pose_ids:
        active_mask[:, active_pose_ids] = 1.0

    def closure():
        optimizer.zero_grad()
        loss = closure_fn()
        loss.backward()
        if latent_pose_param.grad is not None:
            latent_pose_param.grad.mul_(active_mask)
        return loss

    optimizer.step(closure)


def fit_stageii_frame_torch(
    *,
    body_model,
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
):
    latent_pose_init = torch.as_tensor(latent_pose_init, dtype=torch.float32)
    transl_init = torch.as_tensor(transl_init, dtype=torch.float32)
    betas = torch.as_tensor(betas, dtype=torch.float32)
    marker_observations = torch.as_tensor(marker_observations, dtype=torch.float32)

    device = marker_observations.device
    latent_pose_param = latent_pose_init.clone().detach().to(device).requires_grad_(True)
    transl_param = transl_init.clone().detach().to(device).requires_grad_(True)
    expression_param = None
    if expression_init is not None:
        expression_param = (
            torch.as_tensor(expression_init, dtype=torch.float32, device=device).clone().detach().requires_grad_(True)
        )

    betas = betas.to(device)
    hand_pca = _as_hand_pca_spec(hand_pca, device=device, dtype=latent_pose_param.dtype)

    wrapper = SmplxTorchWrapper(body_model=body_model, surface_model_type=layout.surface_model_type)

    def closure_for(current_weights):
        def compute():
            total, _, _, _, _ = _compute_weighted_terms(
                wrapper=wrapper,
                layout=layout,
                hand_pca=hand_pca,
                latent_pose=latent_pose_param,
                transl=transl_param,
                expression=expression_param,
                betas=betas,
                marker_attachment=marker_attachment,
                marker_observations=marker_observations,
                pose_prior=pose_prior,
                weights=current_weights,
                optimize_fingers=optimize_fingers,
                optimize_face=optimize_face,
                velocity_reference=velocity_reference,
            )
            return total

        return compute

    if rigid_init:
        rigid_weights = TorchFrameFitWeights(
            data=weights.data,
            pose_body=0.0,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=0.0,
        )
        _run_lbfgs(
            params=[latent_pose_param, transl_param],
            closure_fn=closure_for(rigid_weights),
            active_pose_ids=list(range(layout.root_slice.start, layout.root_slice.stop)),
            latent_pose_param=latent_pose_param,
            max_iters=options.rigid_iters,
            lr=options.rigid_lr,
        )

    for scale in warmup_pose_scales:
        warmup_weights = TorchFrameFitWeights(
            data=weights.data,
            pose_body=weights.pose_body * scale,
            pose_hand=0.0,
            pose_face=0.0,
            expr=0.0,
            velocity=weights.velocity,
        )
        _run_lbfgs(
            params=[latent_pose_param, transl_param],
            closure_fn=closure_for(warmup_weights),
            active_pose_ids=layout.warmup_ids(optimize_toes=optimize_toes),
            latent_pose_param=latent_pose_param,
            max_iters=options.warmup_iters,
            lr=options.warmup_lr,
        )

    refine_params = [latent_pose_param, transl_param]
    if optimize_face and expression_param is not None:
        refine_params.append(expression_param)
    _run_lbfgs(
        params=refine_params,
        closure_fn=closure_for(weights),
        active_pose_ids=layout.refine_ids(
            optimize_fingers=optimize_fingers,
            optimize_face=optimize_face,
            optimize_toes=optimize_toes,
        ),
        latent_pose_param=latent_pose_param,
        max_iters=options.refine_iters,
        lr=options.refine_lr,
    )

    _, term_tensors, fullpose, body_output, predicted_markers = _compute_weighted_terms(
        wrapper=wrapper,
        layout=layout,
        hand_pca=hand_pca,
        latent_pose=latent_pose_param,
        transl=transl_param,
        expression=expression_param,
        betas=betas,
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=pose_prior,
        weights=weights,
        optimize_fingers=optimize_fingers,
        optimize_face=optimize_face,
        velocity_reference=velocity_reference,
    )

    return TorchFrameFitResult(
        latent_pose=latent_pose_param.detach(),
        fullpose=fullpose.detach(),
        transl=transl_param.detach(),
        expression=expression_param.detach() if expression_param is not None else None,
        predicted_markers=predicted_markers.detach(),
        vertices=body_output.vertices.detach(),
        joints=body_output.joints.detach(),
        loss_terms={k: float(v.detach().cpu()) for k, v in term_tensors.items()},
    )
