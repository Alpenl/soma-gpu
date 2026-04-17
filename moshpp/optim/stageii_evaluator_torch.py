import torch

from moshpp.transformed_lm_torch import decode_marker_attachment


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


class StageIIFrameEvaluator(torch.nn.Module):
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
        weights,
        velocity_reference,
    ):
        fullpose = decode_stageii_latent_pose(latent_pose, self.layout, hand_pca=self.hand_pca)
        body_output = self.wrapper(fullpose=fullpose, betas=betas, transl=transl, expression=expression)
        predicted_markers = decode_marker_attachment(marker_attachment, body_output.vertices[0])
        marker_observations = marker_observations.to(device=predicted_markers.device, dtype=predicted_markers.dtype)

        data_term = torch.sum(((predicted_markers - marker_observations) * weights.data) ** 2)
        pose_term = torch.sum((self.pose_prior(_select_prior_input(fullpose, self.pose_prior)) * weights.pose_body) ** 2)

        hand_term = latent_pose.new_zeros(())
        if self.optimize_fingers:
            hand_term = _slice_norm_square(latent_pose, self.layout.hand_ids(), weights.pose_hand)

        face_term = latent_pose.new_zeros(())
        expr_term = latent_pose.new_zeros(())
        if self.optimize_face:
            face_term = _slice_norm_square(latent_pose, self.layout.face_ids(), weights.pose_face)
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
