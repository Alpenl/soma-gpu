from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from moshpp.models.smplx_torch_wrapper import SmplxTorchWrapper
from moshpp.transformed_lm_torch import decode_marker_attachment


@dataclass
class StageIITorchSmokeResult:
    vertices: torch.Tensor
    joints: torch.Tensor
    predicted_markers: torch.Tensor
    data_residual: torch.Tensor
    prior_residual: torch.Tensor


@dataclass
class StageIIFrameInputs:
    source_format: str
    surface_model_type: str
    fullpose: torch.Tensor
    betas: torch.Tensor
    transl: torch.Tensor
    expression: Optional[torch.Tensor]
    markers_latent: torch.Tensor
    latent_labels: List[str]
    marker_observations: torch.Tensor
    marker_labels: List[str]
    frame_idx: int


@dataclass
class MocapFrameData:
    source_format: str
    markers: torch.Tensor
    labels: List[str]
    frame_rate: float
    frame_idx: int


def _load_pickle_compat(path):
    with Path(path).open("rb") as handle:
        try:
            return pickle.load(handle)
        except UnicodeDecodeError:
            handle.seek(0)
            return pickle.load(handle, encoding="latin1")


def _to_string_list(labels):
    if labels is None:
        return []
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    return [str(label) for label in labels]


def _coerce_frame_matrix(values, frame_idx, *, name):
    array = np.asarray(values)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        if frame_idx < 0 or frame_idx >= array.shape[0]:
            raise IndexError(f"{name} frame_idx {frame_idx} out of range for shape {tuple(array.shape)}")
        return array[frame_idx : frame_idx + 1]
    raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(array.shape)}")


def _coerce_marker_frame(values, frame_idx, *, name):
    array = np.asarray(values)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if frame_idx < 0 or frame_idx >= array.shape[0]:
            raise IndexError(f"{name} frame_idx {frame_idx} out of range for shape {tuple(array.shape)}")
        return array[frame_idx]
    raise ValueError(f"{name} must be 2D or 3D, got shape {tuple(array.shape)}")


def _coerce_optional_expression(values, frame_idx):
    if values is None:
        return None
    return torch.as_tensor(_coerce_frame_matrix(values, frame_idx, name="expression"), dtype=torch.float32)


def _coerce_fullpose_to_smplx_layout(fullpose, surface_model_type):
    fullpose = np.asarray(fullpose, dtype=np.float32)
    pose_dim = fullpose.shape[1]
    if pose_dim == 165:
        return fullpose
    if surface_model_type == "smplh" and pose_dim == 156:
        zeros = np.zeros((fullpose.shape[0], 9), dtype=fullpose.dtype)
        return np.concatenate([fullpose[:, :66], zeros, fullpose[:, 66:]], axis=1)
    raise ValueError(
        f"Unsupported fullpose shape {tuple(fullpose.shape)} for surface_model_type={surface_model_type}"
    )


def _labels_for_frame(labels, frame_idx):
    if labels is None:
        return []
    if isinstance(labels, np.ndarray):
        if labels.ndim == 1:
            return _to_string_list(labels)
        if labels.ndim == 2:
            if frame_idx < 0 or frame_idx >= labels.shape[0]:
                raise IndexError(f"labels frame_idx {frame_idx} out of range for shape {tuple(labels.shape)}")
            return _to_string_list(labels[frame_idx])
    if isinstance(labels, (list, tuple)) and labels:
        first = labels[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            if frame_idx < 0 or frame_idx >= len(labels):
                raise IndexError(f"labels frame_idx {frame_idx} out of range for list length {len(labels)}")
            return _to_string_list(labels[frame_idx])
    return _to_string_list(labels)


def _modern_surface_model_type(stageii_data):
    cfg = stageii_data.get("stageii_debug_details", {}).get("cfg")
    if cfg is None:
        return stageii_data.get("surface_model_type", "smplx")
    if isinstance(cfg, dict):
        return cfg.get("surface_model", {}).get("type", "smplx")
    try:
        return cfg["surface_model"]["type"]
    except Exception:
        return "smplx"


def load_stageii_frame_inputs(stageii_pkl_path, frame_idx=0):
    stageii_data = _load_pickle_compat(stageii_pkl_path)

    if "fullpose" in stageii_data and "trans" in stageii_data:
        source_format = "stageii_pkl"
        surface_model_type = _modern_surface_model_type(stageii_data)
        fullpose = _coerce_frame_matrix(stageii_data["fullpose"], frame_idx, name="fullpose")
        betas = _coerce_frame_matrix(stageii_data["betas"], 0, name="betas")
        transl = _coerce_frame_matrix(stageii_data["trans"], frame_idx, name="trans")
        expression = _coerce_optional_expression(stageii_data.get("expression"), frame_idx)
        markers_latent = _coerce_marker_frame(stageii_data["markers_latent"], frame_idx=0, name="markers_latent")
        latent_labels = _to_string_list(stageii_data["latent_labels"])
        debug = stageii_data.get("stageii_debug_details", {})
        marker_observations = _coerce_marker_frame(debug["markers_obs"], frame_idx, name="markers_obs")
        marker_labels = _labels_for_frame(debug["labels_obs"], frame_idx)
    else:
        source_format = "legacy_stageii_pkl"
        surface_model_type = str(stageii_data["ps"]["fitting_model"])
        fullpose = _coerce_frame_matrix(stageii_data["pose_est_fullposes"], frame_idx, name="pose_est_fullposes")
        betas = _coerce_frame_matrix(stageii_data["shape_est_betas"], 0, name="shape_est_betas")
        transl = _coerce_frame_matrix(stageii_data["pose_est_trans"], frame_idx, name="pose_est_trans")
        expression = _coerce_optional_expression(stageii_data.get("pose_est_exprs"), frame_idx)
        markers_latent = _coerce_marker_frame(stageii_data["shape_est_lmrks"], frame_idx=0, name="shape_est_lmrks")
        latent_labels = _to_string_list(stageii_data["shape_est_lmlabels"])
        marker_observations = _coerce_marker_frame(stageii_data["pose_est_obmrks"], frame_idx, name="pose_est_obmrks")
        marker_labels = _labels_for_frame(stageii_data["pose_est_mrk_labels"], frame_idx)

    fullpose = _coerce_fullpose_to_smplx_layout(fullpose, surface_model_type)

    return StageIIFrameInputs(
        source_format=source_format,
        surface_model_type=surface_model_type,
        fullpose=torch.as_tensor(fullpose, dtype=torch.float32),
        betas=torch.as_tensor(betas, dtype=torch.float32),
        transl=torch.as_tensor(transl, dtype=torch.float32),
        expression=expression,
        markers_latent=torch.as_tensor(markers_latent, dtype=torch.float32),
        latent_labels=latent_labels,
        marker_observations=torch.as_tensor(marker_observations, dtype=torch.float32),
        marker_labels=marker_labels,
        frame_idx=frame_idx,
    )


def load_mocap_frame(mocap_path, frame_idx=0):
    mocap_path = Path(mocap_path)
    suffix = mocap_path.suffix.lower()
    if suffix == ".pkl":
        mocap_data = _load_pickle_compat(mocap_path)
        markers = _coerce_marker_frame(mocap_data["markers"], frame_idx, name="markers")
        labels = _labels_for_frame(mocap_data.get("labels_perframe"), frame_idx) or _to_string_list(
            mocap_data.get("labels")
        )
        frame_rate = float(mocap_data.get("frame_rate", 120.0))
        source_format = "mocap_pkl"
    elif suffix in {".c3d", ".mcp"}:
        from moshpp.tools.c3d import Reader as C3DReader

        with mocap_path.open("rb") as handle:
            reader = C3DReader(handle)
            labels = [str(label).strip() for label in reader.point_labels]
            for current_idx, (_, points, _) in enumerate(reader.read_frames(copy=True)):
                if current_idx == frame_idx:
                    markers = points[:, :3].astype(np.float32, copy=True)
                    invalid = points[:, 3] < 0
                    markers[invalid] = np.nan
                    frame_rate = float(reader.point_rate)
                    source_format = "mocap_c3d"
                    break
            else:
                raise IndexError(f"frame_idx {frame_idx} out of range for {mocap_path}")
    else:
        raise ValueError(f"Unsupported mocap file format: {mocap_path}")

    return MocapFrameData(
        source_format=source_format,
        markers=torch.as_tensor(markers, dtype=torch.float32),
        labels=labels,
        frame_rate=frame_rate,
        frame_idx=frame_idx,
    )


def _validate_smoke_inputs(
    fullpose,
    betas,
    transl,
    marker_attachment,
    marker_observations,
    pose_prior,
    body_model,
    body_model_factory,
):
    if body_model is None and body_model_factory is None:
        raise ValueError("Provide body_model or body_model_factory.")
    if body_model is not None and body_model_factory is not None:
        raise ValueError("Provide only one of body_model or body_model_factory.")
    if fullpose.ndim != 2 or fullpose.shape != (1, 165):
        raise ValueError(f"fullpose must have shape (1, 165), got {tuple(fullpose.shape)}")
    if betas.ndim != 2:
        raise ValueError("betas must be a 2D tensor.")
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError("transl must have shape (B, 3).")
    if marker_attachment is None:
        raise ValueError("marker_attachment is required.")
    if not hasattr(marker_attachment, "closest") or not hasattr(marker_attachment, "coeffs"):
        raise ValueError("marker_attachment must expose closest and coeffs.")
    if marker_observations is None:
        raise ValueError("marker_observations is required.")
    if marker_observations.ndim != 2 or marker_observations.shape[1] != 3:
        raise ValueError("marker_observations must have shape (M, 3).")
    if marker_attachment.coeffs.shape[0] != marker_observations.shape[0]:
        raise ValueError("marker_attachment and marker_observations must have the same marker count.")
    if pose_prior is None or not callable(pose_prior):
        raise ValueError("pose_prior is required and must be callable.")


def _select_prior_input(fullpose, pose_prior):
    if not hasattr(pose_prior, "means"):
        raise ValueError("pose_prior must expose means for dimensionality checks.")
    prior_dim = pose_prior.means.shape[-1]
    if prior_dim == fullpose.shape[1]:
        return fullpose
    if prior_dim == 63:
        return fullpose[:, 3:66]
    if prior_dim == 69:
        return fullpose[:, 3:72]
    raise ValueError(f"Unsupported prior dimension {prior_dim} for fullpose shape {tuple(fullpose.shape)}")


def run_stageii_torch_smoke(
    *,
    fullpose,
    betas,
    transl,
    marker_attachment,
    marker_observations,
    pose_prior,
    body_model=None,
    body_model_factory=None,
    expression=None,
):
    fullpose = torch.as_tensor(fullpose)
    betas = torch.as_tensor(betas)
    transl = torch.as_tensor(transl)
    marker_observations = torch.as_tensor(marker_observations)

    _validate_smoke_inputs(
        fullpose=fullpose,
        betas=betas,
        transl=transl,
        marker_attachment=marker_attachment,
        marker_observations=marker_observations,
        pose_prior=pose_prior,
        body_model=body_model,
        body_model_factory=body_model_factory,
    )

    wrapper = SmplxTorchWrapper(
        body_model=body_model,
        body_model_factory=body_model_factory,
        surface_model_type="smplx",
    )
    body_output = wrapper(fullpose=fullpose, betas=betas, transl=transl, expression=expression)
    predicted_markers = decode_marker_attachment(marker_attachment, body_output.vertices[0])
    data_residual = predicted_markers - marker_observations.to(predicted_markers.dtype)
    prior_residual = pose_prior(_select_prior_input(fullpose, pose_prior))

    return StageIITorchSmokeResult(
        vertices=body_output.vertices,
        joints=body_output.joints,
        predicted_markers=predicted_markers,
        data_residual=data_residual,
        prior_residual=prior_residual,
    )
