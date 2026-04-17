from dataclasses import dataclass
import os

import numpy as np
import torch


@dataclass
class MarkerAttachment:
    closest: torch.Tensor
    coeffs: torch.Tensor

    def to(self, *, device=None, dtype=None):
        closest = self.closest
        coeffs = self.coeffs
        if device is not None and closest.device != torch.device(device):
            closest = closest.to(device=device)
        if device is not None or dtype is not None:
            coeffs = coeffs.to(device=device, dtype=dtype or coeffs.dtype)
        return MarkerAttachment(closest=closest, coeffs=coeffs)

    def index_select(self, index):
        if not torch.is_tensor(index):
            index = torch.as_tensor(index, dtype=torch.long, device=self.closest.device)
        else:
            index = index.to(device=self.closest.device, dtype=torch.long)
        return MarkerAttachment(
            closest=self.closest.index_select(0, index),
            coeffs=self.coeffs.index_select(0, index),
        )


def _normalize_np(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _normalize_torch(vectors):
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True).clamp_min(1e-12)
    return vectors / norms


def _normalize_torch_lastdim(vectors):
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-12)
    return vectors / norms


def _load_no_eyeball_vids(support_base_dir=None):
    if support_base_dir is None:
        support_base_dir = os.path.join(os.path.dirname(__file__), "support_data")
    eyeballs = np.load(os.path.join(support_base_dir, "smplx_eyeballs.npz"))["eyeballs"]
    return np.array(
        sorted(set(np.arange(10475).tolist()).difference(set(eyeballs.tolist()))),
        dtype=np.int64,
    )


def _choose_basis_indices(body_vertices, candidate_ids):
    base_idx = int(candidate_ids[0])
    e1_idx = int(candidate_ids[1])
    e1 = body_vertices[e1_idx] - body_vertices[base_idx]

    for candidate_idx in candidate_ids[2:]:
        candidate_idx = int(candidate_idx)
        e2 = body_vertices[candidate_idx] - body_vertices[base_idx]
        if np.linalg.norm(np.cross(e1, e2)) >= 1e-12:
            return base_idx, e1_idx, candidate_idx

    raise ValueError("Could not find a non-colinear basis from nearest neighbors.")


def _nearest_neighbors(points, queries, n_neighbors):
    distances = np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=2)
    partition = np.argpartition(distances, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
    ordered = np.take_along_axis(
        partition,
        np.argsort(np.take_along_axis(distances, partition, axis=1), axis=1),
        axis=1,
    )
    return ordered


def build_marker_attachment(
    can_body_verts,
    markers_latent,
    surface_model_type="smplx",
    support_base_dir=None,
    n_neighbors=8,
):
    can_body_np = torch.as_tensor(can_body_verts, dtype=torch.float32).detach().cpu().numpy()
    markers_np = torch.as_tensor(markers_latent, dtype=torch.float32).detach().cpu().numpy()

    search_ids = np.arange(can_body_np.shape[0], dtype=np.int64)
    if surface_model_type == "smplx" and can_body_np.shape[0] == 10475:
        search_ids = _load_no_eyeball_vids(support_base_dir=support_base_dir)

    search_body = can_body_np[search_ids]
    nn_count = min(n_neighbors, search_body.shape[0])
    if nn_count < 3:
        raise ValueError("Marker attachment requires at least three candidate neighbors.")

    closest_subset = _nearest_neighbors(search_body, markers_np, nn_count)
    closest_orig = search_ids[closest_subset]

    chosen_triplets = []
    coeffs = []
    for marker, candidate_ids in zip(markers_np, closest_orig):
        base_idx, e1_idx, e2_idx = _choose_basis_indices(can_body_np, candidate_ids)
        base = can_body_np[base_idx]
        e1 = can_body_np[e1_idx] - base
        e2 = can_body_np[e2_idx] - base

        f1 = _normalize_np(e1)
        f2 = _normalize_np(np.cross(e1, e2))
        f3 = np.cross(f1, f2)

        diff = marker - base
        coeffs.append(
            [
                float(np.dot(diff, f1)),
                float(np.dot(diff, f2)),
                float(np.dot(diff, f3)),
            ]
        )
        chosen_triplets.append([base_idx, e1_idx, e2_idx])

    return MarkerAttachment(
        closest=torch.tensor(chosen_triplets, dtype=torch.long),
        coeffs=torch.tensor(coeffs, dtype=torch.float32),
    )


def decode_marker_attachment_batched(attachment, body_verts):
    body_verts = torch.as_tensor(body_verts)
    if not torch.is_floating_point(body_verts):
        body_verts = body_verts.to(torch.float32)

    squeeze_batch = False
    if body_verts.ndim == 2:
        body_verts = body_verts.unsqueeze(0)
        squeeze_batch = True
    if body_verts.ndim != 3 or body_verts.shape[-1] != 3:
        raise ValueError(f"body_verts must have shape (V, 3) or (B, V, 3), got {tuple(body_verts.shape)}")

    closest = attachment.closest
    coeffs = attachment.coeffs
    if closest.device != body_verts.device:
        closest = closest.to(device=body_verts.device)
    if coeffs.device != body_verts.device or coeffs.dtype != body_verts.dtype:
        coeffs = coeffs.to(device=body_verts.device, dtype=body_verts.dtype)

    base = body_verts[:, closest[:, 0], :]
    e1 = body_verts[:, closest[:, 1], :] - base
    e2 = body_verts[:, closest[:, 2], :] - base

    f1 = _normalize_torch_lastdim(e1)
    f2 = _normalize_torch_lastdim(torch.cross(e1, e2, dim=-1))
    f3 = torch.cross(f1, f2, dim=-1)

    markers = (
        base
        + coeffs[None, :, 0:1] * f1
        + coeffs[None, :, 1:2] * f2
        + coeffs[None, :, 2:3] * f3
    )
    if squeeze_batch:
        return markers[0]
    return markers


def decode_marker_attachment(attachment, body_verts):
    return decode_marker_attachment_batched(attachment, body_verts)
