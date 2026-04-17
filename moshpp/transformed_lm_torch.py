from dataclasses import dataclass
import os

import numpy as np
import torch


@dataclass
class MarkerAttachment:
    closest: torch.Tensor
    coeffs: torch.Tensor


def _normalize_np(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _normalize_torch(vectors):
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True).clamp_min(1e-12)
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


def decode_marker_attachment(attachment, body_verts):
    body_verts = torch.as_tensor(body_verts)
    if not torch.is_floating_point(body_verts):
        body_verts = body_verts.to(torch.float32)
    closest = attachment.closest.to(dtype=torch.long, device=body_verts.device)
    coeffs = attachment.coeffs.to(dtype=body_verts.dtype, device=body_verts.device)

    base = body_verts[closest[:, 0]]
    e1 = body_verts[closest[:, 1]] - base
    e2 = body_verts[closest[:, 2]] - base

    f1 = _normalize_torch(e1)
    f2 = _normalize_torch(torch.cross(e1, e2, dim=1))
    f3 = torch.cross(f1, f2, dim=1)

    return (
        base
        + coeffs[:, 0:1] * f1
        + coeffs[:, 1:2] * f2
        + coeffs[:, 2:3] * f3
    )
