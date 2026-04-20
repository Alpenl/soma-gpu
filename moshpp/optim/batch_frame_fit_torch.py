"""Batch-parallel per-frame GPU stageii solver v6.

Strategy: run independent per-frame LBFGS solvers in parallel using
multiple CUDA streams. Each frame gets its own LBFGS with its own
body model forward, but they execute concurrently on the GPU.
"""
import torch
import concurrent.futures

from moshpp.optim.frame_fit_torch import (
    TorchFrameFitWeights,
    TorchFrameFitOptions,
    fit_stageii_frame_torch,
    build_stageii_evaluator,
)
from moshpp.transformed_lm_torch import MarkerAttachment


def _solve_single_frame(
    frame_idx, markers_obs, attachment_subset, weights, options,
    body_model_factory, wrapper_factory, betas, pose_prior, layout, hand_pca,
    latent_pose_init, transl_init, expression_init,
    optimize_fingers, optimize_face, optimize_toes,
    velocity_reference, rigid_init, warmup_scales, device,
):
    """Solve a single frame on a dedicated CUDA stream."""
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        body_model = body_model_factory()
        wrapper = wrapper_factory(body_model)
        evaluator = build_stageii_evaluator(
            wrapper=wrapper, layout=layout, hand_pca=hand_pca,
            pose_prior=pose_prior, optimize_fingers=optimize_fingers,
            optimize_face=optimize_face,
        )
        result = fit_stageii_frame_torch(
            body_model=body_model,
            wrapper=wrapper,
            betas=betas,
            marker_attachment=attachment_subset,
            marker_observations=markers_obs,
            pose_prior=pose_prior,
            layout=layout,
            latent_pose_init=latent_pose_init,
            transl_init=transl_init,
            expression_init=expression_init,
            hand_pca=hand_pca,
            optimize_fingers=optimize_fingers,
            optimize_face=optimize_face,
            optimize_toes=optimize_toes,
            velocity_reference=velocity_reference,
            weights=weights,
            options=options,
            rigid_init=rigid_init,
            warmup_pose_scales=warmup_scales,
            evaluator=evaluator,
        )
    stream.synchronize()
    return frame_idx, result


def fit_batch_frames_v6(
    *,
    body_model_factory,
    wrapper_factory,
    betas,
    marker_attachment,
    all_markers_obs,     # list of (Mi, 3) tensors
    all_visible_ids,     # list of lists of visible marker ids
    pose_prior, layout, hand_pca,
    latent_pose_inits,   # list of (1, D) tensors
    transl_inits,        # list of (1, 3) tensors
    all_weights,         # list of TorchFrameFitWeights
    options,
    optimize_fingers, optimize_face, optimize_toes,
    velocity_references=None,
    expression_inits=None,
    max_workers=8,
    device="cuda",
):
    """Run N independent per-frame LBFGS solvers in parallel."""
    N = len(all_markers_obs)

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(N):
            attachment_subset = MarkerAttachment(
                closest=marker_attachment.closest[all_visible_ids[i]],
                coeffs=marker_attachment.coeffs[all_visible_ids[i]],
            )
            vel_ref = velocity_references[i] if velocity_references is not None else None
            expr_init = expression_inits[i] if expression_inits is not None else None
            rigid_init = (i == 0)
            warmup_scales = (10.0, 5.0, 1.0) if i == 0 else (1.0,)

            fut = executor.submit(
                _solve_single_frame,
                i, all_markers_obs[i], attachment_subset, all_weights[i], options,
                body_model_factory, wrapper_factory, betas, pose_prior, layout, hand_pca,
                latent_pose_inits[i], transl_inits[i], expr_init,
                optimize_fingers, optimize_face, optimize_toes,
                vel_ref, rigid_init, warmup_scales, device,
            )
            futures.append(fut)

    results = [None] * N
    for fut in concurrent.futures.as_completed(futures):
        idx, result = fut.result()
        results[idx] = result

    return results
