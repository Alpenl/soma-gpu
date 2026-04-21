# Adaptive Forward Gate Design

## Goal

Reduce per-frame GPU stageii cost by replacing the current adaptive fast path with a true predictor/verifier path. Fast frames should avoid any optimizer step and only pay for forward evaluation plus lightweight translation correction; exact solving remains the fallback path for quality protection.

## Current Problem

The current `adaptive_exact` path still calls `fit_stageii_frame_torch(...)` for the fast branch. Even when many frames are accepted, the fast branch remains expensive because it still runs a partial optimizer. Benchmarks show this caps throughput far above the one-hour target.

## Proposed Change

For `runtime.frame_solver=adaptive_exact`:

1. Keep frame 0 as full exact solve.
2. For later frames, build a predicted state from the last solved state:
   - latent pose velocity extrapolation
   - translation velocity extrapolation
3. Run a pure forward verification pass with `evaluate_stageii_frame(...)` instead of `fit_stageii_frame_torch(...)`.
4. Apply one cheap translation correction under fixed pose:
   - compute marker residual mean
   - shift translation by that mean residual
   - re-evaluate once
5. Accept the corrected predicted state when residual stays under the configured threshold.
6. Otherwise run the existing exact fallback solver unchanged.

## Why This Shape

This keeps the quality contract simple: all risky frames still fall back to the exact solver already in production. The only behavior change is that acceptable easy frames no longer pay optimizer overhead.

## Validation

- Existing adaptive routing tests must still pass.
- Add focused tests for:
  - forward-only fast accept path
  - translation correction being applied before acceptance
  - exact fallback when corrected residual still exceeds threshold
- Benchmark on:
  - `data/a3.pkl` derived raw mocap
  - `data/4090-haonan-73.c3d` first 100 frames

## Success Criteria

- No regression in test coverage for single-frame, batched, and adaptive paths.
- Adaptive diagnostics expose forward accept/reject counts clearly.
- Real benchmarks show a meaningful drop from the current adaptive wall-clock while keeping mean residual at or near the current exact baseline.
