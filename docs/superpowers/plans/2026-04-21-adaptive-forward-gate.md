# Adaptive Forward Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current adaptive fast solver with a forward-only predictor/verifier path plus exact fallback.

**Architecture:** The adaptive branch keeps exact frame 0 and exact fallback behavior, but fast frames stop calling the optimizer. Instead they use extrapolated latent pose and translation, run one or two forward evaluations, optionally apply a translation correction under fixed pose, and accept only when residual stays under threshold.

**Tech Stack:** Python, PyTorch, existing `StageIIFrameEvaluator`, pytest.

---

### Task 1: Add forward-only adaptive helpers

**Files:**
- Modify: `/home/alpen/DEV/soma-gpu/moshpp/chmosh_torch.py`
- Test: `/home/alpen/DEV/soma-gpu/tests/test_chmosh_torch.py`

- [ ] Add helper(s) to evaluate a predicted frame state without optimizer steps.
- [ ] Add helper(s) to apply one fixed-pose translation correction from marker residual mean.
- [ ] Keep helper outputs compatible with `_append_stageii_frame_result(...)`.

### Task 2: Switch adaptive fast branch to predictor/verifier

**Files:**
- Modify: `/home/alpen/DEV/soma-gpu/moshpp/chmosh_torch.py`
- Test: `/home/alpen/DEV/soma-gpu/tests/test_chmosh_torch.py`

- [ ] Replace `fit_stageii_frame_torch(...)` fast calls in the adaptive branch with the new forward-only path.
- [ ] Preserve exact anchor and exact fallback behavior.
- [ ] Extend adaptive debug stats to distinguish forward accepts from exact fallback frames.

### Task 3: Expand tests

**Files:**
- Modify: `/home/alpen/DEV/soma-gpu/tests/test_chmosh_torch.py`

- [ ] Add a test that accepts the forward-only path.
- [ ] Add a test that proves translation correction changes the accepted state.
- [ ] Keep fallback coverage for above-threshold frames.

### Task 4: Verify and benchmark

**Files:**
- Modify: `/home/alpen/DEV/soma-gpu/moshpp/chmosh_torch.py`
- Modify: `/home/alpen/DEV/soma-gpu/tests/test_chmosh_torch.py`

- [ ] Run targeted adaptive tests.
- [ ] Run the existing frame/batch/chmosh regression suite.
- [ ] Benchmark `real100` and `a3_80` with exact vs adaptive forward gate.
- [ ] Record whether the new structure materially lowers wall-clock and whether quality remains near the exact baseline.
