# Real MCP Baseline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether `real-mcp-baseline` chunked sequence solving is acceptable enough to stop further per-frame speed work.

**Architecture:** Treat this as an evaluation pipeline, not a feature build. Run the production-adjacent chunked preset on the same real sequence slice used by prior docs, compare its stageii/mesh outputs against the existing per-frame reference path, and in parallel capture whether `torch.compile` is realistically worth fixing as an orthogonal optimization.

**Tech Stack:** Python CLI runners, PyTorch stageii pipeline, benchmark/mesh compare utilities, Codex subagents.

---

### Task 1: Lock The Evaluation Procedure

**Files:**
- Create: `docs/superpowers/plans/2026-04-21-real-mcp-baseline-eval.md`
- Read: `docs/GPU逐帧求解器提速分析.md`
- Read: `docs/GPU逐帧求解器重写实验归档.md`
- Read: `run_stageii_torch_official.py`
- Read: `run_stageii_torch_pair.py`

- [ ] **Step 1: Record the acceptance question**

Acceptance question: is `real-mcp-baseline` good enough in production terms that per-frame acceleration work can stop?

- [ ] **Step 2: Record the primary evidence to collect**

Collect:
- wall-clock for the chunked baseline run
- marker residual statistics for the chunked baseline run
- stageii/mesh comparison outputs against the reference path
- `torch.compile` failure evidence and realistic upside assessment

- [ ] **Step 3: Record the decision rule**

Decision rule:
- if chunked baseline quality is acceptable on the reference slice, stop per-frame speed work
- if chunked baseline quality is not acceptable, resume structural work on adaptive pose correction / better predictor
- keep `torch.compile` as an orthogonal investigation, not the primary dependency for the decision

### Task 2: Run The Main Chunked Baseline Benchmark

**Files:**
- Read: `run_stageii_torch_official.py`
- Read: `README.md`
- Write: benchmark outputs under the chosen work directory

- [ ] **Step 1: Resolve the exact input/support/work paths**

Use the same real sequence family discussed in the docs so the result is comparable enough for decision-making.

- [ ] **Step 2: Run the official preset**

Run the official baseline preset on the chosen slice and capture output paths plus wall-clock.

- [ ] **Step 3: Read the generated benchmark report**

Extract residual and timing metrics from the generated artifacts instead of relying on terminal impressions.

### Task 3: Run The Comparison Path

**Files:**
- Read: `run_stageii_torch_pair.py`
- Read: generated `stageii.pkl` and comparison JSON outputs

- [ ] **Step 1: Run the pairwise comparison against the appropriate reference path**

Generate stageii/mesh comparison outputs that help answer whether the chunked result is acceptable.

- [ ] **Step 2: Inspect the resulting deltas**

Focus on residual, seam, jitter, and any obvious production-facing regressions.

### Task 4: Parallel Orthogonal Investigation

**Files:**
- Read: `moshpp/optim/frame_fit_torch.py`
- Read: `moshpp/optim/sequence_evaluator_torch.py`
- Read: any logs produced during compile failure reproduction

- [ ] **Step 1: Reproduce the `torch.compile` failure with full stderr**

Capture the real failure text, not the abbreviated message from the docs.

- [ ] **Step 2: Validate the comparison script and acceptance outputs**

Confirm which reports and metrics the main thread should trust for the stop/continue decision.

### Task 5: Verify And Conclude

**Files:**
- Read: generated reports from Tasks 2-4

- [ ] **Step 1: Re-run the necessary verification commands**

Before claiming any result, run the exact commands that prove the benchmark and comparison outputs exist and contain the metrics cited.

- [ ] **Step 2: Write the conclusion**

State one of:
- stop per-frame acceleration work and use chunked baseline
- continue work because chunked baseline is not acceptable
- continue only the orthogonal `torch.compile` investigation if it shows credible upside
