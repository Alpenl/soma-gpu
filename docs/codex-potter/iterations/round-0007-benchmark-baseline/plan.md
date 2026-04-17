---
round:
  id: "round-0007-benchmark-baseline"
  date: "2026-04-17"
  status: "done"
repo:
  branch: "main"
  base_commit: "52f07b31f0e77d4170bd0026514b68c64d1adf01"
  head_commit: "e67cc13"
roles:
  orchestrator: "main-session"
  workers:
    - "Franklin (explorer, upstream 502)"
    - "Jason (explorer, upstream 502)"
    - "Zeno (explorer, upstream 502)"
    - "Volta (explorer, completed without textual payload)"
    - "Russell (explorer, upstream 502)"
scope_tags:
  - "benchmark"
  - "stageii"
  - "scorecard"
  - "worktree-audit"
---

# 本轮计划（Plan）

上层入口：

- 仓库级入口：[MAIN.md](../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

## 0. 本轮一句话目标

把 round-0007 收敛为一个公开可复跑的 stageii baseline harness，加一份诚实记录外部依赖阻塞与 worktree 采纳边界的第一版 scorecard。

---

## 1. 强制阅读清单（开始前必须完成）

- [x] `MAIN.md`：当前正式目标已经切到 `round-0007-benchmark-baseline`，主会话保持控制面属性。
- [x] 上一轮 `summary.md`：R0006 已要求停止继续主动扩张 docs-only，回到 benchmark/scorecard 主线。
- [x] 上一轮 `test.md`：上一轮只有 control-plane 验证，没有任何 GPU / benchmark 证据。
- [x] 上一轮 `plan.md`：前序 rounds 一直把 benchmark、profiling、candidate 评估列为下一步，但未真正落地。
- [x] `docs/codex-potter/governance/resume-and-handoff.md`：本轮必须补齐八件套，并让下一轮能直接续跑。
- [x] `docs/如何将原生MoSh++改造成GPU版.md`：当前主要瓶颈仍是 `moshpp/chmosh.py` 的 `stageii`，且 released 路线没有现成 GPU backend。
- [x] `.worktrees/gpu-stageii-foundation` 的提交与差异统计：已提交部分是 torch foundation，工作树上还叠了未提交 sequence/render 扩展。

阅读结论（要点）：

- benchmark 首先要解决的是“可复跑”，不是“先得出漂亮提速数字”。
- 当前仓库里真正公开可用的样例资产是 `support_data/tests/mosh_stageii.pkl`；README 中指向的历史 conda 环境在本机不存在。
- 直接调用 `MoSh.load_as_amass_npz(...)` 在当前环境会因为 `body_visualizer.mesh` 缺失而失败，所以 public harness 不能依赖那条导入链。
- `.worktrees/gpu-stageii-foundation` 不能作为一个整体 candidate 进入 merge 讨论；至少要拆成 committed foundation 与 dirty 扩展两层。

---

## 2. 范围（In Scope）与非目标（Out of Scope）

In Scope（本轮要做）：

- 为 `support_data/tests/mosh_stageii.pkl` 新增一个不依赖 Chumpy/Blender/psbody 的 public ingest benchmark harness。
- 输出 `results/public-stageii-benchmark.json` 和 `results/scorecard.md`。
- 用 worktree 审计 + foundation 关键测试给出明确采纳边界。
- 更新 R0007 索引和八件套。

Out of Scope（本轮明确不做）：

- 不把 public benchmark 伪装成 full mesh/mp4 benchmark。
- 不直接接入 torch backend 到 `moshpp/mosh_head.py` 或 released 路线。
- 不在 dirty worktree 上继续追加实现。

约束（必须遵守）：

- benchmark 只使用 repo 内公开样例，不引用私有 tennis 数据路径。
- 不引入新的重依赖；实现尽量只依赖标准库和 `numpy`。
- 外部依赖缺口必须显式写入 scorecard，不做静默跳过。

---

## 3. 产物与完成定义（Deliverables & DoD）

本轮必须产物：

- `round-overview.md`
- `plan.md`（本文件）
- `code.md`
- `test.md`
- `next-round-suggestions.md`
- `summary.md`
- `commit.md`
- `close.md`
- `results/public-stageii-benchmark.json`
- `results/scorecard.md`
- `benchmark_stageii_public.py`
- `utils/stageii_benchmark.py`
- `tests/test_stageii_benchmark.py`

完成定义（DoD），需可验收：

- [x] public stageii sample 能通过一条 repo 内命令生成 JSON 报告。
- [x] benchmark 报告至少含速度、误差、产物/阻塞、工程化字段。
- [x] `code.md` 记录了实际落点与计划偏离。
- [x] `test.md` 记录了 benchmark 测试、结果 JSON 生成和 worktree foundation 测试命令。
- [x] `commit.md` 记录提交策略，并说明 actual hash 由 round close 时补齐。
- [x] `summary.md` 记录了 worktree 的部分采纳结论。
- [x] `close.md` 指明下一轮入口与退出判断。

---

## 4. 任务拆分与派发（面向子代理）

任务列表（计划期望的切分）：

1. 任务：public workload 与 benchmark harness
   - 目标：固定 `support_data/tests/mosh_stageii.pkl` 的兼容加载、重复测量与 JSON 输出。
   - 允许修改：`benchmark_stageii_public.py`、`utils/stageii_benchmark.py`、`tests/test_stageii_benchmark.py`
   - 禁止修改：released GPU 路径、现有 SOMA/MoSh++ 主执行逻辑
   - 验证：`python3 -m pytest tests/test_stageii_benchmark.py -q`

2. 任务：scorecard 与依赖阻塞记录
   - 目标：把 public benchmark 的结果转成 round-0007 可续跑文档。
   - 允许修改：`docs/codex-potter/iterations/round-0007-benchmark-baseline/**`
   - 验证：`python3 benchmark_stageii_public.py ... --output .../public-stageii-benchmark.json`

3. 任务：worktree 候选结论
   - 目标：确认 `.worktrees/gpu-stageii-foundation` 是 committed foundation 部分采纳，而不是整体合入候选。
   - 允许修改：round-0007 文档
   - 验证：`python3 -m pytest tests/test_smplx_torch_wrapper.py tests/test_transformed_lm_torch.py tests/test_gmm_prior_torch.py tests/test_stageii_backend.py tests/test_stageii_torch_smoke.py -q`（在 worktree 内）

说明：

- 计划阶段已尝试使用新的 explorer 子代理，但多次遭遇上游 `502`；因此本轮由主会话做最小收敛，并在文档里保留该限制。

---

## 5. 风险清单（Risks）与应对

- 风险：public workload 只覆盖 stageii-ingest，不覆盖 mesh/mp4
  - 影响：不能给出 full E2E 提速结论
  - 触发信号：缺少 `psbody`、licensed model assets、`blender`
  - 应对：在 scorecard 中显式列为 blocked stages，把下一轮目标改成扩展 benchmark 面而不是冒进优化

- 风险：`MoSh.load_as_amass_npz(...)` 在当前环境导入失败
  - 影响：不能直接复用现有 loader 做 public benchmark
  - 触发信号：`body_visualizer.mesh` 缺失
  - 应对：本轮用独立兼容加载器绕过该依赖，后续若环境补齐再评估是否回切

- 风险：worktree 当前是 dirty tree
  - 影响：无法把整个 worktree 当作稳定 candidate 评估
  - 触发信号：`git status --short` 含 sequence/render 未提交变更
  - 应对：只把 committed foundation 视为部分采纳候选，dirty 扩展暂不合并

---

## 6. 验证计划（Test Plan）

最小验证（必须）：

- `python3 -m pytest tests/test_stageii_benchmark.py -q`
- `python3 benchmark_stageii_public.py --input support_data/tests/mosh_stageii.pkl --output docs/codex-potter/iterations/round-0007-benchmark-baseline/results/public-stageii-benchmark.json --warmup-runs 1 --measured-runs 5`
- 在 `.worktrees/gpu-stageii-foundation` 内运行 foundation 关键测试子集

扩展验证（有时间就做）：

- 记录当前环境的 `torch`/`numpy`/`pytest` 版本与 `blender`/`psbody` 缺口
- 若后续获得 licensed model + Blender，再把本轮 harness 扩到 mesh/mp4

---

## 7. 提交计划（Git Plan）

本轮预计提交策略：

- 分支：`main`
- 提交拆分：优先一个提交收敛 public benchmark harness、round-0007 八件套和索引更新
- 提交信息约定：`docs: add round-0007 public benchmark baseline`
