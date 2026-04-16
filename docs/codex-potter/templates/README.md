# CodexPotter 模板说明

本目录提供每轮必备文档的模板（计划/测试/总结）。目标是把“可续跑”变成默认行为：每轮开始先立计划，每轮结束必有交接包。

模板文件：

- `plan-template.md`：本轮计划（含阅读清单、任务拆分、完成定义、验证计划）
- `test-template.md`：本轮测试记录（含环境、命令、结果、回归点）
- `summary-template.md`：本轮总结与交接（含产出、影响、commit、下一轮建议）

续跑规范：

- `docs/codex-potter/governance/resume-and-handoff.md`

---

## 推荐目录结构（外部文档体系）

模板不强制你必须用某一种目录布局，但建议统一放在：

- `docs/codex-potter/iterations/round-XXXX-<slug>/`

并在该目录下固定最小交接包：

- `round-overview.md`
- `plan.md`
- `test.md`
- `summary.md`
- `next-round-suggestions.md`

这样做的好处：

- 任何人只要进入某个 rollout 目录，就能快速定位计划/验证/交接信息。
- 目录名天然可排序，方便回看历史演进。

---

## 使用方法

1. 新建本轮目录，例如：`docs/codex-potter/iterations/round-0002-benchmark-baseline/`
2. 先创建 `round-overview.md` 和 `next-round-suggestions.md`，再复制模板生成三件套：
   - `plan.md` 从 `plan-template.md` 复制
   - `test.md` 从 `test-template.md` 复制
   - `summary.md` 从 `summary-template.md` 复制
3. 先填 `plan.md` 的“阅读清单”和“范围/非目标”，再开始派发子代理任务。
4. 所有验证命令与关键输出摘要，写入 `test.md`。
5. 结束本轮前填写 `summary.md`，并回链到 `round-overview.md` 与 `next-round-suggestions.md`。

---

## 字段口径与约定

- “范围/非目标”：必须写清楚，避免子代理扩散改动。
- “完成定义（DoD）”：用可验收条款描述，例如“跑通 `pytest -q`（或说明为何不能）”“新增 X 的最小单测”“文档互链可跳转”。
- “验证”：优先写“命令 + 期望结果”。如果受限于私有资产或环境，请把限制写清楚，并给出替代验证（例如 toy test、接口检查、静态检查）。
