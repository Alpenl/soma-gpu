---
round:
  id: "YYYY-MM-DD-<topic>"
  date: "YYYY-MM-DD"
  status: "done"
repo:
  branch: "<branch>"
  commits:
    - "<commit-hash-1>"
    - "<commit-hash-2>"
artifacts:
  docs:
    - "plan.md"
    - "test.md"
    - "summary.md"
  code:
    - "<optional: paths>"
---

# 本轮总结与交接（Summary & Handoff）

## 0. 本轮结论（TL;DR）

用 3 到 6 行写清楚：

- 做了什么（产出）
- 结果如何（验证与影响）
- 还有什么没做（遗留）
- 下一轮最推荐做什么（建议）

---

## 1. 产出清单（Deliverables）

文档：

- `plan.md`：TODO
- `test.md`：TODO
- `summary.md`：本文件

代码/配置/其他：

- TODO（列出关键文件路径与一句话说明）

---

## 2. 验证结论（Verification）

> 这里只写结论与指针；详细命令和输出摘要在 `test.md`。

- 最小验证：PASS / FAIL / PARTIAL
- 关键说明：TODO

---

## 3. 关键决策与取舍（Decisions）

> 写“为什么这样做”，避免下一轮重复争论。

- 决策：TODO
  - 备选方案：TODO
  - 取舍原因：TODO

---

## 4. 风险与遗留（Risks & TODO）

风险：

- TODO（影响 + 触发条件 + 建议应对）

遗留 TODO（下一轮可直接接手）：

1. TODO
2. TODO
3. TODO

---

## 5. 下一轮建议（Next Round Suggestions）

> 这里的建议要“可执行”，尽量能直接变成下一轮的任务拆分。

1. TODO
2. TODO
3. TODO

---

## 6. 子代理回执（Worker Reports，可选但强烈建议）

> 主会话可把每个子代理的汇报粘贴到这里，形成完整审计链。

### Worker: <name>

- 做了什么：TODO
- 改了哪些文件：TODO
- 怎么验证的：TODO
- 风险与疑点：TODO
- 未完成项：TODO

---

## 7. 交接给下一轮：强制阅读清单（Handoff Reading List）

> 下一轮开始前必须读的文件路径，尽量精确到本轮目录内。

- `MAIN.md`
- `docs/codex-potter/governance/resume-and-handoff.md`
- `docs/codex-potter/iterations/<this-round>/round-overview.md`
- `docs/codex-potter/iterations/<this-round>/plan.md`
- `docs/codex-potter/iterations/<this-round>/test.md`
- `docs/codex-potter/iterations/<this-round>/summary.md`
- `docs/codex-potter/iterations/<this-round>/next-round-suggestions.md`
- （按需）TODO：其他关键背景文档/代码入口点
