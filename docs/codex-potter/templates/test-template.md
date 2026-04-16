---
round:
  id: "YYYY-MM-DD-<topic>"
  date: "YYYY-MM-DD"
repo:
  branch: "<branch>"
  head_commit: "<commit-hash>"
environment:
  os: "<linux/mac/windows>"
  python: "<python --version>"
  cuda: "<optional>"
  gpu: "<optional>"
---

# 本轮测试记录（Test Log）

> 原则：记录“别人拿到这份文档，能否复现你说的结论”。成功与失败都必须写。

## 1. 测试范围（What We Tested）

- TODO（例如：单元测试、某条脚本 smoke、文档链接检查）

不在范围（Not Tested）：

- TODO（例如：依赖私有数据/模型资产，当前无法覆盖）

---

## 2. 环境信息（Environment）

> 尽量用命令输出填入（或手工写清楚）。

- OS：TODO
- Python：TODO
- 关键依赖版本：TODO（torch/smplx/pytest 等）
- 硬件信息（如相关）：TODO（GPU 型号、显存、驱动/CUDA）

---

## 3. 执行记录（Commands & Results）

按时间顺序记录每条关键命令：

1. 命令：
   - `TODO`
   - 期望：TODO
   - 结果：PASS / FAIL / SKIP
   - 关键输出摘要：TODO（不要粘贴海量日志，提炼关键信息即可）

2. 命令：
   - `TODO`
   - 期望：TODO
   - 结果：TODO
   - 关键输出摘要：TODO

---

## 4. 失败项与排查（If Any）

失败项清单：

- TODO（失败用例/命令）

已做排查：

- TODO（定位到的原因）

下一步建议：

- TODO（可执行的修复/绕行方案）

---

## 5. 回归风险点（Regression Watchlist）

> 写“可能被本轮改动影响到”的点，方便下一轮重点盯防。

- TODO（例如：接口签名、路径/目录规范、默认配置、性能）

