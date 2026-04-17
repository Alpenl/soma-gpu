---
round:
  id: "round-0002-potter-entry-normalization"
  date: "2026-04-16"
repo:
  branch: "main"
  head_commit: "4fb8c2af7449d6850cfaeb45839287b1524a2408"
environment:
  os: "Linux 6.8.0-94-generic"
  python: "Python 3.10.12 (python3)"
  cuda: "N/A"
  gpu: "N/A"
---

# 本轮测试记录（Test Log）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

> 本轮主要验证“入口规则正确性”与“文档/内部 progress file 是否同步”，不是验证 GPU 功能。
>
> 2026-04-17 补充：在入口归一化之外，追加检查本轮目录的八件套完整性，避免示例轮次与协议脱节。

## 1. 测试范围（What We Tested）

- CodexPotter 官方文档中的 `resume PROJECT_PATH` 解析规则
- CodexPotter 源码中的 workdir 推导规则
- 当前仓库的根入口文档与标准续跑命令是否已写清
- 当前 `.codexpotter` 内部 progress file 是否已同步到最新基线

不在范围（Not Tested）：

- GPU 代码、benchmark、profiling、误差或视觉效果
- `potter-rollout.jsonl` 的内容变更

## 2. 环境信息（Environment）

- OS：Linux 6.8.0-94-generic
- Python：Python 3.10.12（`python` 不存在，使用 `python3`）
- 关键依赖版本：CodexPotter `0.1.13`、Git、ripgrep、shell 内置工具
- 硬件信息（如相关）：N/A

## 3. 执行记录（Commands & Results）

1. 命令：
   - `sed -n '95,140p' /tmp/CodexPotter-src/docs/wiki/cli.md`
   - 期望：出现 `resume [PROJECT_PATH]` 的 candidate 解析规则
   - 结果：PASS
   - 关键输出摘要：文档明确写出 relative path 的两个候选：`<cwd>/.codexpotter/projects/<PROJECT_PATH>/MAIN.md` 与 `<cwd>/<PROJECT_PATH>/MAIN.md`

2. 命令：
   - `sed -n '659,680p' /tmp/CodexPotter-src/cli/src/workflow/resume.rs`
   - 期望：出现 `derive_project_workdir`，并要求 progress file 位于 `.codexpotter`
   - 结果：PASS
   - 关键输出摘要：源码会一路向上查找 `.codexpotter` 目录；若找不到，会报 `progress file is not inside a \`.codexpotter\` directory`

3. 命令：
   - `sed -n '1,160p' .codexpotter/projects/2026/04/16/1/MAIN.md`
   - 期望：front matter 已同步到最新基线提交，并写明 root MAIN 与 runtime progress file 的区别
   - 结果：PASS
   - 关键输出摘要：`git_commit` 已更新为 `462f95d457e6eafcc963e613eba7b08e15241c93`，并新增 `Runtime Notes` 与标准 `resume` 命令

4. 命令：
   - `git diff --check -- MAIN.md docs/codex-potter`
   - 期望：无输出，说明无明显格式问题
   - 结果：PASS
   - 关键输出摘要：无输出

5. 命令：
   - `git status --short`
   - 期望：只看到本轮相关文档改动；`.codexpotter` 变更默认不进入 git
   - 结果：PASS
   - 关键输出摘要：git 仅显示 `MAIN.md`、`docs/codex-potter/...` 的本轮文档改动；`.codexpotter` 仍保持本地 gitignored 状态

6. 命令：
   - `codex-potter resume 2026/04/16/1 --sandbox read-only --rounds 1`
   - 期望：能成功回放项目并进入 action picker，而不是入口解析报错
   - 结果：PASS
   - 关键输出摘要：TUI 成功进入项目回放，并显示 `Select Action` / `Continue & iterate 1 more round`；随后用 `Esc` 退出，未继续执行新一轮

7. 命令：
   - `wc -l .codexpotter/projects/2026/04/16/1/potter-rollout.jsonl`
   - 期望：仍为 3 行，证明只读 resume 探测没有追加新轮次日志
   - 结果：PASS
   - 关键输出摘要：输出为 `3 .codexpotter/projects/2026/04/16/1/potter-rollout.jsonl`

8. 命令：
   - `find docs/codex-potter/iterations/round-0002-potter-entry-normalization -maxdepth 1 -type f | sort`
   - 期望：目录内八件套齐全，且包含 `code.md` / `commit.md` / `close.md`
   - 结果：PASS
   - 关键输出摘要：目录下 8 个阶段文档齐全，入口归一化样例可直接作为续跑输入

## 4. 失败项与排查（If Any）

失败项清单：

- 两个只读探测子代理均因上游 `502 Bad Gateway` 失败

已做排查：

- 改为直接克隆官方源码到 `/tmp/CodexPotter-src`，本地读取 `docs/wiki/cli.md` 与 `cli/src/workflow/resume.rs`

下一步建议：

- 若后续还需要对 Potter 实现做更深检查，优先本地源码验证，不依赖不稳定的外部子代理通道
- 下一轮开始时直接使用 `codex-potter resume 2026/04/16/1 --yolo --rounds 10`（按需改最后的轮数），不要再尝试仓库根 `MAIN.md`

## 5. 回归风险点（Regression Watchlist）

- 未来如果升级 CodexPotter 版本，`resume` 解析规则可能变化
- 仓库根 `MAIN.md` 与内部 progress file 再次分叉
- 下一轮若直接从 `resume` 启动但未先读轮次文档，仍可能造成工作流偏航
