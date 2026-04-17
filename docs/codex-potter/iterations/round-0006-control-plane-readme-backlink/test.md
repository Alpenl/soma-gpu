# 本轮测试记录（Test Log）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

## 1. 测试目标

- 验证 `docs/codex-potter/README.md` 已补齐统一“上层入口”区块
- 验证 `docs/codex-potter/**/*.md` 现在不再存在缺少入口块的活跃控制面文档
- 验证新增 R0006 后，运行索引与快速入口保持一致
- 验证本轮补丁没有引入格式问题或本地断链
- 验证六轮目录的八件套结构保持完整

## 2. 执行环境

- 仓库：`soma-gpu`
- 工作区：主工作区
- 分支：`main`
- 说明：本轮仅涉及控制面 Markdown 文档与本地 `.codexpotter` 记录

## 3. 执行的验证

1. 文本格式检查
   - 命令：
     - `git diff --check -- MAIN.md docs/codex-potter`
   - 期望：无 trailing whitespace、无坏 patch 格式
   - 结果：PASS

2. 入口块缺口扫描
   - 命令：
     - `python3` 脚本统计 `docs/codex-potter/**/*.md` 中未包含 `上层入口：` 的文件数
   - 期望：输出 `missing=0`
   - 结果：PASS
   - 关键输出摘要：输出 `files=62 missing=0`

3. 本地 Markdown 链接校验
   - 命令：
     - `python3` 脚本校验 `README.md`、`MAIN.md` 与 `docs/codex-potter/**/*.md` 的本地 Markdown 链接
   - 期望：所有本地链接可解析
   - 结果：PASS
   - 关键输出摘要：扫描 `64` 个 Markdown 文件，校验 `298` 个本地链接，`errors=0`

4. 六轮八件套完整性校验
   - 命令：
     - `python3` 脚本校验 `round-0001-init` 到 `round-0006-control-plane-readme-backlink` 是否都包含 8 份必备文档
   - 期望：全部 `OK`
   - 结果：PASS
   - 关键输出摘要：`rounds=6`，`round-0001` 到 `round-0006` 均为 `OK`

5. 索引同步检查
   - 命令：
     - `rg -n "R0006|round-0006-control-plane-readme-backlink|控制面总说明入口块补齐" MAIN.md docs/codex-potter/iterations/README.md`
   - 期望：`MAIN.md` 与 `iterations/README.md` 同时命中 R0006
   - 结果：PASS
   - 关键输出摘要：`MAIN.md` 与 `docs/codex-potter/iterations/README.md` 均命中 R0006 行

## 4. 结论

- 结论：PASS
- 备注：主体 commit hash 已回填到 `summary.md` / `commit.md`；metadata 回填提交本身不再自指
