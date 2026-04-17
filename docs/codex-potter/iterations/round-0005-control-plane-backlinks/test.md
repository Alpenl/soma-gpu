# 本轮测试记录（Test Log）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

## 1. 测试目标

- 验证 `docs/codex-potter/` 下当前生效文档不再缺少主入口 Markdown 回链
- 验证新增 R0005 后，运行索引与快速入口一致
- 验证批量回链补丁没有引入格式问题或本地断链
- 验证五轮目录的八件套结构保持完整

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

2. 主入口回链缺口扫描
   - 命令：
     - `python3` 脚本扫描 `docs/codex-potter/**/*.md` 中是否仍有文档缺少指向 `MAIN.md` / 控制面总说明 / 工作流协议 / 轮次索引的 Markdown 回链
   - 期望：输出 `missing=0`
   - 结果：PASS
   - 关键输出摘要：输出 `missing=0`

3. 本地 Markdown 链接校验
   - 命令：
     - `python3` 脚本校验 `README.md`、`MAIN.md` 与 `docs/codex-potter/**/*.md` 的本地 Markdown 链接
   - 期望：所有本地链接可解析
   - 结果：PASS
   - 关键输出摘要：扫描 `56` 个 Markdown 文件，校验 `257` 个本地链接，`errors=0`

4. 五轮八件套完整性校验
   - 命令：
     - `python3` 脚本校验 `round-0001-init` 到 `round-0005-control-plane-backlinks` 是否都包含 8 份必备文档
   - 期望：全部 `OK`
   - 结果：PASS
   - 关键输出摘要：`rounds=5`，`round-0001` 到 `round-0005` 均为 `OK`

5. 索引同步检查
   - 命令：
     - `rg -n "R0005|round-0005-control-plane-backlinks|控制面回链补强" MAIN.md docs/codex-potter/iterations/README.md`
   - 期望：`MAIN.md` 与 `iterations/README.md` 同时命中 R0005
   - 结果：PASS
   - 关键输出摘要：`MAIN.md` 与 `docs/codex-potter/iterations/README.md` 均命中 R0005 行

## 4. 结论

- 结论：PASS
- 备注：提交后仍需把实际 commit hash 回填到 `summary.md` / `commit.md`
