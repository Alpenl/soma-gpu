# 本轮测试记录（Test Log）

上层入口：

- 仓库级入口：[MAIN.md](../../../../MAIN.md)
- 控制面总说明：[docs/codex-potter/README.md](../../README.md)
- 工作流协议：[docs/codex-potter/governance/workflow-protocol.md](../../governance/workflow-protocol.md)
- 轮次索引：[docs/codex-potter/iterations/README.md](../README.md)

日期：2026-04-17

## 1. 测试目标

- 验证活跃入口互链缺口已被补齐
- 验证新增 R0004 后，运行索引与快速入口一致
- 验证本轮新增文档没有引入格式问题或本地断链

## 2. 执行环境

- 仓库：`soma-gpu`
- 工作区：`.worktrees/control-plane-entry-links`
- 分支：`control-plane-entry-links`
- 说明：本轮仅涉及控制面 Markdown 文档与本地 `.codexpotter` 记录

## 3. 执行的验证

1. 入口互链扫描
   - 命令：
     - `rg -n 'MAIN\.md|docs/codex-potter/README\.md|workflow-protocol\.md|iterations/README\.md' MAIN.md docs/codex-potter/README.md docs/codex-potter/governance/workflow-protocol.md docs/codex-potter/iterations/README.md`
   - 期望：4 个主入口都能直接命中其余入口，不再只依赖 `MAIN.md` 单向分发
   - 结果：PASS
   - 关键输出摘要：`workflow-protocol.md` 现在已命中 `iterations/README.md`，`iterations/README.md` 已命中 `MAIN.md`、控制面总说明与工作流协议

2. 文本格式检查
   - 命令：
     - `git diff --check -- MAIN.md docs/codex-potter/governance/workflow-protocol.md docs/codex-potter/iterations/README.md docs/codex-potter/iterations/round-0004-entry-link-hardening`
   - 期望：无 trailing whitespace、无坏 patch 格式
   - 结果：PASS

3. 活跃入口本地 Markdown 链接校验
   - 命令：`python3` 脚本扫描 `MAIN.md`、`docs/codex-potter/README.md`、`docs/codex-potter/governance/workflow-protocol.md`、`docs/codex-potter/iterations/README.md`
   - 期望：全部相对链接可解析
   - 结果：PASS
   - 关键输出摘要：`checked_links=28`

4. 八件套完整性校验
   - 命令：`python3` 脚本检查 `round-0001` / `round-0002` / `round-0003` / `round-0004`
   - 期望：四轮目录都包含 `round-overview.md`、`plan.md`、`code.md`、`test.md`、`next-round-suggestions.md`、`summary.md`、`commit.md`、`close.md`
   - 结果：PASS
   - 关键输出摘要：四轮目录均返回 `OK`

5. 活跃入口旧口径回归扫描
   - 命令：
     - `rg -n '/home/alpen/DEV/soma-gpu|\brollout\b|六件套|截至 2026-04-16' README.md MAIN.md docs/codex-potter/README.md docs/codex-potter/governance docs/codex-potter/templates docs/codex-potter/iterations/README.md`
   - 期望：本轮未重新引入这些旧口径
   - 结果：PASS
   - 关键输出摘要：命令返回空结果

6. 主入口互链段数量表述校验
   - 命令：
     - `rg -n '上述入口必须保持互链|三者必须保持互链' MAIN.md`
   - 期望：仅命中“上述入口必须保持互链”，不再出现与入口数量绑定的旧表述
   - 结果：PASS
   - 关键输出摘要：只命中 `上述入口必须保持互链`

## 4. 关键输出摘要

- 活跃入口本地 Markdown 链接校验：`checked_links=28`
- 四轮八件套完整性：`round-0001-init` / `round-0002-potter-entry-normalization` / `round-0003-control-plane-hardening` / `round-0004-entry-link-hardening` 全部 `OK`
- 活跃入口旧口径扫描：空结果
- `MAIN.md` 主入口互链段：只保留 `上述入口必须保持互链`

## 5. 已知限制

- 本轮不包含 benchmark、profiling 或 GPU 功能验证
- `.codexpotter` 目录中的运行时记录与知识库按约定不进 git
