# 轻量演示 checkpoint — 2026-09-08

## 交付

本地入口 http://127.0.0.1:8001/graph/demo；默认关闭，当前本地实例显式开启。服务端只增加固定静态页面和两个 allowlist 资源，展示操作使用既有 graph API，不加入会话/追问/Jira 写回。

三个独立操作：Epic overview、Epic detail、Dependency evidence。另有业务描述定位起点的预设和可选 Generate grounded answer。页面直接显示实际返回的计数、快照元数据、方向、状态、关系来源与文本；没有硬编码数量或查询结果。

## 浏览器实际验收

Chromium 143.0.7499.4 / Playwright 1.57.0，本机 1440×1000 与 390×844 视口。

- 真实 AIPLAT：35 个事项、8 Epic；AIPLAT-13 子事项 4；AIPLAT-46 → BLOCKS → AIPLAT-16，双方 Done，按记录关系展示；未完成配对筛选后不匹配。
- 真实业务描述预设不带 key，页面返回 Hybrid search 和起点 AIPLAT-46，再显示相同 BLOCKS 路径。单次浏览器端到端约 4164.94 ms，不作为 P95 或负载结论；未调用回答模型，embedding usage 未由 HTTP 捕获。见 jira-graph-demo-live.json。
- 独立视图、固定分页与生成快照、控件修改清空、迟到响应隔离、图故障正文来源、HTML 转义/恶意 URL、截断提示、错误恢复、键盘操作和窄屏均纳入浏览器验收。
- 生成失败保留证据、失败后重试成功的 UI 分支使用注入响应。当前实际模型 model_not_found 限制未解决，未宣称真实成功回答。

截图：data/graph-poc/demo-artifacts/{overview,detail,dependencies,mobile,hybrid}.png（含项目内容，未提交）。

## 红绿与审查

- 页面路由最初 2 failed / 3 passed（缺失页面和资源），实现后 5 passed。默认关闭、资源 allowlist、CSP/no-store/nosniff 验证通过。
- 首轮浏览器 7 passed / 1 failed，暴露项目输入后旧结果残留；修复后进一步用锚点编辑测试复现同类问题并改为 input 事件立即失效。
- Hybrid 起点展示先以失败断言证明缺失，再加入 seed method 和实际起点显示。
- 独立 UI 审查发现回答失败后成功重试仍残留旧错误，新增失败复现再修复；审查确认关闭。路由独立审查无待修复问题。
- JavaScript syntax 使用 node --check；Python lint 与全量测试使用仓库 quality-check.ps1，包含真实 Neo4j 与 opt-in 浏览器测试。最终结果：535 passed（含 14 项浏览器验收），1 个已有弃用提示，67.83 秒；lint 和 JS syntax 通过。

本阶段不做 H/S/G 对照质量评估、不切换模型、不部署公网；OpenSpec 6.2–6.5 留待最终评估/验收阶段。


### Multi-Epic refinement (2026-09-08)

Added a snapshot-backed panorama showing 8 Epic clusters, all 35 issues, 27 CHILD_OF edges, and the one recorded BLOCKS edge. Both endpoints of that BLOCKS edge are Done; the line records a relationship, not an active unresolved blocker. No cross-Epic links were invented. Evidence screenshot: `data/graph-poc/demo-artifacts/panorama.png` (local ignored artifact).

TDD covered full/bounded overview projections, exclusion from answer evidence, and browser checks for complete graph counts, zoom/Fit, Epic drill-down, and dependency navigation pinned to the displayed panorama despite older detail state. Browser failures exposed canvas clipping, non-clickable SVG shapes, and stale snapshot selection; all received targeted fixes before the complete quality run.

Final refinement verification: JavaScript syntax and lint passed; complete suite with live Neo4j and browser checks enabled: **542 passed**, one existing multipart warning, 66.53 seconds. All 17 browser cases passed within that run.
