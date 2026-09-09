# Jira Graph PoC：本地准备

> 2026-09-09 最新收尾与待办验收见 [PoC 交接记录](../evaluation/jira-graph-poc-handoff.md)；GPT-5.5 已打通真实生成，旧模型 404 为历史验证结果。

现已包含只读 Jira 采集、三索引构建、快照发布和结构查询 API；组合检索、证据回答 API 与默认关闭的轻量演示页也已实现。

## 工作区与 Python

在 `.worktrees/jira-backlog-graph-poc` 的 `codex/jira-backlog-graph-poc` 分支工作。
本次 `.venv` 使用 `--system-site-packages` 复用已有应用依赖，单独安装 ruff 0.8.0 与 neo4j 5.28.2，没有修改全局 Python。它隔离新增工具，但不是完全独立的依赖环境。
全新环境可运行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt -r requirements-graph.txt
```

## Neo4j

使用 Community 5.26.12 和 Python driver 5.28.2。服务器镜像拉取 digest 为
`sha256:9f75e8df4325a24f00fdd7a8c0bcce650a58375049b1058e496e8b43d6c36b37`。
版本是本轮固定的兼容基线，不宣称是最新版本。
参考 [官方 Docker 安装](https://neo4j.com/docs/operations-manual/current/docker/introduction/)
与 [Python driver 兼容说明](https://neo4j.com/docs/python-manual/current/upgrade/)。

创建未跟踪的 `.env.graph`，包含 `GRAPH_NEO4J_PASSWORD=<本地随机密码>`，至少 8 个字符。
本次已生成随机密码；不要提交、打印或复用 Jira/OpenAI 凭据。

```powershell
docker compose --env-file .env.graph -f compose.graph.yaml config --quiet
docker compose --env-file .env.graph -f compose.graph.yaml up -d
docker compose --env-file .env.graph -f compose.graph.yaml ps
```

浏览器为 `http://127.0.0.1:7474`，Bolt 为 `bolt://127.0.0.1:7687`，用户为 `neo4j`。
仅绑定本机，数据在独立 Compose volume。停止时保留数据：

```powershell
docker compose --env-file .env.graph -f compose.graph.yaml down
```

不使用 `down -v`，不清理其他数据库。

## 测试

工作区没有复制主目录的 `.env` 或已有数据。测试不需要真实账号。

```powershell
$env:PATH = "$PWD/.venv/Scripts;" + $env:PATH
$env:OPENAI_API_KEY = 'test-key'
powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1
```

`GRAPH_ENABLED=false`、`GRAPH_DEMO_ENABLED=false` 是默认值；单独打开演示而不打开图功能会校验失败。
`GRAPH_NEO4J_PASSWORD` 使用 SecretStr，避免配置 repr 暴露。
## 构建 AIPLAT 快照

从工作区执行；显式读取主目录的凭据文件，不复制它，也不改变其内容：

```powershell
.venv/Scripts/python.exe -m app.graph.cli --project AIPLAT --env-file C:/SourceCode/ai-rag-service/.env
```

CLI 只允许 AIPLAT，读取项目当前账号可见事项，创建新的 snapshot_id。凭据文件需要 JIRA_URL、JIRA_EMAIL、JIRA_API_TOKEN、OPENAI_API_KEY；图密码仍从 `.env.graph` 读取。采集和 embedding 会访问远程 API，Jira 不被修改。

默认 `--data-dir data/graph-poc` 下按站点/项目与快照的短 hash 隔离，包含 raw.json、normalized.json、chunks.json、manifest.json、build-report.json、chroma/ 和 lexical.db。这些私有数据被 Git 忽略。首次默认服务索引不被读取或迁移。

只有 Neo4j、Chroma 和 FTS 全部核验成功才原子切换 active.json。旧快照保留；失败目录包含安全的阶段标识 failed.json，当前 active 不变。禁止旧采集覆盖更新的已发布快照。目录名使用短 hash 避免 Windows 原生数据库路径限制，读取仍校验完整 scope。

同一项目构建用 build.lock 排他。进程异常退出可能留下锁；先核实没有构建进程，再人工移除该项目的锁文件。不要删除默认数据库或猜测性清理其他目录。

启动真实 Neo4j 集成测试（仅创建并清理独立合成测试 scope）：

```powershell
$env:GRAPH_INTEGRATION = '1'
.venv/Scripts/python.exe -m pytest tests/test_graph -q
```

关闭该变量时，Neo4j 集成测试显式跳过；Chroma/FTS 测试仍使用本地临时数据库和合成向量，不调用 OpenAI。实时构建的 build-report.json 记录实际 embedding token，合成测试不伪装为真实模型评估。

## 结构查询 API

在本地 `.env.graph` 中补充 `GRAPH_ENABLED=true`、`GRAPH_SITE_URL=https://30156758.atlassian.net` 和 `OPENAI_API_KEY=test-key`。结构查询不调用模型，测试 key 仅满足现有应用配置要求。

从工作区启动：

```powershell
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --env-file .env.graph
```

Swagger：<http://127.0.0.1:8001/docs>。前台运行用 Ctrl+C 停止。

- `GET /graph/projects/AIPLAT/overview`：全项目确定性统计及各 Epic 的直接子事项数量。
- `GET /graph/projects/AIPLAT/epics/AIPLAT-13`：Epic 自身状态、完整子事项计数及分页列表；`offset` 默认 0，`limit` 默认 50、最大 100。
- `GET /graph/projects/AIPLAT/issues/AIPLAT-46/dependencies`：`direction=inbound|outbound|both`、`hops=1|2`、`limit` 最大 100；`relation_type` 默认 BLOCKS。`unresolved_pair=true` 只保留双方均处于已知未完成状态的阻塞关系。

三个端点均支持可选 `snapshot_id`；省略时每个请求固定一次 active 版本。未知项目/事项/显式快照返回 404，非法参数返回 422，活动快照或图数据不可用返回 503。覆盖和截断通过响应元数据明确标识。

计数不依赖 top-k 或当前页。Epic 状态与子事项状态分别展示。Done/Done 的 BLOCKS 保留为历史记录，不表达当前风险。接口仅供本机 PoC 使用。

## 组合检索与证据回答

Swagger 现包含 `POST /graph/retrieve` 与 `POST /graph/query`。二者都是无会话调用。`retrieve` 不调用回答模型；无明确 key 时会为问句调用 embedding。`query` 在相同检索结果上生成并校验引用；失败时保留 data/structure，返回 `answer.status=answer_unavailable`。

使用真实 embedding 时，从工作区以前台进程加载已有凭据，不复制文件（先停止本次后台 API 以释放 8001）：

```powershell
.venv/Scripts/python.exe -c "from dotenv import dotenv_values; import os, uvicorn; os.environ['OPENAI_API_KEY']=dotenv_values('C:/SourceCode/ai-rag-service/.env')['OPENAI_API_KEY']; uvicorn.run('app.main:app', host='127.0.0.1', port=8001, env_file='.env.graph')"
```

这只读取源文件的 OpenAI key；Jira 采集仍需显式构建命令，查询不访问 Jira。结构查询可继续使用测试 key。2026-09-09 按用户要求将回答生成默认模型切换为 `gpt-5.5-2026-04-23`，已通过真实依赖问答与引用校验。若本地环境显式设置了 `ANSWER_OPENAI_MODEL`，请同步更新该值并重启服务。Embedding 与 reranker 配置不变，无需重建索引。验证详情见 `docs/evaluation/jira-graph-gpt55-validation.md`。

无 ticket key 的实际通过示例，POST 到 `/graph/retrieve`：

```json
{
  "project_key": "AIPLAT",
  "query": "retrieval security unauthorized data access",
  "intent": "dependencies",
  "top_k": 3,
  "direction": "both",
  "hops": 1
}
```

冻结快照中的结果：hybrid 起点 AIPLAT-46 → BLOCKS → AIPLAT-16，文本来自同快照的两方文档。该结果由实际搜索得到，没有硬编码 ticket key。

Epic 关系示例：

```json
{
  "project_key": "AIPLAT",
  "query": "Which Bugs under this Epic have recorded blocking links to Stories?",
  "intent": "dependencies",
  "epic_key": "AIPLAT-13",
  "direction": "outbound",
  "hops": 1
}
```

`structure.epic_membership` 提供 CHILD_OF 归属来源；`data.paths` 提供依赖路径。Epic 下钻与 overview 意图返回确定性 `structure`，不让模型计算统计。

- 支持固定 intent：overview、epic_detail、dependencies。调用者选择意图；自然语言不驱动任意图查询或自动多轮规划。
- `epic_key` 与 `issue_key` 互斥。问句内明确 Jira key 也按精确锚点处理；未知 key 404，不回退到其他相似事项。
- `top_k` 范围 1–20，限制 hybrid 候选 chunks，去重后可能少于该数量的事项。依赖覆盖相对于已选起点；不代表全项目依赖都已搜索。
- 多起点共用最多 100 节点、1000 路径预算；每路径最多两跳。Epic 先取直接子事项，层级这一步与依赖跳数分开。
- `text_budget` 为字符数，默认 16000、最大 32000。回答适配器另限正文 24000 字符；截断明确标记 partial。关系证据不因文本排序/top-k 丢失。
- 图不可用时仅返回同一已发布快照的文本，标记 graph_unavailable，生成端拒绝据此声称完整关系。没有活动快照时 503，不读取默认服务索引。
- 引用校验检查证据 ID、逐句引用、来源绑定及禁止模型生成 URL；它不等价于自动证明每项语义判断正确。当前真实生成成功率/人工正确率未验收。

若命令行 HTTP 客户端受系统代理影响，请将 localhost/127.0.0.1 排除代理；Python 本机探针可使用 `httpx.Client(trust_env=False)`。不要因此修改远程请求的代理配置。

## 演示页面（P9）

入口：<http://127.0.0.1:8001/graph/demo>。本次本机实例已开启；新环境默认关闭。在被 Git 忽略的 `.env.graph` 中加入 `GRAPH_DEMO_ENABLED=true`，保留 `GRAPH_ENABLED=true`，重启上文的 uvicorn 命令即可。关闭演示开关后页面及资源均为 404，平台 graph API 仍可单独使用。

页面是三个独立视图，不保存会话，不自动追问，不写回 Jira；打开页面不会调用生成模型，也不会自动发起检索。

1. **Epic overview → Load overview**：查看全快照计数和 8 个 Epic。每行分别展示 Epic 自身状态、直接子事项数量、子事项完成比例；点击 **Inspect Epic** 在页内下钻，点击 ticket key 打开 Jira 来源。
2. **Epic detail → Load Epic**：可独立输入 AIPLAT-13，查看其 4 个直接子事项；分页固定首次返回的快照。与已有概览关联下钻时沿用概览快照。
3. **Dependency evidence → Retrieve evidence**：可独立查询 AIPLAT-46，也可选择 Epic key 查看 AIPLAT-13 的依赖。箭头按源记录方向绘制；展示双方状态、关系字段/方向/link ID 与正文来源。勾选 **Only pairs with known unfinished statuses** 后，Done/Done 这条关系不再匹配；空态不宣称现实中不存在依赖。

额外能力演示：在第三个视图点击 **Use security preset**，再 **Retrieve evidence**。预设只包含业务描述，不包含 ticket key。真实浏览器已验证显示 Hybrid search → Starting issues: AIPLAT-46 → BLOCKS → AIPLAT-16。该操作调用 query embedding；不会调用回答模型。

**Generate grounded answer** 是可选的独立按钮，固定已经展示的快照。当前模型 404 限制仍在，失败时页面保留图与文本。浏览器测试中的成功回答/重试恢复由注入响应验证，不代表真实模型成功。证据与诊断中的技术 ID 收在可展开详情中。

页面会显示快照范围、时间、覆盖/截断、错误和来源。切换项目或编辑锚点会立即清除旧结果；迟到的响应不会恢复旧项目的数据。来源链接仅允许快照站点的 HTTPS URL；正文按文本渲染。

### 可选浏览器验收

浏览器依赖不影响静态页面运行。已安装 Playwright 1.57.0、Chromium 143.0.7499.4；新测试环境可执行：

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-demo.txt
.venv/Scripts/python.exe -m playwright install chromium
```

先按上述方式启动本地页面和已发布的 AIPLAT 快照，再执行：

```powershell
$env:PATH = "$PWD/.venv/Scripts;" + $env:PATH
$env:OPENAI_API_KEY = 'test-key'
$env:GRAPH_INTEGRATION = '1'
$env:GRAPH_DEMO_BROWSER = '1'
node --check app/graph/static/demo.js
powershell -ExecutionPolicy Bypass -File ./scripts/quality-check.ps1
```

`GRAPH_DEMO_BROWSER` 未设置时浏览器测试跳过；Python/API 测试照常运行。浏览器测试包含真实只读结构/显式关系调用，以及注入响应的错误、截断、转义、快照固定和重试场景；不会真实调用回答模型。真实业务描述的 embedding 演示另有一次手动验收记录，不在每次测试中重复付费调用。

截图位于被忽略的 `data/graph-poc/demo-artifacts/`。界面在 1440px 桌面和 390px 窄屏检查过；宽表格及关系图在各自区域横向滚动。


### Multi-Epic panorama

Reload `/graph/demo`, choose Overview, then Load overview. The panorama draws the same snapshot's 8 Epics and all 35 issues, with 27 CHILD_OF arrows and one dashed orange BLOCKS arrow. Select a node for its title/status and an Epic or dependency drill-down. Use +/−, drag the background, and Fit to navigate. Tables remain below the graph. No cross-Epic dependency is inferred.

The overview response includes `data.graph` (`nodes`, `edges`, `total_nodes`, `total_edges`, `truncated`), capped at 100 nodes and 500 edges independently of complete overview counts. Graph query evidence excludes this visual projection to avoid duplication.


## 最终评估重跑与回滚

完整结果见 [最终报告](../evaluation/jira-graph-final-report.md)。问题集固定冻结快照和 normalized SHA；重建后若要评估新快照，应重新标注问题集，不能复用旧快照的预期数量。

从本 worktree 运行（第一条会调用 query embedding，第二条会尝试一次当前生成模型；都不访问或写入 Jira）：

```powershell
.venv/Scripts/python.exe scripts/evaluate-graph-poc.py --env-file C:/SourceCode/ai-rag-service/.env
.venv/Scripts/python.exe scripts/graph-acceptance.py --normalized data/graph-poc/9f54ac536147f3064f730c8c/snapshots/84651bb20502364a91fcc0ee/normalized.json --output docs/evaluation/jira-graph-final-acceptance.json
```

评估的 H 使用同快照 hybrid/exact 适配，S 使用直接 parent/filter，G 使用实际 graph retrieval。查询 embedding 预热后共用，正文预算均为 16000 字符；结构证据额外返回，不将这项对照称为相等总上下文的生成评估。真实与合成场景分别记录。计费金额、真实生成正确率等不可得字段保持 null/未测。

功能回滚：停止当前本地 API，将 `GRAPH_ENABLED=false` 和 `GRAPH_DEMO_ENABLED=false` 写入本机未跟踪配置，再按启动命令重启服务。两个开关必须一起关闭；旧 /retrieve、/query 不依赖 Neo4j。若需停止 Neo4j，运行 `docker compose --env-file .env.graph -f compose.graph.yaml down`，保留 volume，不加 `-v`。

数据回退：旧快照完整保留，可先在平台 API 显式传入旧 `snapshot_id` 查询核验，不改 active 指针。重建失败会自动保留旧 active。当前发布器禁止旧采集覆盖新版本，没有提供管理员强制 active 回退命令；不要直接手改 manifest 或绕过校验。需要恢复实时数据时用上文只读 CLI 新建完整快照。

本次验收保留本地 API 和 Neo4j 运行供演示；未实际关闭功能、删除数据、提交、推送或部署。生产权限与运维不在本 PoC 范围内。


## 2026-09-09 跨 Epic 演示快照

当前 active 为 `7b9d816aa2c445c9919635aff149f705`：35 事项、27 CHILD_OF、2 BLOCKS。新增 Jira 记录为 AIPLAT-37（Epic AIPLAT-36）BLOCKS AIPLAT-23（Epic AIPLAT-17）；两端均 To Do，符合 unresolved_pair 筛选，不自动表示延期风险。只读重新采集并构建了三个索引，未写回 Jira。昨天的冻结评估快照/标注/结果保留原版本，不将本次数据变化混入旧报告。

刷新演示页并 Load overview 可看到跨 Epic 连线；第三个视图输入 AIPLAT-37，选择 outbound，可核对 AIPLAT-23 及关系来源。关系线可能经过其他分组区域，仅箭头两端表示实际关系。实时核验记录见 ../evaluation/jira-graph-cross-epic-live.json。

概览的无父事项标题改为 **No parent (not assignee)**；图中未进入 Epic 分组的区域标题为 **No Epic grouping / other hierarchy**。两者均不表示负责人 assignee 是否分配。


## Grouped / Network 全景切换

Load overview 后，全景右上角提供 **Grouped**（默认分组）与 **Network**（网状）按钮。切换只重新排列当前快照，不发送查询、不生成或更改关系。

- Network 自动排列后静止。Epic 使用较大矩形，Story/Bug 为圆形；节点颜色继续表示状态，事项下方显示归属 Epic（或其他父事项）。
- 拖动节点调整布局，连线同步跟随；拖动背景平移，用 +/− 缩放。**Fit** 显示全部节点，**Reset layout** 恢复初始排列。
- 单击节点或聚焦后 Enter/Space，突出该节点、直接邻居及关联边。**Clear selection** 恢复全部显示；选择只是高亮，不删除证据或筛选检索结果。
- 下方详情显示标题、状态、父事项和 Jira 来源，**Inspect Epic / Inspect dependencies** 沿用当前快照。
- 跨 Epic 示例：切换 Network，点击 AIPLAT-37，橙色 BLOCKS 箭头指向 AIPLAT-23。归属分别为 AIPLAT-36 和 AIPLAT-17。图形距离没有业务语义，连线交叉也不是新增节点；必要时拖动节点改善观看角度。

布局在浏览器做固定步数的有界计算，不使用外部 CDN、模型或持续物理动画。保留已有 100 节点/500 边数据预算和截断提示。小屏通过画布区域滚动与缩放查看。原 API 和数据索引没有变化。
