## Why

固定 Epic 分组图能展示层级，但跨 Epic 依赖难以整体探索。新增可切换网状布局，让 BA/PM 直观看到记录关系，同时保留原分组视图。

## What Changes

- 在同一全景加入 Grouped / Network 切换，不重新检索或切换快照。
- Network 自动排列节点，保持 Epic 归属可辨识；可拖动节点、平移、缩放和恢复布局。
- 选择节点突出直接相连的真实关系和邻居，支持清除高亮与原有下钻。
- 稳定布局，无持续动画；不从视觉距离推断业务关系。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `backlog-graph-demo`: 新增分组/网状视图切换及节点探索要求。

## Impact

仅静态全景渲染和浏览器测试、操作说明。复用 overview 的 nodes/edges，API、Neo4j、Chroma、SQLite 与检索逻辑不变。无 Jira 写回，无远程图形依赖，无模型调用。原 PoC 归档和原始产品目标保持不变。
