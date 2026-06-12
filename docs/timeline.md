# Memory @ DeepResearch — 6 月开发 Timeline

> **起点（6/11）**：MVP 已可离线跑通 —— `working / episodic / semantic` 单库 + 词法检索 +
> 单任务 demo（`python demo/run_demo.py`）。
> **目标（6/30）**：把 MVP 推进到「**能接进真实 DR pipeline、能用、可评测**」。
> **原则**：① 接入优先（先能接进去，再优化）；② **每个 phase 结束都有可运行产物**；
> ③ 所有访问只走 `MemoryManager`，底层换实现不影响调用方。

---
## 阶段一览

| Phase | 日期 | 目标 | 可运行产物 | 退出标准 |
| --- | --- | --- | --- | --- |
| **P0 接入契约** | 6/11–6/13 | 让现有 `MemoryManager` 不重写就能挂进真实 loop | 薄 adapter + 一张 **tool→artifact→memory** 映射表 + KV 安全的 `assemble_context`（append-only）；demo 仍跑通 | 真实 pipeline 能在 mock 语料上端到端调用 `remember/recall/seen_action/update_state` |
| **P1 存储分层** | 6/14–6/18 | 给每个 facet 合适的后端 + 修任务内 recency | 扩字段（`evidence`/`uri`/`step`/`status`/`content_hash`）；episodic→JSONL 追加、semantic→SQLite/JSON + 原文落 `artifacts/`、working→markdown 状态文件；recency 改为 **step 序** | 长跑后原文不进 live context，落盘可检视 |
| **P2 向量召回 + 压缩** | 6/19–6/23 | 更好的召回 + 控溢出 | semantic 上可选 embeddings 索引（保留词法 fallback）；**分级压缩**（写入即抽取丢原文、读取 top-k+预算、近重 claim 合并、provenance 不压） | 向量开关可用；报告「注入 token vs 留存 token」 |
| **P3 一致性 + 评测** | 6/24–6/27 | 处理时效/矛盾 + 证明价值 | `supersedes`/`conflicts_with` 标注 + 更新路径；**A/B eval harness**（memory on/off） | 跑出：重复检索节省、context 占用、目标连贯性、证据覆盖 |
| **P4 集成联调** | 6/28–6/30 | 接真实模块 + 收尾 | 对接 长程任务/精准溯源（若就绪）的集成 demo + 结果短报告 + 缓冲 | 一个接近真实的任务跑通整条记忆链路 |

---

## 各阶段细化

### P0 接入契约（6/11–6/13）
- **做什么**：写一个薄 adapter，把 plan/search/read/synthesize 各步映射到 `MemoryManager` 调用；
  固化 `assemble_context` 为 **append-only（召回内容只拼到尾部，不改前缀）**，保护 KV cache。
- **产物（可运行）**：`demo/run_demo.py` 照常跑；新增一张 tool→artifact→memory 映射表（doc）。
- **退出标准**：真实 pipeline 可直接调 `MemoryManager`，无需触碰底层 store。

### P1 存储分层（6/14–6/18）
- **做什么**：`MemoryItem` 扩字段；按 facet 落盘（episodic=追加日志、semantic=结构化+原文 `artifacts/`、
  working=markdown）；**把 72h wall-clock recency 改成 step 序**（单任务仅 ~15 分钟，时间几乎不区分）。
- **产物（可运行）**：新 schema 下跑通，原文留磁盘、不进上下文，store 可检视。
- **退出标准**：长任务跑完，live context 不被原文撑爆。

### P2 向量召回 + 压缩（6/19–6/23）
- **做什么**：semantic 上加 embeddings 索引（可选依赖，缺失则退回词法）；落地分级压缩策略
  （写入抽取丢原文 / 读取 top-k+字符预算 / 近重合并 / **provenance 不压**）。
- **产物（可运行）**：向量召回开关；压缩前后 token 量化指标。
- **退出标准**：能报出「留存 vs 注入」的 token 对比。

### P3 一致性 + 评测（6/24–6/27）
- **做什么**：claim 加 `status`/`supersedes`/`conflicts_with` 与更新路径；搭 A/B eval（memory on/off）。
- **产物（可运行）**：eval harness 一键出数。
- **退出标准**：四个指标有数：重复检索↓、context 占用、目标连贯性、证据覆盖。

### P4 集成联调（6/28–6/30）
- **做什么**：接真实 长程任务/精准溯源 模块（若就绪）；写结果短报告；留缓冲吸收延期。
- **产物（可运行）**：集成 demo + 报告。
- **退出标准**：一个接近真实的任务端到端跑通。

---

## 关键里程碑（checkpoints）

- **6/13** — 接入契约就绪：真实 loop 能调记忆。
- **6/18** — 存储分层就绪：原文落盘、上下文不溢出。
- **6/23** — 向量 + 压缩就绪：召回更准、token 可控。
- **6/27** — 评测出数：有/无记忆的对比数字。
- **6/30** — 集成联调 + 报告。

---

## 风险与缓冲

- **真实 pipeline 未就绪**：P4 退化为「加固 adapter + 在 mock loop 上跑完整 eval」，前序阶段不受影响（皆走 `MemoryManager`）。
- **向量后端选型未定**：默认本地 `sentence-transformers` + 词法 fallback，保住「无 key 也能跑」。
- **进度滑动**：P4 自带缓冲；P2 的向量为可选项，必要时可后置，不阻塞 P3 评测。
