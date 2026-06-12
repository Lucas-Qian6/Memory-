# 记忆设计结论：以什么形式存、什么时候存、什么时候用

> 本文是配合 [`memory-capability.md`](memory-capability.md) 的**落地结论**。
> 做法：把 demo 从假数据换成**真实的 wenyon `/api/v1/search` + LLM 驱动的研究循环**
> （`plan → search → extract → reflect → synthesize`），记忆仍然只经
> `MemoryManager` 读写。然后真实跑一遍（gpugeek/Claude 网关），**观察记忆实际
> 存了什么、何时存、何时取**，据此回答三个问题：**什么字段、什么形式
> （markdown / 向量库）、什么时候存与用**。
>
> 相关代码：[`demo/search_client.py`](../demo/search_client.py)（真实 API 适配器）、
> [`demo/llm.py`](../demo/llm.py)（规划/抽取/反思/综合）、
> [`demo/pipeline.py`](../demo/pipeline.py)（记忆增强循环）。
>
> **实现状态（6/12 更新）**：本文结论已落地为**分层存储 + 本地向量召回**——
> working→`state.md`、episodic→`episodic.jsonl`、semantic→`semantic.json` + 原文落
> `artifacts/`，claim 向量化（`sentence-transformers`，缺失则退回词法）。新增字段
> `evidence / uri / step / status / content_hash`；recency 改为 **step 序**；写入保持**内联同步**。
> 详见 [`src/memory_dr/backends.py`](../src/memory_dr/backends.py)、
> [`src/memory_dr/embeddings.py`](../src/memory_dr/embeddings.py)。

---

## 1. 真实搜索 API 返回了什么（以 paper 为例）

一次 `bizTypes:["paper"]` 调用，`SearchResponse` 顶层是
`{traceId, total, page, pageSize, papers[], scholars[], patents[], facets, workerTrace, fallback}`，
每个 `papers[]` 条目的真实字段（来自 dev 机实跑）：

```json
{"docId":"817338009555828737","titleZh":"医学教育中的人工智能","titleEn":"Artificial intelligence in medical education",
 "snippet":"<摘要式长文本>","authors":["Keng Siau | Weiyu Wang"],"venue":"Medical Teacher _ ",
 "year":null,"doi":"10.1080/...","arxivId":null,"subjects":null,"citationCount":177,
 "isOpenAccess":false,"score":211.45,"scoreDetails":{"recallScore":...,"finalScore":211.45}}
```

判断哪些**值得进记忆**、哪些**丢弃**：

- 值得留：`docId`（稳定 ID）、`title`、`snippet`（既是抽 claim 的来源、又是证据句）、
  以及引用所需元数据 `authors / year / venue / doi / url / citationCount`。
- 丢弃：`score / scoreDetails`（检索内部打分，对下游研究无意义）、`highlights`、
  `isOpenAccess`、以及 `subjects` 那个**被逗号拆碎的脏 JSON**（清洗后只留 `name`）。

> 适配器把所有脏活隔离在一个地方（[`demo/search_client.py`](../demo/search_client.py) 的
> `_parse_paper`）：`authors` 按 `" | "` 拆平、`venue` 去掉 `" _ "` 垃圾后缀、
> `year` 容忍 `null`、从 `subjects` 碎片里正则捞回 `name`、由 `doi/arxivId` 推出 `url`。
> 所以**核心 schema 一个字段都不用改**，scholar/patent 将来非空时只动对应 `_parse_*`。

---

## 2. 记忆该存什么字段（按 facet）

`MemoryItem` 的通用字段（`content/type/source/tags/keywords/links/task_id/created_at`）
已经够用，三个 facet 靠**约定**区分。下面是真实跑出来的存储项：

**semantic（一条 = 一个 claim）** — 真实存储示例：

```json
{"content":"Incorporating reaction context ... improves retrosynthesis prediction accuracy.",
 "type":"semantic","source":"900000000000000002","tags":["chemical sciences"],
 "links":{"title":"Reaction-Context-Aware Retrosynthesis Prediction",
          "authors":["A. Researcher","B. Coauthor"],"year":null,
          "venue":"Journal of Chemical Information and Modeling - JCIM",
          "doi":"10.1021/...","url":"https://doi.org/10.1021/...",
          "evidence":"<支撑该 claim 的原文句>","citationCount":12,"docId":"900000000000000002"}}
```

- `content` = 一句自洽的 claim；`source` = `docId`（稳定，`doi` 作次选）；
  `tags` = API 的 `subjects` 名称或 query 主题；
  `links` = **写报告引用所需的一切**（标题/作者/年份/期刊/doi/url/证据句/被引数），
  这样综合阶段无需回查原文即可成文带引用。粒度：**一条 claim**。

**episodic（一条 = 一次检索动作）** — 真实存储示例：

```json
{"content":"retrosynthesis prediction reaction context influence","type":"episodic",
 "source":"planner","links":{"bizTypes":["paper"],"hits":2}}
```

- `content` = **检索词本身**（不带 `search:`/`bizTypes` 等模板噪声，否则会污染判重的关键词重叠）；
  `links` 记录 `bizTypes` 与命中数。粒度：**一次动作**。

**working（一条 = 一次状态快照）**：`content` = `goal + plan`（任务开始）/`progress`（每轮结束），
`source="state"`。粒度：**一次快照**，最新一条即当前进度。

---

## 3. 以什么形式存：markdown 还是向量库？

**结论：不是单一选择，按 facet 分。** “全 markdown”和“全向量库”都不对。

| facet | 存储形态 | 去重方式 | 召回方式 | 需要向量库吗 |
| --- | --- | --- | --- | --- |
| working | **markdown 文本快照**（goal/提纲/草稿/进度） | 覆盖式（取最新一条） | 直接取最新 `get_state()` | 否 |
| episodic | **追加式结构化日志**（query + bizTypes + hits + ts） | 关键词/字符串相似（`seen_action`） | 按 query 判“做过没” | 否 |
| semantic | **结构化记录**（claim + 来源 + 证据 + 引用元数据），原文落 `artifacts/` | content_hash 精确 + 关键词 jaccard 去重 | 向量+词法 hybrid + step 新近度 top-k | **是（已实现：本地 embeddings + 词法 fallback）** |
| 最终报告 | markdown（**是产物，不是记忆**） | — | — | — |

要点：

- **working 用 markdown**：目标、子问题、提纲、草稿、进度天然是人读文本，覆盖式追加即可，不需要向量。
- **episodic 用结构化日志**：它的用途只是“这条 query 是否做过”，靠字符串/关键词判重就够，
  既不需要 markdown 也不需要向量。
- **semantic 用结构化记录**：MVP 阶段 JSON + 关键词召回已可用；**它是唯一真正受益于向量库的 facet**
  —— claim 的“语义相似召回”比关键词更准。演进时把 semantic 接向量库：`content` 向量化，
  `links/source/tags` 作为 payload/metadata 一起存，关键词召回升级/补充为向量召回。
  因为所有访问都过 `MemoryManager`，这只是**换存储后端**，调用方不动。

---

## 4. 什么时候存（write path）

- **episodic：动作发生时立即写**（在抽取之前），这样后续重规划能马上判重。
- **semantic：每个结果抽出 claim 后立即写**，且**选择性 + 去重**——
  丢弃过短内容、按关键词 jaccard 合并近重复。真实跑里 3 次检索抽出 ~17 条、去重后留 14 条。
  > **关键**：存的是 **claim 而不是整段 `snippet`**。否则记忆会变成“第二个被污染的上下文”，
  > 这正是 selective write 要避免的。
- **working：任务开始写 `goal/plan`；每轮结束写 `progress` 快照**，把进度移出 prompt。

---

## 5. 什么时候用（read path）

- **episodic：每次动作前** `seen_action(query)` 判重 → 跳过重复检索。
  （离线跑：**3 次真跑 / 6 次被跳过**。）
- **semantic：两处召回**
  1. **规划时**召回已知 findings 喂给 planner → 让 LLM 发现 gap、不重复探索。
     （真实 LLM 跑：第 2 轮 `decide_followups` 判断已覆盖 → **提前停**。）
  2. **综合时** top-k + compress → 只把精炼后的少量内容注入 prompt 写带引用的报告。
     （真实跑：留存 14 条、综合只注入 **765 字符**。）
- **working：每步开始** `get_state()` 重新锚定目标/进度，长任务不丢线索。

---

## 6. 这次跑出来的观察（指标）

| 模式 | steps | 真跑 | 跳过 | working / episodic / semantic | 留存 vs 注入 |
| --- | --- | --- | --- | --- | --- |
| `--mock --no-llm`（离线确定性） | 9 | 3 | **6** | 4 / 3 / 10 | 1264 → 845 字符 |
| `--mock`（真实 LLM，gpugeek） | 3 | 3 | 0（早停） | 2 / 3 / 14 | 1585 → 765 字符 |

解读：**episodic 防重复劳动**（离线 6 次跳过），**semantic 留存 > 注入**（对抗上下文溢出），
**working 维持目标**（每轮快照、末轮仍守住最初 goal）。真实 LLM 跑因反思早停所以 0 跳过，
但 episodic 始终在场——只要 planner 提出近重复 query 就会被拦。

---

## 7. 演进建议

- **semantic 接向量库**：✅ 已实现——claim 经 `sentence-transformers` 向量化存入 sink 的
  `{id: vector}` sidecar，召回为「向量 cosine + 词法」hybrid（缺包则纯词法）。`links` 仍作
  引用 payload，原文落 `artifacts/`、`uri` 指回。见 [`src/memory_dr/retrieval.py`](../src/memory_dr/retrieval.py)。
- **scholar / patent**：待真实非空响应后补 `_parse_scholar` / `_parse_patent`（其余逻辑与字段无关）。
- **时效与冲突**（capability 文档挑战三）：同一 claim 多来源、被后续证据推翻的旧结论——
  可在 `links` 加 `conflicts` 标注或加更新策略。
- **真实 API 跑法**：在 dev 机 / 隧道里设 `SEARCH_API_BASE_URL` 后
  `python demo/run_demo.py "<你的问题>"`，再用真实响应校准 `_parse_paper`。
