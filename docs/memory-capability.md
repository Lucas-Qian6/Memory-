# DeepResearch 单任务 Memory 能力（任务内工作记忆）

> **是什么**：DeepResearch 链路里的 **memory 能力**（与「精准溯源」「长程任务」并列），只聚焦**单个长程任务内部**的工作记忆。
> **目标**：在一次交互触发、持续十几分钟、需多次操作的研究任务里，让 agent **不丢目标、不重复劳动、不丢结论**。
> **范围**：不跨会话、不做长期知识库；任务结束记忆即弃。
> **配套**：进度 → [timeline.md](timeline.md)；字段与落地结论 → [memory-schema-findings.md](memory-schema-findings.md)。

---

## 1. 动机：单一对话上下文的三个失败模式

- **上下文溢出**：检索结果 / 论文片段 / 中间结论不断累积，早期内容被挤出窗口，后半程开始「遗忘」。
- **重复劳动**：长任务多次重规划，agent 忘了搜过什么 → 反复检索，浪费 token 与时间。
- **丢目标 / 结论散落**：目标、子问题、提纲、claim 只活在上下文里，跑到后面就丢失、难汇总成报告。

## 2. 核心问题与三个 facet

> **如何在有限上下文下、于单个长任务内，持续维护工作记忆——存任务状态、避免重复劳动、留住关键结论？**

| facet | 解决的失败模式 | 例子 | 粒度 |
| --- | --- | --- | --- |
| **working** | 丢目标 | 研究问题、子问题、提纲、进度快照 | 一次状态快照 |
| **episodic** | 重复劳动 | 已检索词、已读论文、已执行动作 | 一次动作 |
| **semantic** | 结论散落 / 被挤出 | 论文里的 claim、证据句、来源 | 一条 claim |

**关键**：三者不是三套独立系统，而是**同一 `MemoryStore` 里带 `type` 标签**的条目，靠**写入策略**（记什么、如何去重）与**读取策略**（何时召回、如何排序）区分。

## 3. 设计与实现

```text
pipeline (规划 → 检索 → 阅读 → 综合)
   │  仅通过统一接口读写
   ▼
MemoryManager  (remember / recall / seen_action / update_state / get_state / archive)
   │
   ▼
MemoryStore (facade：单存储 + type 标签)
   ├─ backends : working=state.md | episodic=episodic.jsonl | semantic=semantic.json + artifacts/ + 向量 sidecar
   ├─ policies : 写入(选择/去重/关联) + 读取(召回/排序/组装)
   └─ retrieval: 词法 + 可选向量(step 新近度)
```

- **Schema `MemoryItem`**：`id / type / content / source / tags / keywords / links / evidence / uri / step / status / content_hash / task_id / created_at`。
- **写路径**：抽取 claim（**不存整段原文，原文落 `artifacts/`**）→ `classify`（精确 `content_hash` 去重 → embedding 近邻候选 → **LLM judge** 判 合并/取代/冲突）→ 合并并 union `sources`（多源佐证）/ 标注 `supersedes`·`conflicts_with`，**不丢弃、不删除**。
- **读路径**：按 query 做 **hybrid 召回**（向量 + token重叠程度）+ **优先取后面写** → 只放行 `status=active`（过时 `superseded` / 合并 `merged` 留档不召回）→ 取 **top-k + 字符限制**，**引用完全保留**。
- **增强循环**：根据目前的进度+已经有的claim规划 → 检索 `seen_action` 查episodic防止重复搜索 + `remember` 沉淀 → 写报告时 recall top-k semantic → 更新目前进度。

## 4. 

| 主题 | 状态 | 落地 |
| --- | --- | --- |
| 写入策略（选择 / 去重 / 关联） | ✅ | `policies.WritePolicy`、`pipeline._paper_links` |
| 读取策略（相关 + 新近 + top-k 压缩） | ✅ | `policies.ReadPolicy`、`retrieval.py`、`pipeline._evidence_context` |
| 粒度（claim / 动作 / 快照） | ✅ | `schema.py` + `pipeline.py` |
| 干净接口（可换后端） | ✅ | `MemoryManager` + `store.py` facade + `backends.py` |
| 存什么形式 / 字段 / 何时存用 | ✅ | `schema.py` + `backends.py` + [memory-schema-findings.md](memory-schema-findings.md) |
| 向量召回 | ✅ | `embeddings.py`（本地 `sentence-transformers`，缺失退词法） |
| 是否有用 / 如何帮长任务 | ✅（部分跑） | `eval/` ON vs OFF：**coverage 12/13、来源 +3.2、LLM 输入 −30%** |
| 一致性 / 时效（过时 / 矛盾） | ✅ | `status`(active/superseded/merged) + `links.supersedes`/`conflicts_with`；写时 LLM 判矛盾、读时只放行 active（`policies.classify` + `manager.remember`） |
| 近重 claim 合并（压缩） | ✅ | 近重 LLM 判同断言 → 合并为 canonical + union `sources`（多源佐证），留档不丢弃（`store.merge_into`） |
| 集成联调（接真实模块） | ⬜ | P4 |

## 5. 下一步

- **P2 ✅**：近重 claim 合并——合并而非丢弃，union 来源以支持「多源佐证」引用。
- **P3 ✅**：任务内一致性——`supersedes` / `conflicts_with` 标注 + 写时 LLM 判定 + 读时过滤。
- **架构变更**：关系判定一律走 LLM（无确定性兜底），已移除全仓库离线/mock 脚手架，运行必须有真实搜索 API + LLM。
- **P4**：对接 长程任务 / 精准溯源 模块的集成 demo。
- **演进**：已按 facet 分库；后续可换向量库后端或加 entity store，调用方不动。

