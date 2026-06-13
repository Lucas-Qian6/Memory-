# Memory @ DeepResearch — Timeline & 进度

> **目标（6/30）**：把 MVP 推进到「能接进真实 DR pipeline、能用、可评测」。
> **原则**：① 接入优先；② 每个 phase 有可运行产物；③ 所有访问只走 `MemoryManager`，底层可换。
> **配套**：能力说明 → [memory-capability.md](memory-capability.md)；落地结论 → [memory-schema-findings.md](memory-schema-findings.md)。

## 现状速览（6/13）

P0 接入 ✅ · P1 存储分层 ✅ · P2 向量召回 + 近重合并 ✅ · P3 一致性 + A/B 评测 ✅（已出数）
**余下**：P4 集成联调。（注：已移除离线/mock 脚手架，关系判定与规划/抽取/综合全流程必须真实 LLM）

## 阶段


| Phase            | 日期      | 状态  | 可运行产物（要点）                                                                                                                                                                   |
| ---------------- | ------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P0 接入契约**      | 6/11–13 | ✅   | pipeline 端到端只走 `MemoryManager`；`demo/run_demo.py` 跑通                                                                                                                        |
| **P1 存储分层**      | 6/14–18 | ✅   | working=`state.md`、episodic=`episodic.jsonl`、semantic=`semantic.json`+原文`artifacts/`+向量 sidecar；扩字段 `evidence/uri/step/status/content_hash`；recency 改 **step 序**；写入**内联同步** |
| **P2 向量召回 + 压缩** | 6/19–23 | ✅  | 向量召回 ✅（`sentence-transformers`，缺失退词法）；分级压缩 distill + top-k + 字符预算 + provenance 不压 ✅；**近重 claim 合并 ✅**（LLM 判同断言 → `store.merge_into` 并源）                                                                       |
| **P3 一致性 + 评测**  | 6/24–27 | ✅  | A/B eval harness ✅（`eval/`，process 指标 + LLM-judge，逐 run 落盘）；**一致性 ✅**：`status` + `supersedes`/`conflicts_with`，写时 LLM 判矛盾、读时过滤                                                                  |
| **P4 集成联调**      | 6/28–30 | ⬜   | 对接 长程任务/精准溯源（若就绪）的集成 demo + 短报告                                                                                                                                             |


## 评测结果（6/13，真实 API + Claude-4.6-opus，13/21 对）

memory ON vs OFF（同一 pipeline，只换 `MemoryManager`，OFF = 无外部记忆的有限上下文窗口）：

- **证据覆盖**：judge coverage ON 胜 **12/13**；引用来源 **7.4 vs 4.2**。
- **效率**：检索次数 **9.4 vs 14.1**；LLM 输入 **−30%**（≈128k vs 183k 字符）。
- **faithfulness / coherence**：基本打平（略偏 ON）。
- **结论**：记忆主要赢在「同等注入预算下覆盖更多相关来源 + 更省检索/token」。

## 关键产物

- `demo/run_demo.py` — 单任务记忆增强 loop（mock / 真实 API / `--no-llm`）。
- `demo/server.py` + `demo/webui/` — 记忆检视器（live read→decide→write trace + artifacts 链接）。
- `eval/` — A/B 评测（`run_eval.py` 编排、`arms.py` 的 OFF 消融、`judge.py` 盲对评分；结果落 `eval/results/`）。

## 风险与缓冲

- **真实 pipeline 未就绪** → P4 退化为「在真实 API 上跑完整 eval」（已做到）。
- **召回后端** → 本地 `sentence-transformers` + 词法 hybrid 保留（召回层）；但关系判定与规划/抽取/综合**必须真实 LLM**，已移除离线/mock 兜底（不再「无 key 也能跑」）。
- **进度** → P2 近重合并、P3 一致性已完成；仅余 P4 集成联调。

