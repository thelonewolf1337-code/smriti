# Changelog

## 0.3.0 — 2026-06-12

### Added
- **Self-improvement benchmark** (`benchmarks/self_improvement.py`): proves day-1 trial-and-error becomes day-30 one-step skill recall — 100% skill hit rate, 75% step reduction on the reference suite. Runs in CI as a regression gate.
- **Recall reinforcement** (testing effect): the top-ranked event gains a small importance bump on each retrieval, so frequently-used memories resist decay. Disable per-call with `recall(..., reinforce=False)`.
- **Content-hash dedup**: re-remembering the same event within 7 days strengthens the existing memory instead of duplicating it. Bypass with `remember_event(..., dedupe=False)`.
- **OllamaEmbedder**: real semantic embeddings via local Ollama (`nomic-embed-text`), zero API keys, optional graceful fallback.
- **`ollama_llm()` adapter**: one-line local-LLM hook for `Consolidator` (batteries-included reflection).
- **LLM-backed `act_guard`**: optional semantic conscience check (`Brain(guard_llm=...)`) that catches paraphrased violations the keyword tripwire misses. Unparseable LLM output never weakens the keyword result.
- **GitHub Actions CI**: test suite + benchmark on Python 3.10/3.11/3.12.

### Changed
- **Thread safety**: WAL journaling, `check_same_thread=False`, and a re-entrant store lock around every write/commit path. Concurrent-writer test added.
- **Recall hot path**: in-memory vector cache (no more JSON-parsing every embedding on every query); invalidated precisely on writes.
- **`decay()`** now also prunes spent affect rows (emotions >30d, outcomes >90d).

### Notes
- `act_guard` remains a tripwire, not a security boundary — enforcement belongs in the execution layer. This is now documented explicitly.

## 0.2.0 — 2026-06-12

- Bhava affect system (PAD mood, 10 emotions, personality, empathy, strategy escalation with safety invariant), Manas cognition layer (working memory, drives, soul/values), Brain facade (perceive/think/act_guard), flashbulb encoding, gist compression, brain-state export/import, 20 MCP tools.

## 0.1.0 — 2026-06-12

- Initial release: episodic/semantic(temporal)/procedural memory, hybrid recall, consolidation, decay, audit, export/import, MCP server.
