# Smriti 🧠

[![CI](https://github.com/thelonewolf1337-code/smriti/actions/workflows/ci.yml/badge.svg)](https://github.com/thelonewolf1337-code/smriti/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Zero dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)](pyproject.toml)

> *Smriti (स्मृति) — Sanskrit for "memory".*

**Portable memory + self-improvement loop for AI agents** — now with a brain: simulated affect (**Bhava**: emotions, mood, personality) and cognition (**Manas**: working memory, drives, values). Framework-agnostic, ships as an MCP server, zero required dependencies (pure Python + SQLite).

> Emotions here are functional signals (appraisal → salience, strategy, tone), not claims of sentience. Safety invariant: **emotions suggest, values veto** — anger maps to cooldown + help-seeking, never destructive action. See `ULTRA-PLAN.md` for the full roadmap.

Most memory tools (Mem0, Zep, Letta) remember **facts**. Hermes Agent's skill loop learns **procedures** — but it's locked inside Hermes. Smriti does both, for any agent that speaks MCP: Claude, OpenClaw, Hermes, your own.

## The three layers

| Layer | Stores | Example |
|---|---|---|
| **Episodic** | What happened (raw events) | "deploy failed, .env was missing" |
| **Semantic** | What is true, *and since when* (temporal facts) | `ankesh / lives in / Bangalore` — supersedes Mumbai |
| **Procedural** | How to do things (skill docs) | `handle-deploy`: steps, failure modes, verification |

Plus the glue that makes it a *learning loop*:

- **Consolidation (reflection):** a cron-able pass that turns recent events into facts + skill docs. LLM-pluggable (`llm(prompt) -> str`, works with Claude/Ollama/anything) with an offline heuristic fallback.
- **Hybrid recall:** `0.45·vector + 0.25·keyword + 0.20·recency + 0.10·importance`. Superseded facts never surface.
- **Forgetting is a feature:** importance-weighted decay archives stale noise; hard `forget()` for right-to-delete, all audited.
- **Progressive disclosure:** agents read skill summaries first, expand the full procedure only when executing.
- **Memory portability:** full JSON export/import; embeddings regenerate on import, so memory outlives any embedding model.

## Quickstart

```bash
pip install -e ".[dev,mcp]"
python examples/demo.py        # 3 days in an agent's life (memory loop)
python examples/brain_demo.py  # one emotional day (affect + strategy + rage guard)
pytest                         # or: python tests/run_tests.py (zero deps)
```

### The brain in 10 lines

```python
from smriti import Brain, Personality

b = Brain("~/.smriti/memory.db", personality=Personality(openness=0.7, neuroticism=0.4))
b.perceive("deploy failed: docker build error", task="deploy", success=False)  # feels frustration
b.perceive("deploy failed again", task="deploy", success=False)
b.bhava.suggest_strategy("deploy")   # -> {"strategy": "switch-approach", ...}
b.bhava.empathize("wtf I am so angry!!")  # -> acknowledge first, solve second
b.act_guard("wipe the server in revenge") # -> {"allowed": False, ...} values veto rage
print(b.think("fix the deploy?"))    # mood + focus + memories + drives + values -> LLM prompt
```

```python
from smriti import MemoryEngine, Consolidator

eng = MemoryEngine("~/.smriti/memory.db")
eng.remember_event("deploy failed: .env missing — copy .env.example first", tags=["deploy"])
eng.save_fact("ankesh", "lives in", "Bangalore")     # supersedes any older value
Consolidator(eng).run()                              # reflection: events -> facts + skills
eng.recall("how do I deploy")                        # ranked hits across all layers
```

### As an MCP server (any agent)

```json
{ "mcpServers": { "smriti": { "command": "smriti-mcp", "env": { "SMRITI_DB": "~/.smriti/memory.db" } } } }
```

Memory tools: `remember_event`, `save_fact`, `recall`, `get_skill`, `save_skill`, `record_skill_use`, `consolidate`, `forget`, `export_memory`, `memory_stats`.
Brain tools: `perceive`, `think`, `mood`, `feel`, `suggest_strategy`, `empathize`, `motivations`, `act_guard`, `compress_memory`, `soul`.

### Plug a real LLM into reflection

```python
import anthropic
client = anthropic.Anthropic()
llm = lambda p: client.messages.create(model="claude-sonnet-4-6", max_tokens=2000,
                                       messages=[{"role": "user", "content": p}]).content[0].text
Consolidator(eng, llm=llm).run()
```

### Swap embeddings

Default `HashEmbedder` is deterministic and offline. For semantic depth, implement `embed(text) -> list[float]` with sentence-transformers or any API and pass it: `MemoryEngine(path, embedder=MyEmbedder())`.

## Benchmarked self-improvement

Most memory systems claim agents "learn". Smriti measures it (`python benchmarks/self_improvement.py`, runs in CI):

| Metric | Day 1 (no memory) | Day 30 (with memory) |
|---|---|---|
| Steps to complete 5 ops tasks | 20 (trial & error) | **5** (skill recall) |
| Skill recalled as top hit | — | **5/5 tasks** |

Repetition also strengthens: recalled memories gain importance (testing effect), and repeated events consolidate into one stronger memory instead of duplicates.

## Local-first LLM hooks (optional)

```python
from smriti import Brain, Consolidator, MemoryEngine, OllamaEmbedder, ollama_llm

eng = MemoryEngine("~/.smriti/memory.db", embedder=OllamaEmbedder())  # semantic recall
Consolidator(eng, llm=ollama_llm("llama3.2")).run()                   # smarter reflection
b = Brain(engine=eng, guard_llm=ollama_llm("llama3.2"))               # paraphrase-proof conscience
```

Everything still works with zero dependencies if you skip these — the hooks are plain `text -> text` / `text -> vector` callables, so any provider fits.

## Design positions (opinionated)

1. **Temporal facts, not vector soup.** "Moved to Bangalore" must supersede "lives in Mumbai". Pure vector stores fail this; Smriti's `(subject, predicate)` slots with `valid_from/valid_to` don't.
2. **Procedures ≠ facts.** Skills carry steps, failure modes, and verification — and track their own success rate, which feeds their retrieval rank.
3. **Measure improvement or don't claim it.** `record_skill_use` + stats exist so you can benchmark day-1 vs day-30 on the same task.
4. **Memory belongs to the user.** Export everything, import anywhere, delete with an audit trail.

## Roadmap

Full phased plan in **`ULTRA-PLAN.md`** — highlights: LLM-driven appraisal, dream mode (replay + recombination during consolidation), social memory (per-person profiles), ANN index for millions of memories, knowledge-graph facts, multimodal memories, confidence calibration, LongMemEval + self-improvement + EQ benchmarks.

MIT licensed. Built by Ankesh.
