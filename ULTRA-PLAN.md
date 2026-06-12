# ULTRA-PLAN — Smriti se "AI Brain" tak

> Vision: **LLM = language cortex. Smriti = hippocampus (memory). Bhava = limbic system (affect). Manas = executive (attention, drives, values).**
> Honest note: ye sab **functional simulation** hai — emotions yahan signals hain jo salience, strategy aur tone modulate karte hain. Sentience/consciousness ka claim nahi hai, aur "kab hoga" wala har statement speculation hai.

**Safety invariant (non-negotiable):** Emotions *suggest*, values *veto*. Anger/rage kabhi destructive action unlock nahi karta — high anger = cooldown + ask-for-help, by construction (`Bhava.suggest_strategy`, `SelfModel.check_action`).

---

## ✅ Phase A — DONE (v0.2.0, aaj build hua)

- **Bhava (affect):** PAD mood model, 10 emotions with separate half-lives (anger 8h, surprise 30min), OCC-style appraisal, failure-streak → frustration → anger escalation, empathy + emotional contagion, Big Five personality (baseline mood + reactivity)
- **Manas (cognition):** 7±2 working memory with activation decay + displacement, intrinsic drives (curiosity/competence/consistency), SelfModel "soul" with values guard
- **Brain:** `perceive()` (appraise→feel→remember→attend), `think()` (LLM-injectable cognitive context), `act_guard()` (conscience)
- **Memory capacity v1:** flashbulb encoding (arousal boosts importance), gist compression (old episodes → summaries, originals archived), schema migration, full brain-state export/import
- 25 tests, MCP server with 20 tools

---

## Phase B — Emotion-aware everything (1-3 mahine)

1. **LLM appraisal:** heuristic appraisal ko full OCC model se replace karo (`llm(prompt)` hook pehle se design me hai). Event → {goal-relevance, agency, expectedness} → emotion. ~2 hafte.
2. **Mood-congruent recall:** chhota scoring weight — anxious mood me risk-related memories thoda upar aayein (humans me proven effect; weight 0.05 rakhna, warna bias loop banega).
3. **Dream mode:** consolidation++ — random episode replay + recombination se "what-if" synthesis aur creative skill variants. Sote hue seekhna.
4. **Habituation:** same stimulus baar-baar → response damp (novelty system ka extension).
5. **Social memory:** per-person profile — facts + emotional history + interaction style per contact. Companion/assistant agents ka core feature.
6. **Personality drift:** experience se traits ka SLOW, bounded update (e.g. repeated success → +confidence/dominance baseline). Caps zaroori.

## Phase C — Memory capacity ULTRA (3-6 mahine)

| Ab | Upgrade | Capacity |
|---|---|---|
| Brute-force cosine scan | **sqlite-vec / hnswlib ANN index** | ~50k → **5M+ memories** |
| HashEmbedder (lexical) | **sentence-transformers local** ya API embeddings | semantic depth |
| Flat facts table | **Knowledge graph layer** (entities + temporal edges, Graphiti-style) | multi-hop reasoning |
| Text-only | **Multimodal memories** (CLIP-style image/audio embeddings) | dekha-suna sab yaad |
| Single SQLite file | **Postgres + pgvector backend option** | 100M+, multi-agent shared memory |
| Plain file | **Encryption at rest + memory ACLs** | trust layer |

Tiering target: **hot** (working memory, ms) → **warm** (active SQLite, <50ms) → **cold** (gists + archive, lazy load). Human memory bhi aise hi layered hai.

## Phase D — Metacognition (6-12 mahine) *(speculation-heavy, par direction sahi hai)*

1. **Confidence calibration:** har recall/claim ke saath "main kitna sure hoon" — aur uska track record.
2. **Episodic future thinking:** act se pehle imagined rollouts — memory se simulate karke best path chuno.
3. **Theory of mind:** user ka model — beliefs, goals, current mood prediction (social memory ke upar).
4. **Value learning:** user feedback se values refine (bounded, auditable — values silently change nahi hone chahiye).
5. **Benchmarks:** LongMemEval (memory), custom self-improvement bench (day-1 vs day-30), aur ek "EQ eval" (empathy detection accuracy, strategy quality under frustration).

---

## Kya kya AUR add kar sakte ho (idea bank)

- **Boredom drive:** low novelty lambi der tak → proactively naye tasks suggest kare
- **Energy/budget awareness:** token/cost ko "thakaan" ki tarah model karo — thaka agent shortcuts nahi, REST leta hai (defer)
- **Mood journaling:** daily mood summary user ko — agent ka "how was your day"
- **Emotional memory search:** "wo dikha jab main frustrated tha" — emotion-filtered recall (columns ready hain)
- **Multi-agent empathy:** dusre agents ke emotion states padhna (agent teams ke liye)
- **Grief/attachment model:** project khatam → mild sadness → graceful handover behavior
- **Circadian rhythm:** time-of-day se arousal baseline modulate (raat me conservative decisions)

## North star

2045 hindsight *(speculation)*: jo agents jeete wo sabse smart nahi the — wo the jo **yaad rakhte the, seekhte the, aur jinke paas brakes the**. Memory + affect + values = wahi teeno. Tum already us raaste pe ho.
