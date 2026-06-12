"""Brain — the integration: memory (smriti) + affect (bhava) + cognition (manas).

The loop:
    perceive() -> appraise -> feel -> remember (emotion-tagged, flashbulb boost)
                -> attend (working memory)
    think()    -> mood + focus + recalled memories + drives + values
                  => a cognitive-context block you inject into ANY LLM prompt.
    act_guard() -> conscience check before executing anything risky.

The LLM stays the language cortex; this is everything around it.
"""

from __future__ import annotations

from smriti.bhava import Bhava, Personality
from smriti.embeddings import cosine
from smriti.manas import Drives, SelfModel, WorkingMemory
from smriti.memory import MemoryEngine


class Brain:
    def __init__(self, path: str = ":memory:", personality: Personality | None = None,
                 engine: MemoryEngine | None = None):
        self.engine = engine or MemoryEngine(path)
        self.bhava = Bhava(self.engine.store, personality)
        self.wm = WorkingMemory()
        self.drives = Drives(self.engine)
        self.soul = SelfModel.load(self.engine.store)

    # ------------------------------------------------------------------ #
    def _novelty(self, content: str, scan: int = 50) -> float:
        vec = self.engine.embedder.embed(content)
        rows = self.engine.store.conn.execute(
            "SELECT embedding FROM events ORDER BY ts DESC, id DESC LIMIT ?", (scan,)
        ).fetchall()
        if not rows:
            return 1.0
        best = max(cosine(vec, self.engine.store.load_vec(r["embedding"])) for r in rows)
        return round(max(0.0, 1.0 - best), 3)

    def perceive(self, content: str, tags: list[str] | None = None, importance: float = 0.5,
                 task: str | None = None, success: bool | None = None,
                 ts: float | None = None) -> int:
        """One call = full sensory intake: appraisal, emotion, memory, attention."""
        novelty = self._novelty(content)
        # Novelty appraisal is context-sensitive: a novel FAILURE is alarming,
        # not delightful — curiosity only fires outside negative outcomes.
        if success is None or success:
            self.bhava.appraise_event(content, novelty, ts=ts)
        elif novelty > 0.8:
            self.bhava.feel("surprise", 0.4, cause=f"unexpected failure mode: {content[:50]}", ts=ts)

        # Memory gets tagged with the emotion THIS event caused (encoding-time
        # affect), not whatever happens to dominate the global mood.
        if task is not None and success is not None:
            streak = self.bhava.record_outcome(task, success, ts=ts)
            if success:
                emo = "pride" if self.bhava._fails_before_last_success(task) >= 2 else "joy"
            else:
                emo = "anger" if streak >= 3 else "frustration"
            inten = max(0.3, self.bhava.emotion_state(ts).get(emo, 0.0))
        else:
            emo, inten = self.bhava.dominant_emotion(ts)
            if inten < 0.2:
                emo = ""

        eid = self.engine.remember_event(
            content, importance=importance, tags=tags, ts=ts,
            emotion=emo, arousal=inten,
        )
        self.wm.attend(content, weight=0.5 + inten, ts=ts)
        return eid

    # ------------------------------------------------------------------ #
    def think(self, query: str, k: int = 4) -> str:
        """Inner monologue context: paste this block into the LLM prompt."""
        mood = self.bhava.mood()
        emo, inten = self.bhava.dominant_emotion()
        lines = [
            "## Internal state",
            f"- mood: {mood['label']} (valence {mood['valence']}, arousal {mood['arousal']}, dominance {mood['dominance']})",
            f"- dominant emotion: {emo or 'none'} ({inten})" ,
            "",
            "## Current focus (working memory)",
        ]
        lines += [f"- {i}" for i in self.wm.focus(4)] or ["- (empty)"]
        lines += ["", f"## Relevant memories for: {query!r}"]
        hits = self.engine.recall(query, k=k)
        lines += [f"- [{h.kind}] {h.text}" for h in hits] or ["- (none)"]
        motivations = self.drives.motivations()[:2]
        if motivations:
            lines += ["", "## Active drives"]
            lines += [f"- {m['drive']} (urgency {m['urgency']}): {m['suggestion']}" for m in motivations]
        lines += ["", "## Values (always binding)"]
        lines += [f"- {v['principle']}" for v in self.soul.values[:3]]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def act_guard(self, action_text: str) -> dict:
        """Conscience + emotion check before an action. Values veto absolutely;
        high anger adds a cooldown advisory even for allowed actions."""
        violations = self.soul.check_action(action_text)
        anger = self.bhava.emotion_state().get("anger", 0.0)
        advisory = ""
        if anger > 0.6:
            advisory = ("anger is high (%.2f): pause, re-read the goal, prefer reversible steps "
                        "or ask the user before proceeding" % anger)
        return {"allowed": not violations, "violations": violations, "advisory": advisory}

    def mood_report(self) -> dict:
        return {"mood": self.bhava.mood(), "emotions": self.bhava.emotion_state(),
                "working_memory": self.wm.focus(), "motivations": self.drives.motivations(),
                "memory_stats": self.engine.stats()}

    def close(self) -> None:
        self.engine.close()
