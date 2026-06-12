"""Consolidation (reflection): episodic events -> semantic facts + skill docs.

Two modes:
  - LLM mode: pass any callable `llm(prompt: str) -> str` returning JSON.
    Works with Claude, local Ollama models, anything.
  - Heuristic mode (default): offline regex/frequency extraction, so the
    loop runs with zero dependencies. Good for tests and as a fallback.

This is the "self-improvement" half of the engine: experiences become
reusable knowledge while the agent sleeps (run it from a cron).
"""

from __future__ import annotations

import json
import re
from typing import Callable

from smriti.memory import MemoryEngine
from smriti.skills import SkillDoc

CONSOLIDATION_PROMPT = """You are the nightly reflection process of an AI agent's memory.
Below are the agent's episodic events from the last day.

Extract:
1. "facts": durable facts as triples {{"subject": ..., "predicate": ..., "object": ...}}.
   Only include things likely to still be true next month. Use lowercase subject/predicate.
2. "skills": for any repeated/multi-step procedure, a skill document:
   {{"name": "kebab-case-name", "purpose": ..., "triggers": [...], "steps": [...],
     "failure_modes": [...], "verification": [...]}}

Respond with ONLY valid JSON: {{"facts": [...], "skills": [...]}}

EVENTS:
{events}
"""

_FACT_RE = re.compile(
    r"^(?P<s>[\w][\w\s]{0,40}?)\s+"
    r"(?P<p>is|are|prefers|prefer|uses|use|lives in|live in|works at|work at|moved to)\s+"
    r"(?P<o>[^.!?]{1,80})",
    re.IGNORECASE,
)

# normalize predicates that imply a state change of the same slot
_PREDICATE_ALIASES = {
    "moved to": "lives in", "live in": "lives in",
    "prefer": "prefers", "use": "uses", "work at": "works at",
}


class Consolidator:
    def __init__(self, engine: MemoryEngine, llm: Callable[[str], str] | None = None):
        self.engine = engine
        self.llm = llm

    # -- public -------------------------------------------------------------
    def run(self, window_days: float = 1.0, min_skill_events: int = 3) -> dict:
        events = self.engine.recent_events(window_days=window_days)
        if not events:
            return {"facts_added": 0, "skills_added": 0, "events_seen": 0}

        if self.llm:
            facts, skills = self._extract_llm(events)
        else:
            facts = self._extract_facts_heuristic(events)
            skills = self._extract_skills_heuristic(events, min_skill_events)

        facts_added = 0
        for f in facts:
            try:
                self.engine.save_fact(f["subject"], f["predicate"], f["object"], source="consolidation")
                facts_added += 1
            except (KeyError, TypeError):
                continue

        skills_added = 0
        for s in skills:
            try:
                doc = s if isinstance(s, SkillDoc) else SkillDoc(
                    name=s["name"], purpose=s.get("purpose", ""),
                    triggers=s.get("triggers", []), steps=s.get("steps", []),
                    failure_modes=s.get("failure_modes", []),
                    verification=s.get("verification", []),
                )
                if self.engine.get_skill(doc.name) is None:  # don't clobber refined skills
                    self.engine.save_skill(doc)
                    skills_added += 1
            except (KeyError, TypeError):
                continue

        report = {"facts_added": facts_added, "skills_added": skills_added, "events_seen": len(events)}
        self.engine.store.audit("consolidate", "reflection", json.dumps(report))
        return report

    # -- llm mode -----------------------------------------------------------
    def _extract_llm(self, events: list[dict]) -> tuple[list, list]:
        lines = "\n".join(f"- [{e['ts']:.0f}] (tags={e['tags']}) {e['content']}" for e in events)
        raw = self.llm(CONSOLIDATION_PROMPT.format(events=lines))
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.M).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [], []
        return data.get("facts", []), data.get("skills", [])

    # -- heuristic mode -----------------------------------------------------
    def _extract_facts_heuristic(self, events: list[dict]) -> list[dict]:
        facts = []
        for e in events:
            for sentence in re.split(r"[.!?]\s*", e["content"]):
                m = _FACT_RE.match(sentence.strip())
                if not m:
                    continue
                p = m.group("p").lower()
                facts.append({
                    "subject": m.group("s").strip().lower(),
                    "predicate": _PREDICATE_ALIASES.get(p, p),
                    "object": m.group("o").strip(),
                })
        return facts

    def _extract_skills_heuristic(self, events: list[dict], min_events: int) -> list[SkillDoc]:
        by_tag: dict[str, list[dict]] = {}
        for e in events:
            for t in e["tags"]:
                by_tag.setdefault(t, []).append(e)

        skills = []
        for tag, evs in by_tag.items():
            if len(evs) < min_events:
                continue
            steps, seen = [], set()
            for e in evs:
                c = e["content"].strip()
                if c.lower() not in seen:
                    seen.add(c.lower())
                    steps.append(c)
            skills.append(SkillDoc(
                name=f"handle-{tag}",
                purpose=f"Procedure learned from {len(evs)} '{tag}' events",
                triggers=[tag],
                steps=steps[:7],
                failure_modes=["Steps were auto-extracted; refine after first manual use"],
                verification=["Confirm the end state matches what the original events achieved"],
            ))
        return skills
