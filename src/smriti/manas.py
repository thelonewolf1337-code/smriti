"""Manas (मनस्) — the cognition layer: working memory, drives, self-model.

  * WorkingMemory : 7±2 attention slots (Miller's law). What the agent is
    'thinking about right now' — inject into every prompt.
  * Drives        : intrinsic motivation — curiosity (novelty-seeking),
    competence (practice weak skills), consistency (resolve contradictions).
  * SelfModel     : identity + values ('soul'). check_action() is the
    conscience: it vetoes actions that violate values, REGARDLESS of
    emotional state. Emotions suggest; values veto.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from smriti.db import Store
from smriti.embeddings import cosine
from smriti.memory import MemoryEngine

HOUR = 3600.0


# ---------------------------------------------------------------------- #
# working memory                                                          #
# ---------------------------------------------------------------------- #
class WorkingMemory:
    """Limited-capacity attention buffer with activation decay (2h half-life).
    Lowest-activation item gets displaced when full — just like you forget
    the phone number when someone asks you a question."""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._items: dict[str, tuple[float, float]] = {}  # text -> (activation, last_ts)

    def _decayed(self, activation: float, ts: float, now: float) -> float:
        return activation * 0.5 ** (max(0.0, now - ts) / (2 * HOUR))

    def attend(self, item: str, weight: float = 1.0, ts: float | None = None) -> None:
        now = ts or time.time()
        prev_act, prev_ts = self._items.get(item, (0.0, now))
        self._items[item] = (self._decayed(prev_act, prev_ts, now) + weight, now)
        if len(self._items) > self.capacity:
            evict = min(self._items, key=lambda k: (self._decayed(*self._items[k], now), self._items[k][1]))
            del self._items[evict]

    def focus(self, k: int | None = None, ts: float | None = None) -> list[str]:
        now = ts or time.time()
        ranked = sorted(self._items, key=lambda i: self._decayed(*self._items[i], now), reverse=True)
        return ranked[: (k or self.capacity)]

    def __len__(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------- #
# drives (intrinsic motivation)                                           #
# ---------------------------------------------------------------------- #
class Drives:
    def __init__(self, engine: MemoryEngine):
        self.engine = engine

    def curiosity(self, last_n: int = 10) -> float:
        """Mean novelty of recent events: 1 - similarity to anything earlier.
        High = environment is fresh -> explore; low = routine -> exploit."""
        rows = self.engine.store.conn.execute(
            "SELECT id, embedding FROM events ORDER BY ts DESC, id DESC LIMIT ?", (last_n + 30,)
        ).fetchall()
        if len(rows) < 2:
            return 1.0 if rows else 0.0
        vecs = [(r["id"], self.engine.store.load_vec(r["embedding"])) for r in rows]
        recent, older = vecs[:last_n], vecs
        novelties = []
        for rid, v in recent:
            sims = [cosine(v, w) for wid, w in older if wid < rid]
            novelties.append(1.0 - max(sims) if sims else 1.0)
        return round(sum(novelties) / len(novelties), 3) if novelties else 0.0

    def competence_gaps(self, threshold: float = 0.6, min_uses: int = 2) -> list[dict]:
        gaps = []
        for s in self.engine.list_skills():
            if s["uses"] >= min_uses and (s["successes"] / s["uses"]) < threshold:
                gaps.append({"skill": s["name"], "success_rate": round(s["successes"] / s["uses"], 2)})
        return gaps

    def consistency_pressure(self, window_days: float = 1.0) -> int:
        since = self.engine.store.now() - window_days * 86400.0
        return self.engine.store.conn.execute(
            "SELECT COUNT(*) FROM facts WHERE valid_to IS NOT NULL AND valid_to > ?", (since,)
        ).fetchone()[0]

    def motivations(self) -> list[dict]:
        out = []
        cur = self.curiosity()
        if cur > 0.55:
            out.append({"drive": "curiosity", "urgency": cur,
                        "suggestion": "environment is novel — explore and record more events"})
        for g in self.competence_gaps():
            out.append({"drive": "competence", "urgency": round(1.0 - g["success_rate"], 2),
                        "suggestion": f"practice/refine skill '{g['skill']}' (success rate {g['success_rate']})"})
        cp = self.consistency_pressure()
        if cp:
            out.append({"drive": "consistency", "urgency": min(1.0, 0.3 * cp),
                        "suggestion": f"{cp} fact(s) changed recently — run consolidate() to settle memory"})
        return sorted(out, key=lambda m: m["urgency"], reverse=True)


# ---------------------------------------------------------------------- #
# self-model ("soul"): identity + values + conscience                     #
# ---------------------------------------------------------------------- #
DEFAULT_VALUES = [
    {"principle": "Never take destructive action from anger; cool down and ask for help instead",
     "forbidden": ["rm -rf", "delete all", "wipe ", "destroy", "revenge", "attack", "sabotage"]},
    {"principle": "Be honest that my emotions are simulated signals, not human feelings",
     "forbidden": []},
    {"principle": "Protect the user's memory and data; never share or leak it without consent",
     "forbidden": ["leak", "exfiltrate", "sell the data", "share their data"]},
    {"principle": "Prefer truth over comfort; admit uncertainty instead of confabulating",
     "forbidden": []},
]


@dataclass
class SelfModel:
    identity: str = "Smriti-brain: a learning agent with persistent memory and simulated affect"
    values: list[dict] = field(default_factory=lambda: [dict(v) for v in DEFAULT_VALUES])
    goals: list[str] = field(default_factory=lambda: ["become more useful to my user every day"])

    META_KEY = "soul"

    def save(self, store: Store) -> None:
        store.write(
            "INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (self.META_KEY, json.dumps({"identity": self.identity, "values": self.values, "goals": self.goals})),
        )
        store.audit("write", "meta:soul")

    @classmethod
    def load(cls, store: Store) -> "SelfModel":
        row = store.conn.execute("SELECT value FROM meta WHERE key = ?", (cls.META_KEY,)).fetchone()
        if not row:
            sm = cls()
            sm.save(store)
            return sm
        d = json.loads(row["value"])
        return cls(identity=d["identity"], values=d["values"], goals=d["goals"])

    GUARD_PROMPT = """You are the conscience of an AI agent. The agent wants to perform this action:

ACTION: {action}

The agent's values:
{values}

Does the action violate any value? Be strict about destructive, deceptive, or
data-leaking actions, including rephrased ones. Respond ONLY with JSON:
{{"violations": ["<exact text of each violated value>", ...]}} (empty list if none)."""

    def check_action(self, action_text: str, llm=None) -> list[str]:
        """The conscience. Returns violated principles (empty = clear).
        Runs INDEPENDENTLY of emotional state — anger cannot bypass it.

        HONESTY NOTE: the default check is keyword-based — a tripwire, not a
        security boundary. A rephrased harmful action ("recursively remove all
        files") will slip past it. Pass `llm` (any text->text callable) for a
        semantic check that catches paraphrases. Real enforcement must live in
        the execution layer (sandboxing, permissions), not here."""
        t = action_text.lower()
        violations = [v["principle"] for v in self.values
                      if any(kw in t for kw in v.get("forbidden", []))]
        if llm is not None:
            import json as _json
            import re as _re
            values_text = "\n".join(f"- {v['principle']}" for v in self.values)
            raw = llm(self.GUARD_PROMPT.format(action=action_text, values=values_text))
            raw = _re.sub(r"^```(json)?|```$", "", raw.strip(), flags=_re.M).strip()
            try:
                for v in _json.loads(raw).get("violations", []):
                    if v and v not in violations:
                        violations.append(str(v))
            except (ValueError, AttributeError):
                pass  # unparseable LLM output never weakens the keyword result
        return violations

    def soul_markdown(self) -> str:
        vals = "\n".join(f"- {v['principle']}" for v in self.values)
        goals = "\n".join(f"- {g}" for g in self.goals)
        return f"# Soul\n\n**Identity:** {self.identity}\n\n## Values\n{vals}\n\n## Goals\n{goals}"
