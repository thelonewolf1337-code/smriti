"""MemoryEngine — the facade over all three memory layers.

Hybrid recall score:
    0.45 * vector similarity
  + 0.25 * keyword overlap
  + 0.20 * recency (exponential decay, 7-day half-life)
  + 0.10 * importance
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smriti.db import Store
from smriti.embeddings import Embedder, HashEmbedder, cosine, keyword_overlap
from smriti.skills import SkillDoc

RECENCY_HALF_LIFE_DAYS = 7.0
W_VECTOR, W_KEYWORD, W_RECENCY, W_IMPORTANCE = 0.45, 0.25, 0.20, 0.10
DAY = 86400.0


@dataclass
class Hit:
    kind: str          # event | fact | skill
    ref: str           # id or skill name
    text: str          # what the agent should read
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref, "text": self.text, "score": round(self.score, 4)}


class MemoryEngine:
    def __init__(self, path: str | Path = ":memory:", embedder: Embedder | None = None):
        self.store = Store(path)
        self.embedder = embedder or HashEmbedder()

    # ------------------------------------------------------------------ #
    # episodic                                                            #
    # ------------------------------------------------------------------ #
    def remember_event(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        ts: float | None = None,
        emotion: str = "",
        arousal: float = 0.0,
    ) -> int:
        # Flashbulb effect: emotionally arousing events encode more strongly,
        # so they surface more easily in recall (importance feeds the ranking).
        if arousal > 0:
            importance = min(1.0, importance + 0.3 * min(1.0, arousal))
        vec = self.embedder.embed(content)
        cur = self.store.conn.execute(
            "INSERT INTO events(ts, content, tags, importance, embedding, emotion, arousal) VALUES (?,?,?,?,?,?,?)",
            (ts or self.store.now(), content, json.dumps(tags or []), importance,
             self.store.dump_vec(vec), emotion, arousal),
        )
        self.store.conn.commit()
        self.store.audit("write", f"event:{cur.lastrowid}", content[:80])
        return int(cur.lastrowid)

    def recent_events(self, window_days: float = 1.0, include_archived: bool = False) -> list[dict]:
        since = self.store.now() - window_days * DAY
        q = "SELECT * FROM events WHERE ts >= ?" + ("" if include_archived else " AND archived = 0")
        rows = self.store.conn.execute(q + " ORDER BY ts ASC", (since,)).fetchall()
        return [dict(r) | {"tags": json.loads(r["tags"])} for r in rows]

    # ------------------------------------------------------------------ #
    # semantic (temporal facts)                                           #
    # ------------------------------------------------------------------ #
    def save_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: str = "",
        ts: float | None = None,
    ) -> int:
        """Store a fact. A newer fact about the same (subject, predicate)
        supersedes the old one — the old row gets valid_to set, so state
        changes ('lived in London' -> 'lives in Tokyo') are handled."""
        now = ts or self.store.now()
        subject, predicate = subject.strip().lower(), predicate.strip().lower()

        # Skip exact duplicates of the currently valid fact.
        cur_fact = self.current_fact(subject, predicate)
        if cur_fact and cur_fact["object"].strip().lower() == obj.strip().lower():
            return int(cur_fact["id"])

        self.store.conn.execute(
            "UPDATE facts SET valid_to = ? WHERE subject = ? AND predicate = ? AND valid_to IS NULL",
            (now, subject, predicate),
        )
        vec = self.embedder.embed(f"{subject} {predicate} {obj}")
        cur = self.store.conn.execute(
            "INSERT INTO facts(subject, predicate, object, valid_from, source, embedding) VALUES (?,?,?,?,?,?)",
            (subject, predicate, obj, now, source, self.store.dump_vec(vec)),
        )
        self.store.conn.commit()
        self.store.audit("write", f"fact:{cur.lastrowid}", f"{subject} {predicate} {obj}")
        return int(cur.lastrowid)

    def current_fact(self, subject: str, predicate: str) -> dict | None:
        row = self.store.conn.execute(
            "SELECT * FROM facts WHERE subject = ? AND predicate = ? AND valid_to IS NULL "
            "AND archived = 0 ORDER BY valid_from DESC LIMIT 1",
            (subject.strip().lower(), predicate.strip().lower()),
        ).fetchone()
        return dict(row) if row else None

    def fact_history(self, subject: str, predicate: str) -> list[dict]:
        rows = self.store.conn.execute(
            "SELECT * FROM facts WHERE subject = ? AND predicate = ? ORDER BY valid_from ASC",
            (subject.strip().lower(), predicate.strip().lower()),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # procedural (skills)                                                 #
    # ------------------------------------------------------------------ #
    def save_skill(self, skill: SkillDoc) -> str:
        md = skill.to_markdown()
        vec = self.embedder.embed(f"{skill.name} {skill.purpose} {' '.join(skill.triggers)} {' '.join(skill.steps)}")
        now = self.store.now()
        self.store.conn.execute(
            "INSERT INTO skills(name, summary, markdown, created_ts, updated_ts, embedding) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET summary=excluded.summary, markdown=excluded.markdown, "
            "updated_ts=excluded.updated_ts, embedding=excluded.embedding",
            (skill.name, skill.summary(), md, now, now, self.store.dump_vec(vec)),
        )
        self.store.conn.commit()
        self.store.audit("write", f"skill:{skill.name}", skill.purpose[:80])
        return skill.name

    def get_skill(self, name: str, full: bool = False) -> str | None:
        """Progressive disclosure: summary by default, full markdown on demand."""
        row = self.store.conn.execute("SELECT * FROM skills WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return row["markdown"] if full else row["summary"]

    def list_skills(self) -> list[dict]:
        rows = self.store.conn.execute("SELECT name, summary, uses, successes FROM skills ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def record_skill_use(self, name: str, success: bool) -> None:
        self.store.conn.execute(
            "UPDATE skills SET uses = uses + 1, successes = successes + ? WHERE name = ?",
            (1 if success else 0, name),
        )
        self.store.conn.commit()
        self.store.audit("skill_use", f"skill:{name}", "success" if success else "failure")

    # ------------------------------------------------------------------ #
    # hybrid retrieval                                                    #
    # ------------------------------------------------------------------ #
    def recall(self, query: str, k: int = 5, kinds: tuple[str, ...] = ("event", "fact", "skill")) -> list[Hit]:
        qvec = self.embedder.embed(query)
        now = self.store.now()
        hits: list[Hit] = []

        def score(vec: list[float], text: str, ts: float, importance: float) -> float:
            recency = 0.5 ** ((now - ts) / (RECENCY_HALF_LIFE_DAYS * DAY))
            return (
                W_VECTOR * cosine(qvec, vec)
                + W_KEYWORD * keyword_overlap(query, text)
                + W_RECENCY * max(0.0, min(recency, 1.0))
                + W_IMPORTANCE * importance
            )

        if "event" in kinds:
            for r in self.store.conn.execute("SELECT * FROM events WHERE archived = 0").fetchall():
                s = score(self.store.load_vec(r["embedding"]), r["content"], r["ts"], r["importance"])
                hits.append(Hit("event", str(r["id"]), r["content"], s))

        if "fact" in kinds:
            for r in self.store.conn.execute("SELECT * FROM facts WHERE valid_to IS NULL AND archived = 0").fetchall():
                text = f"{r['subject']} {r['predicate']} {r['object']}"
                s = score(self.store.load_vec(r["embedding"]), text, r["valid_from"], 0.7)
                hits.append(Hit("fact", str(r["id"]), text, s))

        if "skill" in kinds:
            for r in self.store.conn.execute("SELECT * FROM skills").fetchall():
                success_rate = (r["successes"] / r["uses"]) if r["uses"] else 0.5
                s = score(self.store.load_vec(r["embedding"]), r["markdown"], r["updated_ts"], 0.4 + 0.6 * success_rate)
                hits.append(Hit("skill", r["name"], r["summary"], s))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    # ------------------------------------------------------------------ #
    # forgetting / decay                                                  #
    # ------------------------------------------------------------------ #
    def decay(self, half_life_days: float = 30.0, archive_below: float = 0.15) -> int:
        """Archive episodic events whose decayed importance has dropped below
        threshold. Forgetting is a feature: it keeps retrieval clean."""
        now = self.store.now()
        archived = 0
        for r in self.store.conn.execute("SELECT id, ts, importance FROM events WHERE archived = 0").fetchall():
            effective = r["importance"] * (0.5 ** ((now - r["ts"]) / (half_life_days * DAY)))
            if effective < archive_below:
                self.store.conn.execute("UPDATE events SET archived = 1 WHERE id = ?", (r["id"],))
                archived += 1
        self.store.conn.commit()
        if archived:
            self.store.audit("decay", "events", f"archived {archived}")
        return archived

    def compress(self, older_than_days: float = 30.0, summarizer=None, min_group: int = 2) -> dict:
        """Capacity upgrade: old episodic detail -> compact 'gist' memories.

        Mirrors human autobiographical memory: you keep the gist of last
        year, not every sentence. Old events are grouped by tag, summarized
        into one fresh gist event, and the originals are archived (kept on
        disk for audit/export, excluded from recall scans).

        summarizer: optional callable(list[str]) -> str (plug an LLM here).
        """
        cutoff = self.store.now() - older_than_days * DAY
        rows = self.store.conn.execute(
            "SELECT * FROM events WHERE ts < ? AND archived = 0 AND tags NOT LIKE '%\"gist\"%'",
            (cutoff,),
        ).fetchall()
        groups: dict[str, list] = {}
        for r in rows:
            tag = (json.loads(r["tags"]) or ["misc"])[0]
            groups.setdefault(tag, []).append(r)

        gists, archived = 0, 0
        for tag, grp in groups.items():
            if len(grp) < min_group:
                continue
            texts = [g["content"] for g in grp]
            if summarizer:
                summary = summarizer(texts)
            else:
                head = "; ".join(t[:80] for t in texts[:5])
                summary = f"Gist of {len(grp)} '{tag}' memories: {head}"
            emotions = [g["emotion"] for g in grp if g["emotion"]]
            self.remember_event(
                summary,
                importance=max(0.5, max(g["importance"] for g in grp)),
                tags=[tag, "gist"],
                emotion=max(set(emotions), key=emotions.count) if emotions else "",
                arousal=sum(g["arousal"] for g in grp) / len(grp),
            )
            gists += 1
            for g in grp:
                self.store.conn.execute("UPDATE events SET archived = 1 WHERE id = ?", (g["id"],))
                archived += 1
        self.store.conn.commit()
        report = {"gists_created": gists, "events_archived": archived}
        if gists:
            self.store.audit("compress", "events", json.dumps(report))
        return report

    def forget(self, kind: str, ref: str) -> bool:
        """Hard delete (user right-to-forget). Audited."""
        if kind == "event":
            cur = self.store.conn.execute("DELETE FROM events WHERE id = ?", (int(ref),))
        elif kind == "fact":
            cur = self.store.conn.execute("DELETE FROM facts WHERE id = ?", (int(ref),))
        elif kind == "skill":
            cur = self.store.conn.execute("DELETE FROM skills WHERE name = ?", (ref,))
        else:
            return False
        self.store.conn.commit()
        ok = cur.rowcount > 0
        if ok:
            self.store.audit("forget", f"{kind}:{ref}")
        return ok

    # ------------------------------------------------------------------ #
    # portability                                                         #
    # ------------------------------------------------------------------ #
    def export_json(self) -> dict:
        """Full memory export. Embeddings are intentionally excluded —
        they are regenerated on import so memory stays portable across
        embedding models."""
        ev = self.store.conn.execute("SELECT * FROM events").fetchall()
        fa = self.store.conn.execute("SELECT * FROM facts").fetchall()
        sk = self.store.conn.execute("SELECT * FROM skills").fetchall()
        em = self.store.conn.execute("SELECT * FROM emotions").fetchall()
        oc = self.store.conn.execute("SELECT * FROM outcomes").fetchall()
        me = self.store.conn.execute("SELECT * FROM meta").fetchall()
        strip = lambda r: {k: v for k, v in dict(r).items() if k != "embedding"}
        return {
            "smriti_export": 1,
            "exported_at": self.store.now(),
            "events": [strip(r) for r in ev],
            "facts": [strip(r) for r in fa],
            "skills": [strip(r) for r in sk],
            "emotions": [dict(r) for r in em],
            "outcomes": [dict(r) for r in oc],
            "meta": [dict(r) for r in me],
        }

    def import_json(self, data: dict) -> dict:
        assert data.get("smriti_export") == 1, "not a smriti export"
        counts = {"events": 0, "facts": 0, "skills": 0}
        for e in data.get("events", []):
            vec = self.embedder.embed(e["content"])
            self.store.conn.execute(
                "INSERT INTO events(ts, content, tags, importance, embedding, archived, emotion, arousal) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (e["ts"], e["content"], e.get("tags", "[]"), e.get("importance", 0.5),
                 self.store.dump_vec(vec), e.get("archived", 0), e.get("emotion", ""), e.get("arousal", 0.0)),
            )
            counts["events"] += 1
        for em in data.get("emotions", []):
            self.store.conn.execute(
                "INSERT INTO emotions(ts, emotion, intensity, cause) VALUES (?,?,?,?)",
                (em["ts"], em["emotion"], em["intensity"], em.get("cause", "")),
            )
        for oc in data.get("outcomes", []):
            self.store.conn.execute(
                "INSERT INTO outcomes(ts, task, success) VALUES (?,?,?)",
                (oc["ts"], oc["task"], oc["success"]),
            )
        for me in data.get("meta", []):
            self.store.conn.execute(
                "INSERT INTO meta(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (me["key"], me["value"]),
            )
        for f in data.get("facts", []):
            vec = self.embedder.embed(f"{f['subject']} {f['predicate']} {f['object']}")
            self.store.conn.execute(
                "INSERT INTO facts(subject, predicate, object, valid_from, valid_to, source, embedding, archived) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f["subject"], f["predicate"], f["object"], f["valid_from"], f.get("valid_to"),
                 f.get("source", "import"), self.store.dump_vec(vec), f.get("archived", 0)),
            )
            counts["facts"] += 1
        for s in data.get("skills", []):
            doc = SkillDoc.from_markdown(s["markdown"])
            self.save_skill(doc)
            self.store.conn.execute(
                "UPDATE skills SET uses = ?, successes = ? WHERE name = ?",
                (s.get("uses", 0), s.get("successes", 0), doc.name),
            )
            counts["skills"] += 1
        self.store.conn.commit()
        self.store.audit("import", "all", json.dumps(counts))
        return counts

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        g = lambda q: self.store.conn.execute(q).fetchone()[0]
        return {
            "events": g("SELECT COUNT(*) FROM events WHERE archived = 0"),
            "events_archived": g("SELECT COUNT(*) FROM events WHERE archived = 1"),
            "facts_current": g("SELECT COUNT(*) FROM facts WHERE valid_to IS NULL AND archived = 0"),
            "facts_superseded": g("SELECT COUNT(*) FROM facts WHERE valid_to IS NOT NULL"),
            "skills": g("SELECT COUNT(*) FROM skills"),
            "audit_entries": g("SELECT COUNT(*) FROM audit"),
        }

    def close(self) -> None:
        self.store.close()
