"""Smriti MCP server — plug persistent memory into ANY agent.

Run:  smriti-mcp  (or: python -m smriti.server)
Env:  SMRITI_DB=/path/to/memory.db  (default: ~/.smriti/memory.db)

Add to an MCP client config (Claude, OpenClaw, Hermes, anything):
  { "mcpServers": { "smriti": { "command": "smriti-mcp" } } }
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from smriti.brain import Brain
from smriti.consolidate import Consolidator
from smriti.memory import MemoryEngine
from smriti.skills import SkillDoc

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover
    FastMCP = None


def build_server(db_path: str | None = None):
    if FastMCP is None:
        raise RuntimeError("Install the MCP extra first: pip install 'smriti-memory[mcp]'")

    path = db_path or os.environ.get("SMRITI_DB") or str(Path.home() / ".smriti" / "memory.db")
    brain = Brain(path)
    engine = brain.engine
    mcp = FastMCP("smriti", instructions=(
        "Persistent memory + simulated affect for this agent. Call `recall` BEFORE answering "
        "anything that might depend on past context, and `think` to get full cognitive context "
        "(mood, focus, memories, drives, values). Use `perceive` instead of remember_event when "
        "an event should also be felt (task outcomes especially). Call `save_fact` for durable "
        "facts. Run `consolidate` at session end. Emotions are functional signals, not feelings; "
        "values always override emotions."
    ))

    @mcp.tool()
    def remember_event(content: str, importance: float = 0.5, tags: list[str] | None = None) -> str:
        """Store an episodic event (something that happened). importance: 0..1."""
        eid = engine.remember_event(content, importance=importance, tags=tags)
        return f"stored event #{eid}"

    @mcp.tool()
    def save_fact(subject: str, predicate: str, object: str) -> str:
        """Store a durable fact triple. Newer facts supersede older ones for the same subject+predicate."""
        fid = engine.save_fact(subject, predicate, object, source="agent")
        return f"stored fact #{fid}"

    @mcp.tool()
    def recall(query: str, k: int = 5) -> str:
        """Hybrid search across events, facts, and skill summaries. Returns ranked JSON."""
        return json.dumps([h.as_dict() for h in engine.recall(query, k=k)], indent=2)

    @mcp.tool()
    def get_skill(name: str, full: bool = False) -> str:
        """Get a skill. Summary by default; full=true returns the complete procedure markdown."""
        return engine.get_skill(name, full=full) or f"no skill named '{name}'"

    @mcp.tool()
    def save_skill(name: str, purpose: str, triggers: list[str], steps: list[str],
                   failure_modes: list[str] | None = None, verification: list[str] | None = None) -> str:
        """Save/update a reusable skill document after completing a non-trivial procedure."""
        engine.save_skill(SkillDoc(name, purpose, triggers, steps, failure_modes or [], verification or []))
        return f"saved skill '{name}'"

    @mcp.tool()
    def record_skill_use(name: str, success: bool) -> str:
        """Record that a skill was used and whether it worked. Powers the self-improvement metric."""
        engine.record_skill_use(name, success)
        return "recorded"

    @mcp.tool()
    def consolidate(window_days: float = 1.0) -> str:
        """Reflection pass: turn recent events into facts + skills. Run at session end / from a cron."""
        return json.dumps(Consolidator(engine).run(window_days=window_days))

    @mcp.tool()
    def forget(kind: str, ref: str) -> str:
        """Hard-delete a memory (kind: event|fact|skill, ref: id or skill name). Audited."""
        return "deleted" if engine.forget(kind, ref) else "not found"

    @mcp.tool()
    def export_memory() -> str:
        """Export ALL memory as portable JSON (no embeddings — they regenerate on import)."""
        return json.dumps(engine.export_json())

    @mcp.tool()
    def memory_stats() -> str:
        """Counts of events/facts/skills/audit entries."""
        return json.dumps(engine.stats())

    # ---- brain tools (affect + cognition) ------------------------------- #
    @mcp.tool()
    def perceive(content: str, tags: list[str] | None = None, importance: float = 0.5,
                 task: str | None = None, success: bool | None = None) -> str:
        """Full intake: appraise -> feel -> remember (emotion-tagged) -> attend.
        Pass task+success for outcomes so frustration/pride dynamics work."""
        eid = brain.perceive(content, tags=tags, importance=importance, task=task, success=success)
        return f"perceived as event #{eid}; mood now: {brain.bhava.mood()['label']}"

    @mcp.tool()
    def think(query: str) -> str:
        """Inner-monologue context block (mood, focus, memories, drives, values).
        Inject this into your prompt before answering anything important."""
        return brain.think(query)

    @mcp.tool()
    def mood() -> str:
        """Current mood (PAD + label) and active emotions with intensities."""
        return json.dumps({"mood": brain.bhava.mood(), "emotions": brain.bhava.emotion_state()})

    @mcp.tool()
    def feel(emotion: str, intensity: float, cause: str = "") -> str:
        """Manually register an emotion episode (joy, anger, fear, sadness, curiosity, ...)."""
        brain.bhava.feel(emotion, intensity, cause)
        return json.dumps(brain.bhava.mood())

    @mcp.tool()
    def suggest_strategy(task: str) -> str:
        """Frustration-aware next move for a failing task: retry / switch / ask-for-help / cooldown."""
        return json.dumps(brain.bhava.suggest_strategy(task))

    @mcp.tool()
    def empathize(user_text: str) -> str:
        """Detect the user's emotional state and get guidance on response tone."""
        return json.dumps(brain.bhava.empathize(user_text))

    @mcp.tool()
    def motivations() -> str:
        """Intrinsic drives ranked by urgency: curiosity, competence gaps, consistency pressure."""
        return json.dumps(brain.drives.motivations())

    @mcp.tool()
    def act_guard(action: str) -> str:
        """Conscience check BEFORE risky actions. Values veto; high anger adds a cooldown advisory."""
        return json.dumps(brain.act_guard(action))

    @mcp.tool()
    def compress_memory(older_than_days: float = 30.0) -> str:
        """Capacity maintenance: fold old episodic detail into compact gist memories."""
        return json.dumps(engine.compress(older_than_days=older_than_days))

    @mcp.tool()
    def soul() -> str:
        """The agent's self-model: identity, values, goals."""
        return brain.soul.soul_markdown()

    return mcp


def main() -> None:  # pragma: no cover
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
