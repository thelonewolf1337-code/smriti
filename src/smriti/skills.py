"""Procedural memory: skill documents (Hermes-style).

A skill is a markdown doc with a fixed structure: purpose, triggers,
steps, failure modes, verification. Progressive disclosure: agents read
the summary first and expand to the full doc only when needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SkillDoc:
    name: str
    purpose: str
    triggers: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)

    def summary(self) -> str:
        trig = ", ".join(self.triggers) or "-"
        return f"[skill] {self.name}: {self.purpose} (triggers: {trig})"

    def to_markdown(self) -> str:
        def sec(title: str, items: list[str]) -> str:
            body = "\n".join(f"- {i}" for i in items) or "- (none recorded)"
            return f"## {title}\n{body}"

        return "\n\n".join(
            [
                f"# Skill: {self.name}",
                f"**Purpose:** {self.purpose}",
                sec("Triggers", self.triggers),
                sec("Steps", self.steps),
                sec("Failure modes", self.failure_modes),
                sec("Verification", self.verification),
            ]
        )

    @classmethod
    def from_markdown(cls, md: str) -> "SkillDoc":
        name_m = re.search(r"^# Skill:\s*(.+)$", md, re.M)
        purpose_m = re.search(r"\*\*Purpose:\*\*\s*(.+)$", md, re.M)

        def items(title: str) -> list[str]:
            m = re.search(rf"## {title}\n(.*?)(?=\n## |\Z)", md, re.S)
            if not m:
                return []
            out = [ln[2:].strip() for ln in m.group(1).strip().splitlines() if ln.startswith("- ")]
            return [i for i in out if i and i != "(none recorded)"]

        return cls(
            name=(name_m.group(1).strip() if name_m else "unnamed"),
            purpose=(purpose_m.group(1).strip() if purpose_m else ""),
            triggers=items("Triggers"),
            steps=items("Steps"),
            failure_modes=items("Failure modes"),
            verification=items("Verification"),
        )
