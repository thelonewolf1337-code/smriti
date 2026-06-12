"""Smriti — portable memory + self-improvement loop for AI agents.

Three memory layers:
  - episodic   : raw events (what happened)
  - semantic   : facts with temporal validity (what is true, and since when)
  - procedural : skill documents (how to do things), Hermes-style

Plus: hybrid retrieval, decay/forgetting, consolidation (reflection),
audit log, and full export/import for memory portability.
"""

from smriti.memory import MemoryEngine
from smriti.consolidate import Consolidator, ollama_llm
from smriti.skills import SkillDoc
from smriti.embeddings import HashEmbedder, OllamaEmbedder, cosine
from smriti.bhava import Bhava, Personality
from smriti.manas import Drives, SelfModel, WorkingMemory
from smriti.brain import Brain

__version__ = "0.3.0"
__all__ = [
    "MemoryEngine", "Consolidator", "SkillDoc", "HashEmbedder", "OllamaEmbedder",
    "cosine", "ollama_llm",
    "Bhava", "Personality", "Drives", "SelfModel", "WorkingMemory", "Brain",
]
