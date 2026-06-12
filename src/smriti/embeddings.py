"""Pluggable embeddings.

Default: HashEmbedder — deterministic, offline, zero dependencies.
Good enough for hybrid retrieval in an MVP; swap in a real model
(sentence-transformers, OpenAI-compatible endpoint) via the same interface.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """Character n-gram hashing embedder. Deterministic and offline.

    Not semantically deep, but captures lexical similarity well enough to
    make hybrid retrieval (vector + keyword + recency + importance) useful
    without any API key or model download.
    """

    def __init__(self, dim: int = 256, ngram: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram = ngram

    def _ngrams(self, text: str):
        toks = tokenize(text)
        for tok in toks:
            yield tok  # whole word
            padded = f"_{tok}_"
            for n in range(self.ngram[0], self.ngram[1] + 1):
                for i in range(len(padded) - n + 1):
                    yield padded[i : i + n]

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for g in self._ngrams(text):
            h = int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OllamaEmbedder:
    """Real semantic embeddings via a local Ollama server (no API key, private).

    Usage:
        eng = MemoryEngine(path, embedder=OllamaEmbedder())  # needs `ollama pull nomic-embed-text`

    Falls back loudly (raises) if the server is unreachable — pass
    fallback=HashEmbedder() to degrade gracefully instead.
    """

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434",
                 dim: int = 768, fallback: "Embedder | None" = None, timeout: float = 10.0):
        self.model, self.host, self.dim, self.fallback, self.timeout = model, host, dim, fallback, timeout

    def embed(self, text: str) -> list[float]:
        import json as _json
        import urllib.request

        try:
            req = urllib.request.Request(
                f"{self.host}/api/embeddings",
                data=_json.dumps({"model": self.model, "prompt": text}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                vec = _json.loads(resp.read())["embedding"]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            self.dim = len(vec)
            return [v / norm for v in vec]
        except Exception:
            if self.fallback is not None:
                return self.fallback.embed(text)
            raise


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))  # inputs are L2-normalized


def keyword_overlap(query: str, text: str) -> float:
    """BM25-lite: fraction of query tokens present in text."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    return len(q & t) / len(q)
