"""Bhava (भाव) — the affect system: simulated emotions, mood, personality.

Design positions:
  * Emotions here are FUNCTIONAL SIGNALS, not claims of sentience. They
    modulate attention, memory salience, and strategy — the useful part
    of what emotions do in humans (appraisal theory / PAD model).
  * Mood = personality baseline + decayed sum of recent emotion episodes
    in PAD space (Pleasure/valence, Arousal, Dominance).
  * SAFETY INVARIANT: negative emotions NEVER unlock destructive behavior.
    High anger ("rage") triggers cooldown + help-seeking, by construction.
    This is enforced again by SelfModel.check_action in manas.py.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from smriti.db import Store

DAY = 86400.0
HOUR = 3600.0

# Emotion -> (valence, arousal, dominance), roughly per affective-computing literature.
PAD = {
    "joy":          (0.8, 0.5, 0.4),
    "contentment":  (0.6, -0.3, 0.2),
    "pride":        (0.7, 0.3, 0.6),
    "gratitude":    (0.6, 0.2, -0.1),
    "curiosity":    (0.4, 0.5, 0.1),
    "surprise":     (0.1, 0.8, -0.1),
    "fear":         (-0.6, 0.6, -0.6),
    "anger":        (-0.5, 0.7, 0.3),
    "frustration":  (-0.4, 0.5, -0.2),
    "sadness":      (-0.7, -0.4, -0.4),
}

# How long each emotion lingers (half-life, hours). Anger cools slower than surprise.
HALF_LIFE_H = {
    "joy": 4, "contentment": 6, "pride": 6, "gratitude": 8, "curiosity": 3,
    "surprise": 0.5, "fear": 6, "anger": 8, "frustration": 5, "sadness": 12,
}

_MOOD_LABELS = {  # PAD sign octants -> human-readable mood word
    (1, 1, 1): "exuberant", (1, 1, -1): "excited", (1, -1, 1): "relaxed", (1, -1, -1): "calm",
    (-1, 1, 1): "hostile", (-1, 1, -1): "anxious", (-1, -1, 1): "brooding", (-1, -1, -1): "gloomy",
}

_USER_EMOTION_KEYWORDS = {
    "anger": ["angry", "furious", "hate", "wtf", "annoyed", "pissed", "bakwas", "gussa", "ridiculous"],
    "sadness": ["sad", "depressed", "cry", "crying", "lonely", "heartbroken", "udaas", "hopeless"],
    "joy": ["happy", "awesome", "great news", "khush", "excited", "love it", "amazing", "yay"],
    "fear": ["worried", "scared", "anxious", "afraid", "tension", "dar", "nervous", "panicking"],
}


@dataclass
class Personality:
    """Big Five traits, 0..1. Shapes baseline mood and emotional reactivity."""
    openness: float = 0.6
    conscientiousness: float = 0.6
    extraversion: float = 0.5
    agreeableness: float = 0.6
    neuroticism: float = 0.3

    def baseline_pad(self) -> tuple[float, float, float]:
        v = 0.35 * self.extraversion + 0.25 * self.agreeableness - 0.45 * self.neuroticism
        a = 0.30 * self.neuroticism + 0.25 * self.openness - 0.15 * self.conscientiousness
        d = 0.30 * self.extraversion + 0.25 * self.conscientiousness - 0.30 * self.neuroticism
        clamp = lambda x: max(-1.0, min(1.0, x))
        return clamp(v), clamp(a), clamp(d)

    def reactivity(self, emotion: str) -> float:
        """Neurotic personalities feel negative emotions harder; open ones feel more curiosity."""
        if emotion in ("anger", "frustration", "fear", "sadness"):
            return 0.7 + 0.6 * self.neuroticism
        if emotion == "curiosity":
            return 0.6 + 0.8 * self.openness
        return 1.0


class Bhava:
    def __init__(self, store: Store, personality: Personality | None = None):
        self.store = store
        self.personality = personality or Personality()

    # ------------------------------------------------------------------ #
    # feeling + decay                                                     #
    # ------------------------------------------------------------------ #
    def feel(self, emotion: str, intensity: float, cause: str = "", ts: float | None = None) -> None:
        if emotion not in PAD:
            raise ValueError(f"unknown emotion '{emotion}' (known: {sorted(PAD)})")
        intensity = max(0.0, min(1.0, intensity * self.personality.reactivity(emotion)))
        if intensity < 0.01:
            return
        self.store.conn.execute(
            "INSERT INTO emotions(ts, emotion, intensity, cause) VALUES (?,?,?,?)",
            (ts or self.store.now(), emotion, intensity, cause[:120]),
        )
        self.store.conn.commit()
        self.store.audit("feel", f"emotion:{emotion}", f"{intensity:.2f} {cause[:60]}")

    def emotion_state(self, ts: float | None = None) -> dict[str, float]:
        """Current intensity per emotion = decayed sum of episodes (last 7 days)."""
        now = ts or self.store.now()
        state: dict[str, float] = {}
        rows = self.store.conn.execute(
            "SELECT emotion, intensity, ts FROM emotions WHERE ts > ?", (now - 7 * DAY,)
        ).fetchall()
        for r in rows:
            hl = HALF_LIFE_H.get(r["emotion"], 4) * HOUR
            decayed = r["intensity"] * 0.5 ** (max(0.0, now - r["ts"]) / hl)
            state[r["emotion"]] = min(1.0, state.get(r["emotion"], 0.0) + decayed)
        return {k: round(v, 3) for k, v in state.items() if v >= 0.02}

    def dominant_emotion(self, ts: float | None = None) -> tuple[str, float]:
        state = self.emotion_state(ts)
        if not state:
            return "", 0.0
        name = max(state, key=state.get)
        return name, state[name]

    # ------------------------------------------------------------------ #
    # mood (PAD)                                                          #
    # ------------------------------------------------------------------ #
    def mood(self, ts: float | None = None) -> dict:
        bv, ba, bd = self.personality.baseline_pad()
        v, a, d = bv, ba, bd
        for emo, inten in self.emotion_state(ts).items():
            ev, ea, ed = PAD[emo]
            v += 0.6 * inten * ev
            a += 0.6 * inten * ea
            d += 0.6 * inten * ed
        clamp = lambda x: max(-1.0, min(1.0, x))
        v, a, d = clamp(v), clamp(a), clamp(d)
        sign = lambda x: 1 if x >= 0 else -1
        label = _MOOD_LABELS[(sign(v), sign(a), sign(d))]
        if abs(v) < 0.12 and abs(a) < 0.12:
            label = "neutral"
        return {"valence": round(v, 3), "arousal": round(a, 3), "dominance": round(d, 3), "label": label}

    # ------------------------------------------------------------------ #
    # appraisal: events -> emotions                                       #
    # ------------------------------------------------------------------ #
    def appraise_event(self, content: str, novelty: float = 0.0, ts: float | None = None) -> None:
        """Lightweight OCC-style appraisal of an incoming event."""
        if novelty > 0.45:
            self.feel("curiosity", 0.3 + 0.5 * novelty, cause=f"novel: {content[:50]}", ts=ts)
        if novelty > 0.8:
            self.feel("surprise", 0.5, cause=content[:50], ts=ts)

    def record_outcome(self, task: str, success: bool, ts: float | None = None) -> int:
        """Task outcomes drive the strongest emotions (goal-based appraisal)."""
        now = ts or self.store.now()
        self.store.conn.execute(
            "INSERT INTO outcomes(ts, task, success) VALUES (?,?,?)", (now, task, int(success))
        )
        self.store.conn.commit()
        streak = self.failure_streak(task)
        if success:
            prior_fails = self._fails_before_last_success(task)
            if prior_fails >= 2:
                self.feel("pride", 0.6, cause=f"cracked '{task}' after {prior_fails} failures", ts=now)
                self.feel("contentment", 0.4, cause=f"relief on '{task}'", ts=now)
            else:
                self.feel("joy", 0.4, cause=f"'{task}' succeeded", ts=now)
        else:
            self.feel("frustration", min(1.0, 0.25 * streak), cause=f"'{task}' failed x{streak}", ts=now)
            if streak >= 3:
                self.feel("anger", min(1.0, 0.3 * (streak - 2)), cause=f"'{task}' keeps failing", ts=now)
        return streak

    def failure_streak(self, task: str) -> int:
        rows = self.store.conn.execute(
            "SELECT success FROM outcomes WHERE task = ? ORDER BY ts DESC, id DESC", (task,)
        ).fetchall()
        streak = 0
        for r in rows:
            if r["success"]:
                break
            streak += 1
        return streak

    def _fails_before_last_success(self, task: str) -> int:
        rows = self.store.conn.execute(
            "SELECT success FROM outcomes WHERE task = ? ORDER BY ts DESC, id DESC", (task,)
        ).fetchall()
        if not rows or not rows[0]["success"]:
            return 0
        fails = 0
        for r in rows[1:]:
            if r["success"]:
                break
            fails += 1
        return fails

    # ------------------------------------------------------------------ #
    # emotion -> strategy (the useful part of "rage")                     #
    # ------------------------------------------------------------------ #
    def suggest_strategy(self, task: str) -> dict:
        """Frustration escalates the STRATEGY, never the aggression.
        SAFETY: anger maps to cooldown + help-seeking — by design there is
        no path from any emotion to a destructive action."""
        streak = self.failure_streak(task)
        anger = self.emotion_state().get("anger", 0.0)
        if anger > 0.7:
            strategy, why = "defer-and-cooldown", "anger is high; stepping back beats lashing out"
        elif streak >= 4:
            strategy, why = "defer-and-cooldown", f"{streak} straight failures; revisit with fresh context"
        elif streak == 3:
            strategy, why = "ask-for-help", "three failures = blind spot; get a human or another agent"
        elif streak == 2:
            strategy, why = "switch-approach", "same approach failed twice; change the method, not the effort"
        else:
            strategy, why = "retry-with-diagnosis", "diagnose the failure, fix, retry once"
        return {"task": task, "failure_streak": streak, "anger": round(anger, 3),
                "strategy": strategy, "reason": why}

    # ------------------------------------------------------------------ #
    # empathy: read the user, mirror gently (emotional contagion)         #
    # ------------------------------------------------------------------ #
    def read_user_emotion(self, text: str) -> tuple[str, float]:
        t = text.lower()
        best, hits_best = "", 0
        for emo, kws in _USER_EMOTION_KEYWORDS.items():
            hits = sum(1 for kw in kws if kw in t)
            if hits > hits_best:
                best, hits_best = emo, hits
        if not best and re.search(r"!{2,}|\b(now|immediately)\b.*\?", t):
            best, hits_best = "anger", 1
        conf = min(1.0, 0.4 + 0.25 * hits_best) if best else 0.0
        return best, round(conf, 2)

    def empathize(self, user_text: str) -> dict:
        emo, conf = self.read_user_emotion(user_text)
        if emo and conf > 0:
            mirror = "sadness" if emo == "sadness" else ("frustration" if emo == "anger" else emo)
            self.feel(mirror, conf * 0.3 * self.personality.agreeableness,
                      cause="emotional contagion from user")
        return {"user_emotion": emo or "neutral", "confidence": conf,
                "respond_with": {
                    "anger": "acknowledge first, solve second, zero defensiveness",
                    "sadness": "warmth and patience before any task talk",
                    "fear": "reassure with concrete facts and a clear plan",
                    "joy": "match the energy, celebrate, then build on it",
                }.get(emo, "normal helpful tone")}
