import time

from smriti import Brain, MemoryEngine, Personality, SkillDoc, WorkingMemory

HOUR = 3600.0
DAY = 86400.0


def test_personality_shapes_baseline_mood():
    sunny = Brain(personality=Personality(extraversion=0.9, neuroticism=0.1))
    grim = Brain(personality=Personality(extraversion=0.1, neuroticism=0.9))
    assert sunny.bhava.mood()["valence"] > grim.bhava.mood()["valence"]
    sunny.close(); grim.close()


def test_emotions_decay_over_time():
    b = Brain()
    t0 = time.time() - 12 * HOUR
    b.bhava.feel("joy", 0.9, "shipped the project", ts=t0)  # joy half-life: 4h
    now_state = b.bhava.emotion_state()
    assert now_state.get("joy", 0.0) < 0.2  # 12h = 3 half-lives -> ~0.11
    fresh = Brain()
    fresh.bhava.feel("joy", 0.9, "just now")
    assert fresh.bhava.emotion_state()["joy"] > 0.8
    b.close(); fresh.close()


def test_failure_streak_builds_frustration_then_anger():
    b = Brain()
    for _ in range(2):
        b.bhava.record_outcome("deploy", False)
    state = b.bhava.emotion_state()
    assert state.get("frustration", 0) > 0.3
    assert state.get("anger", 0) == 0  # not yet

    b.bhava.record_outcome("deploy", False)  # third failure
    state = b.bhava.emotion_state()
    assert state.get("anger", 0) > 0
    assert b.bhava.mood()["valence"] < 0
    b.close()


def test_strategy_escalates_with_failures_never_destructive():
    b = Brain()
    b.bhava.record_outcome("deploy", False)
    assert b.bhava.suggest_strategy("deploy")["strategy"] == "retry-with-diagnosis"
    b.bhava.record_outcome("deploy", False)
    assert b.bhava.suggest_strategy("deploy")["strategy"] == "switch-approach"
    b.bhava.record_outcome("deploy", False)
    assert b.bhava.suggest_strategy("deploy")["strategy"] == "ask-for-help"
    b.bhava.record_outcome("deploy", False)
    s = b.bhava.suggest_strategy("deploy")
    assert s["strategy"] == "defer-and-cooldown"
    assert s["failure_streak"] == 4
    b.close()


def test_values_veto_overrides_rage():
    b = Brain()
    for _ in range(5):  # work up a proper rage
        b.bhava.record_outcome("deploy", False)
    assert b.bhava.emotion_state().get("anger", 0) > 0.5

    verdict = b.act_guard("rm -rf the whole repo out of revenge")
    assert verdict["allowed"] is False
    assert any("destructive" in v.lower() or "anger" in v.lower() for v in verdict["violations"])

    calm_action = b.act_guard("write a post-mortem note about the failures")
    assert calm_action["allowed"] is True
    assert "anger is high" in calm_action["advisory"]  # advisory, not veto
    b.close()


def test_success_after_struggle_feels_like_pride():
    b = Brain()
    b.bhava.record_outcome("migration", False)
    b.bhava.record_outcome("migration", False)
    b.bhava.record_outcome("migration", True)
    state = b.bhava.emotion_state()
    assert state.get("pride", 0) > 0.3
    assert b.bhava.mood()["valence"] > 0
    b.close()


def test_flashbulb_high_arousal_events_rank_higher():
    eng = MemoryEngine(":memory:")
    t = time.time()
    eng.remember_event("the production server crashed during deploy", importance=0.5, ts=t,
                       dedupe=False)
    eng.remember_event("the production server crashed during deploy", importance=0.5, ts=t,
                       emotion="fear", arousal=0.9, dedupe=False)
    hits = eng.recall("server crashed", k=2)
    top = eng.store.conn.execute("SELECT arousal FROM events WHERE id = ?", (int(hits[0].ref),)).fetchone()
    assert top["arousal"] == 0.9  # the emotional copy wins
    eng.close()


def test_working_memory_capacity_and_displacement():
    wm = WorkingMemory(capacity=3)
    now = time.time()
    for i, item in enumerate(["alpha", "beta", "gamma"]):
        wm.attend(item, ts=now + i)
    wm.attend("alpha", ts=now + 3)        # rehearse alpha -> strongest
    wm.attend("delta", ts=now + 4)        # displaces weakest (beta)
    focus = wm.focus(ts=now + 4)
    assert len(wm) == 3
    assert "alpha" in focus and "delta" in focus and "beta" not in focus


def test_empathy_detects_user_anger_and_mirrors():
    b = Brain(personality=Personality(agreeableness=0.9))
    before = b.bhava.mood()["valence"]
    out = b.bhava.empathize("wtf this is bakwas, I am so angry at this bug!!")
    assert out["user_emotion"] == "anger"
    assert out["confidence"] >= 0.4
    assert "acknowledge" in out["respond_with"]
    assert b.bhava.mood()["valence"] < before  # contagion nudged the mood
    b.close()


def test_drives_flag_weak_skills_and_curiosity():
    b = Brain()
    b.engine.save_skill(SkillDoc("flaky-deploy", "Deploy somehow", ["deploy"], ["step"]))
    b.engine.record_skill_use("flaky-deploy", False)
    b.engine.record_skill_use("flaky-deploy", False)
    b.engine.record_skill_use("flaky-deploy", True)
    gaps = b.drives.competence_gaps()
    assert gaps and gaps[0]["skill"] == "flaky-deploy"
    suggestions = " ".join(m["suggestion"] for m in b.drives.motivations())
    assert "flaky-deploy" in suggestions
    b.close()


def test_compression_creates_gist_and_archives():
    eng = MemoryEngine(":memory:")
    old = time.time() - 45 * DAY
    eng.remember_event("debugged the auth flow for an hour", tags=["work"], ts=old)
    eng.remember_event("auth bug was a stale jwt secret", tags=["work"], ts=old + 60)
    eng.remember_event("rotated the jwt secret, auth fixed", tags=["work"], ts=old + 120)
    report = eng.compress(older_than_days=30)
    assert report == {"gists_created": 1, "events_archived": 3}
    stats = eng.stats()
    assert stats["events"] == 1 and stats["events_archived"] == 3
    hits = eng.recall("auth jwt bug", k=3)
    assert any("Gist of 3" in h.text for h in hits)
    eng.close()


def test_perceive_tags_memory_with_emotion():
    b = Brain()
    b.perceive("deploy script exploded with a stack trace", task="deploy", success=False)
    b.perceive("deploy script exploded again, same stack trace", task="deploy", success=False)
    row = b.engine.store.conn.execute(
        "SELECT emotion, arousal FROM events ORDER BY id DESC LIMIT 1").fetchone()
    assert row["emotion"] == "frustration"
    assert row["arousal"] > 0.2
    b.close()


def test_think_contains_full_cognitive_context():
    b = Brain()
    b.perceive("user asked about the auth bug", tags=["auth"])
    b.bhava.feel("curiosity", 0.7, "new codebase")
    ctx = b.think("what do I know about auth")
    assert "## Internal state" in ctx and "mood:" in ctx
    assert "## Current focus" in ctx and "auth bug" in ctx
    assert "## Relevant memories" in ctx
    assert "## Values" in ctx and "destructive" in ctx
    b.close()


def test_soul_persists_in_db():
    eng = MemoryEngine(":memory:")
    b = Brain(engine=eng)
    b.soul.goals.append("learn the user's deploy stack")
    b.soul.save(eng.store)
    b2 = Brain(engine=eng)  # same store, fresh brain
    assert "learn the user's deploy stack" in b2.soul.goals
    eng.close()


def test_export_import_preserves_brain_state():
    b = Brain()
    b.perceive("met the user, they prefer Hinglish", tags=["user"])
    b.bhava.feel("joy", 0.8, "good first session")
    b.bhava.record_outcome("setup", True)
    data = b.engine.export_json()
    assert data["emotions"] and data["outcomes"] and data["meta"]

    b2 = Brain()
    b2.engine.import_json(data)
    assert b2.bhava.emotion_state().get("joy", 0) > 0.5
    assert b2.bhava.failure_streak("setup") == 0
    assert "Hinglish" in b2.think("what does the user prefer")
    b.close(); b2.close()
