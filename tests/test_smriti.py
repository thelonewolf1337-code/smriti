import time

import pytest

from smriti import Consolidator, MemoryEngine, SkillDoc

DAY = 86400.0


@pytest.fixture()
def eng():
    e = MemoryEngine(":memory:")
    yield e
    e.close()


def test_event_roundtrip_and_recall(eng):
    eng.remember_event("Deployed the API with docker compose on the VPS", tags=["deploy"])
    eng.remember_event("Cooked pasta for dinner, used too much salt", tags=["life"])
    hits = eng.recall("how do I deploy with docker", k=2)
    assert hits[0].kind == "event"
    assert "docker" in hits[0].text.lower()


def test_fact_supersede_temporal(eng):
    t0 = time.time() - 30 * DAY
    eng.save_fact("ankesh", "lives in", "London", ts=t0)
    eng.save_fact("ankesh", "lives in", "Tokyo")

    cur = eng.current_fact("ankesh", "lives in")
    assert cur["object"] == "Tokyo"

    hist = eng.fact_history("ankesh", "lives in")
    assert len(hist) == 2
    assert hist[0]["object"] == "London" and hist[0]["valid_to"] is not None
    assert hist[1]["valid_to"] is None

    # exact duplicate of current fact is a no-op
    fid = eng.save_fact("ankesh", "lives in", "tokyo")
    assert fid == cur["id"]
    assert len(eng.fact_history("ankesh", "lives in")) == 2


def test_skill_progressive_disclosure(eng):
    doc = SkillDoc(
        name="deploy-api",
        purpose="Deploy the API to the VPS",
        triggers=["deploy", "release"],
        steps=["git pull", "docker compose build", "docker compose up -d", "curl /health"],
        verification=["GET /health returns 200"],
    )
    eng.save_skill(doc)

    summary = eng.get_skill("deploy-api")
    full = eng.get_skill("deploy-api", full=True)
    assert summary.startswith("[skill] deploy-api")
    assert len(summary) < len(full)
    assert "docker compose up -d" in full and "docker compose up -d" not in summary

    parsed = SkillDoc.from_markdown(full)
    assert parsed.name == "deploy-api"
    assert parsed.steps == doc.steps


def test_consolidation_heuristic_creates_fact_and_skill(eng):
    eng.remember_event("Ankesh lives in Mumbai. The project uses Postgres", tags=[])
    eng.remember_event("ssh into the vps and git pull", tags=["deploy"])
    eng.remember_event("docker compose build then up -d", tags=["deploy"])
    eng.remember_event("verify with curl localhost:8000/health", tags=["deploy"])

    report = Consolidator(eng).run(window_days=1)
    assert report["facts_added"] >= 2
    assert report["skills_added"] == 1
    assert eng.current_fact("ankesh", "lives in")["object"] == "Mumbai"
    assert eng.get_skill("handle-deploy") is not None
    assert "git pull" in eng.get_skill("handle-deploy", full=True)


def test_consolidation_llm_mode(eng):
    eng.remember_event("set up the staging box", tags=["ops"])
    fake_llm = lambda prompt: (
        '```json\n{"facts": [{"subject": "team", "predicate": "uses", "object": "staging box"}],'
        '"skills": [{"name": "setup-staging", "purpose": "Set up staging", "triggers": ["staging"],'
        '"steps": ["provision box", "install docker"]}]}\n```'
    )
    report = Consolidator(eng, llm=fake_llm).run()
    assert report == {"facts_added": 1, "skills_added": 1, "events_seen": 1}
    assert eng.current_fact("team", "uses")["object"] == "staging box"


def test_decay_archives_old_low_importance(eng):
    old = time.time() - 90 * DAY
    eng.remember_event("random old chatter", importance=0.3, ts=old)
    eng.remember_event("critical: prod db password rotated", importance=0.95, ts=old)
    eng.remember_event("fresh event", importance=0.3)

    # 0.3 * 0.5^(90/30) = 0.0375 < 0.05 -> archived
    # 0.95 * 0.5^(90/30) = 0.1188 > 0.05 -> survives (importance matters)
    archived = eng.decay(half_life_days=30, archive_below=0.05)
    assert archived == 1
    texts = [h.text for h in eng.recall("random old chatter", k=10)]
    assert "random old chatter" not in texts  # archived = excluded from recall
    assert any("prod db password" in t for t in texts)  # important memory survived


def test_skill_use_stats_boost_ranking(eng):
    eng.save_skill(SkillDoc("deploy-api", "Deploy the API", ["deploy"], ["step"]))
    eng.record_skill_use("deploy-api", success=True)
    eng.record_skill_use("deploy-api", success=True)
    s = eng.list_skills()[0]
    assert s["uses"] == 2 and s["successes"] == 2


def test_export_import_roundtrip(eng):
    eng.remember_event("event one", tags=["a"])
    eng.save_fact("ankesh", "lives in", "London", ts=time.time() - DAY)
    eng.save_fact("ankesh", "lives in", "Tokyo")
    eng.save_skill(SkillDoc("deploy-api", "Deploy", ["deploy"], ["git pull", "build"]))
    eng.record_skill_use("deploy-api", True)

    data = eng.export_json()
    assert "embedding" not in data["events"][0]

    eng2 = MemoryEngine(":memory:")
    counts = eng2.import_json(data)
    assert counts == {"events": 1, "facts": 2, "skills": 1}
    assert eng2.current_fact("ankesh", "lives in")["object"] == "Tokyo"
    assert eng2.list_skills()[0]["uses"] == 1
    assert eng2.recall("deploy", k=3)  # embeddings regenerated -> recall works
    eng2.close()


def test_forget_and_audit(eng):
    eid = eng.remember_event("sensitive thing")
    assert eng.forget("event", str(eid)) is True
    assert eng.forget("event", "9999") is False
    actions = [r["action"] for r in eng.store.conn.execute("SELECT action FROM audit").fetchall()]
    assert "forget" in actions and "write" in actions


def test_recall_ranks_current_fact_over_superseded(eng):
    eng.save_fact("ankesh", "lives in", "London", ts=time.time() - 60 * DAY)
    eng.save_fact("ankesh", "lives in", "Tokyo")
    hits = eng.recall("where does ankesh live", k=5, kinds=("fact",))
    assert len(hits) == 1  # superseded facts never surface in recall
    assert "Tokyo" in hits[0].text
