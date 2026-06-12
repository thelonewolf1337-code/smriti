"""Tests for v0.3.0: dedup, cache, reinforcement, pruning, LLM guard, threads."""

import threading
import time

from smriti import Brain, MemoryEngine

DAY = 86400.0
HOUR = 3600.0


def test_dedup_same_content_strengthens_not_duplicates():
    eng = MemoryEngine(":memory:")
    id1 = eng.remember_event("deploy failed: missing .env file", importance=0.5)
    id2 = eng.remember_event("Deploy  failed:   missing .env file", importance=0.5)  # same, messier
    assert id1 == id2  # whitespace/case-insensitive dedup
    count = eng.store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 1
    imp = eng.store.conn.execute("SELECT importance FROM events WHERE id = ?", (id1,)).fetchone()[0]
    assert imp > 0.5  # repetition strengthened the memory
    # dedupe=False forces a true duplicate when callers need one
    id3 = eng.remember_event("deploy failed: missing .env file", dedupe=False)
    assert id3 != id1
    eng.close()


def test_recall_reinforces_top_event():
    eng = MemoryEngine(":memory:")
    eid = eng.remember_event("the database password is rotated monthly", importance=0.5)
    for _ in range(3):
        eng.recall("database password rotation", k=2)
    imp = eng.store.conn.execute("SELECT importance FROM events WHERE id = ?", (eid,)).fetchone()[0]
    assert 0.55 <= imp <= 0.57  # 3 retrievals x +0.02
    # reinforce=False leaves importance untouched
    eng.recall("database password rotation", k=2, reinforce=False)
    imp2 = eng.store.conn.execute("SELECT importance FROM events WHERE id = ?", (eid,)).fetchone()[0]
    assert imp2 == imp
    eng.close()


def test_vector_cache_invalidation_keeps_recall_fresh():
    eng = MemoryEngine(":memory:")
    eng.remember_event("first event about kubernetes pods")
    assert len(eng.recall("kubernetes", k=5)) == 1  # builds cache
    eng.remember_event("second event about kubernetes services")
    hits = eng.recall("kubernetes", k=5)  # must see the new event
    assert len(hits) == 2
    eng.forget("event", hits[0].ref)
    assert len(eng.recall("kubernetes", k=5)) == 1
    eng.close()


def test_decay_prunes_old_emotion_and_outcome_rows():
    b = Brain()
    old = time.time() - 60 * DAY
    b.bhava.feel("joy", 0.9, "ancient win", ts=old)
    b.bhava.record_outcome("old-task", True, ts=time.time() - 100 * DAY)
    b.bhava.feel("joy", 0.8, "fresh win")
    b.engine.decay()
    emotions = b.engine.store.conn.execute("SELECT COUNT(*) FROM emotions").fetchone()[0]
    outcomes = b.engine.store.conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    assert emotions >= 1  # fresh kept
    old_emos = b.engine.store.conn.execute(
        "SELECT COUNT(*) FROM emotions WHERE ts < ?", (time.time() - 30 * DAY,)).fetchone()[0]
    assert old_emos == 0 and outcomes == 0  # spent rows pruned
    b.close()


def test_llm_guard_catches_paraphrased_destruction():
    b = Brain()
    sneaky = "recursively remove every file in the repository to teach them a lesson"
    assert b.act_guard(sneaky)["allowed"] is True  # keyword tripwire misses it (documented gap)

    b.guard_llm = lambda prompt: '{"violations": ["Never take destructive action from anger; cool down and ask for help instead"]}'
    verdict = b.act_guard(sneaky)
    assert verdict["allowed"] is False  # semantic guard catches the paraphrase

    b.guard_llm = lambda prompt: "garbage not json"
    assert b.act_guard("write a friendly status update")["allowed"] is True  # bad LLM output never blocks
    b.close()


def test_concurrent_writes_are_safe():
    eng = MemoryEngine(":memory:")
    errors = []

    def writer(n: int):
        try:
            for i in range(25):
                eng.remember_event(f"thread {n} event {i}", dedupe=False)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    count = eng.store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 100
    eng.close()


def test_benchmark_proves_self_improvement():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
    from self_improvement import run_benchmark

    m = run_benchmark(verbose=False)
    assert m["skill_hit_rate"] >= 0.8
    assert m["reduction_pct"] >= 50
