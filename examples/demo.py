"""Smriti end-to-end demo: 3 days in an agent's life.

Day 1: agent does a messy multi-step deploy, learns facts about the user.
Night: consolidation turns experience into facts + a skill.
Day 2: user moves city (temporal fact supersede).
Day 3: agent faces the same deploy task -> instant skill recall (self-improvement).
Finally: export -> brand new engine -> import (memory portability).
"""

import json
import time

from smriti import Consolidator, MemoryEngine

DAY = 86400.0
now = time.time()

eng = MemoryEngine(":memory:")
print("=" * 64)
print("DAY 1 — agent works, memory records")
print("=" * 64)

eng.remember_event("Ankesh lives in Mumbai. Ankesh prefers Hinglish replies", importance=0.8, ts=now - 2 * DAY)
eng.remember_event("The project uses FastAPI with Postgres", importance=0.7, ts=now - 2 * DAY)
eng.remember_event("ssh into the vps then git pull the main branch", tags=["deploy"], ts=now - 2 * DAY)
eng.remember_event("docker compose build, then docker compose up -d", tags=["deploy"], ts=now - 2 * DAY)
eng.remember_event("verify deploy with curl localhost:8000/health", tags=["deploy"], ts=now - 2 * DAY)
eng.remember_event("deploy failed once because .env was missing — copy .env.example first", tags=["deploy"], ts=now - 2 * DAY)
print("6 episodic events stored.")

print("\n>>> NIGHT — consolidation (reflection) runs from cron...")
report = Consolidator(eng).run(window_days=3)
print("reflection report:", report)

print("\n>>> What did the agent learn?")
for f in [("ankesh", "lives in"), ("ankesh", "prefers"), ("the project", "uses")]:
    fact = eng.current_fact(*f)
    print(f"  fact: {f[0]} / {f[1]} -> {fact['object'] if fact else '?'}")
print("  skill summary:", eng.get_skill("handle-deploy"))

print("\n" + "=" * 64)
print("DAY 2 — state change: Ankesh moves to Bangalore")
print("=" * 64)
eng.save_fact("ankesh", "lives in", "Bangalore")
cur = eng.current_fact("ankesh", "lives in")
hist = eng.fact_history("ankesh", "lives in")
print(f"current: {cur['object']}   (history: {[h['object'] for h in hist]})")
print("-> a naive vector store would still happily return 'Mumbai'. Smriti won't.")

print("\n" + "=" * 64)
print("DAY 3 — same task again: SELF-IMPROVEMENT CHECK")
print("=" * 64)
print('agent asks memory: recall("deploy the api to the vps")')
for h in eng.recall("deploy the api to the vps", k=3):
    print(f"  [{h.score:.3f}] ({h.kind}) {h.text[:80]}")

print("\nagent expands the top skill (progressive disclosure):")
print("-" * 40)
print(eng.get_skill("handle-deploy", full=True))
print("-" * 40)
eng.record_skill_use("handle-deploy", success=True)
print("Day 1: 4 events of trial-and-error. Day 3: one recall + one skill read. That's the loop.")

print("\n" + "=" * 64)
print("PORTABILITY — export -> new engine -> import")
print("=" * 64)
data = eng.export_json()
eng2 = MemoryEngine(":memory:")
print("imported into fresh engine:", eng2.import_json(data))
print("fresh engine knows:", eng2.current_fact("ankesh", "lives in")["object"])
print("fresh engine recalls skill:", eng2.recall("deploy", k=1)[0].text[:70])

print("\nfinal stats:", json.dumps(eng.stats()))
print("\nDemo complete — memory survives restarts, supersedes stale facts, and turns experience into skills.")
