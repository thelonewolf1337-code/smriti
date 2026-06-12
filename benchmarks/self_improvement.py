"""Self-improvement benchmark: does memory actually make the agent better?

Protocol (offline, deterministic, zero deps):
  DAY 1  — the agent solves N multi-step ops tasks by trial and error.
           Every step is recorded as an episodic event (cost = steps taken).
  NIGHT  — consolidation runs: experiences become skill documents.
  DAY 30 — the agent faces the same tasks again. For each task it queries
           memory; if the learned skill is the top recall hit, the cost is
           1 step (read the skill) instead of redoing the trial-and-error.

Metrics:
  skill_hit_rate   — fraction of tasks where recall ranks the right skill #1
  steps_day1 / steps_day30 — total step cost before/after learning
  reduction_pct    — % fewer steps on day 30

This is the claim most memory systems never measure. Run: python benchmarks/self_improvement.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smriti import Consolidator, MemoryEngine  # noqa: E402

DAY = 86400.0

TASKS = {
    "deploy-api": [
        "ssh into the vps and git pull the api repo",
        "docker compose build then docker compose up -d for the api",
        "copy .env.example to .env before the api container starts",
        "verify api deploy with curl localhost:8000/health",
    ],
    "rotate-secrets": [
        "generate a new jwt secret with openssl rand for auth",
        "update the jwt secret in the vault and the .env for auth",
        "restart the auth service so the new jwt secret loads",
        "verify old jwt tokens for auth are rejected after rotation",
    ],
    "backup-database": [
        "pg_dump the production database to a timestamped backup file",
        "compress the database backup file with gzip",
        "upload the database backup to the offsite bucket",
        "verify the database backup restores into a scratch instance",
    ],
    "fix-ssl-renewal": [
        "check certbot logs for the ssl renewal failure reason",
        "open port 80 in the firewall so the ssl http challenge passes",
        "run certbot renew for the ssl certificate manually",
        "verify the ssl certificate expiry date with openssl s_client",
    ],
    "onboard-developer": [
        "create the developer account and add it to the github org",
        "grant the developer access to staging but not production",
        "send the developer the onboarding checklist and dev setup doc",
        "verify the developer can run the test suite locally",
    ],
}


def run_benchmark(verbose: bool = True) -> dict:
    eng = MemoryEngine(":memory:")
    day1 = time.time() - 29 * DAY

    # -- DAY 1: trial and error ------------------------------------------- #
    steps_day1 = 0
    for task, steps in TASKS.items():
        for i, step in enumerate(steps):
            eng.remember_event(step, tags=[task], ts=day1 + steps_day1 * 60 + i * 60)
            steps_day1 += 1

    # -- NIGHT: consolidation --------------------------------------------- #
    report = Consolidator(eng).run(window_days=30, min_skill_events=3)

    # -- DAY 30: same tasks, with memory ---------------------------------- #
    steps_day30, hits = 0, 0
    results = []
    for task, steps in TASKS.items():
        query = f"how do I {task.replace('-', ' ')}"
        top = eng.recall(query, k=3)
        skill_first = bool(top) and top[0].kind == "skill" and top[0].ref == f"handle-{task}"
        if skill_first:
            hits += 1
            cost = 1  # read the skill, execute
            eng.record_skill_use(f"handle-{task}", success=True)
        else:
            cost = len(steps)  # no usable skill -> redo trial and error
        steps_day30 += cost
        results.append((task, skill_first, cost))

    metrics = {
        "tasks": len(TASKS),
        "skills_learned": report["skills_added"],
        "skill_hit_rate": round(hits / len(TASKS), 2),
        "steps_day1": steps_day1,
        "steps_day30": steps_day30,
        "reduction_pct": round(100 * (1 - steps_day30 / steps_day1), 1),
    }

    if verbose:
        print("=" * 60)
        print("SMRITI SELF-IMPROVEMENT BENCHMARK")
        print("=" * 60)
        for task, hit, cost in results:
            print(f"  {task:20s} skill-recalled={'YES' if hit else 'NO ':3s} day30-cost={cost}")
        print("-" * 60)
        for k, v in metrics.items():
            print(f"  {k:16s} = {v}")
        print("=" * 60)

    eng.close()
    return metrics


if __name__ == "__main__":
    m = run_benchmark()
    # CI gate: learning must demonstrably work.
    assert m["skill_hit_rate"] >= 0.8, f"skill_hit_rate too low: {m['skill_hit_rate']}"
    assert m["reduction_pct"] >= 50, f"reduction too low: {m['reduction_pct']}%"
    print("PASS: memory made the agent measurably better.")
