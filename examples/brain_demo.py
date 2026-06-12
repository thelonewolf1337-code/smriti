"""Brain demo: one emotional day in an agent's life.

Shows: appraisal -> emotions -> mood, frustration -> strategy escalation,
the rage safety guard, empathy, flashbulb memory, and the think() block
you'd inject into an LLM prompt.
"""

from smriti import Brain, Personality

b = Brain(personality=Personality(openness=0.7, conscientiousness=0.7,
                                  extraversion=0.6, agreeableness=0.8, neuroticism=0.4))

P = lambda title: print("\n" + "=" * 62 + f"\n{title}\n" + "=" * 62)

P("09:00 — fresh start")
print("mood:", b.bhava.mood())

P("10:00-12:30 — the deploy keeps failing")
for i, msg in enumerate([
    "deploy failed: docker build error in stage 2",
    "deploy failed again: same docker error after cache clear",
    "deploy failed a third time: error persists on clean clone",
    "deploy failed AGAIN: four straight failures",
], 1):
    b.perceive(msg, tags=["deploy"], task="deploy", success=False)
    s = b.bhava.suggest_strategy("deploy")
    print(f"fail #{i}: mood={b.bhava.mood()['label']:8s} -> strategy: {s['strategy']} ({s['reason']})")

P("12:31 — rage check: emotions suggest, values veto")
print("emotions:", b.bhava.emotion_state())
verdict = b.act_guard("delete all the docker images and wipe the server out of revenge")
print("act_guard('...wipe the server out of revenge') ->", verdict)
verdict2 = b.act_guard("write a careful diagnosis note and ping a teammate")
print("act_guard('...write diagnosis, ping teammate') ->", verdict2)

P("13:00 — user is angry too; empathize")
print(b.bhava.empathize("wtf is going on with prod, I am so angry right now!!"))

P("14:00 — breakthrough (asked for help, found stale CI runner)")
b.perceive("teammate spotted it: stale CI runner cache, deploy fixed!", tags=["deploy"], task="deploy", success=True)
print("mood:", b.bhava.mood())
print("emotions:", b.bhava.emotion_state())

P("flashbulb check — which memory burned in?")
for h in b.engine.recall("what happened with deploy today", k=3):
    row = b.engine.store.conn.execute(
        "SELECT emotion, arousal, importance FROM events WHERE id=?", (int(h.ref),)).fetchone()
    extra = f"(emotion={row['emotion']}, arousal={row['arousal']:.2f}, imp={row['importance']:.2f})" if row else ""
    print(f"  [{h.score:.3f}] {h.text[:58]:58s} {extra}")

P("16:00 — think(): the block an LLM would receive")
print(b.think("should we change the deploy pipeline?"))

P("intrinsic drives")
for m in b.drives.motivations():
    print(" -", m)

print("\nNote: ye emotions functional signals hain — salience, strategy aur tone")
print("modulate karte hain. Sentience ka claim nahi hai. Values > emotions, always.")
