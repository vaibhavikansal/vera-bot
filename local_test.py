"""
Local end-to-end test (no LLM key needed).
  1. start the bot:   uvicorn bot:app --port 8080
  2. run:             python local_test.py
Pushes the full expanded dataset, ticks the 30 test pairs, and plays a few
reply scenarios. Prints every message so you can read them.
"""
import glob, json, os, subprocess, sys, time, urllib.request

BOT = os.getenv("BOT_URL", "http://localhost:8080")
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "expanded")
if not os.path.isdir(EXP):
    subprocess.run([sys.executable, os.path.join(HERE, "dataset", "generate_dataset.py"),
                    "--seed-dir", os.path.join(HERE, "dataset"), "--out", EXP], check=True, stdout=subprocess.DEVNULL)


def call(method, path, body=None):
    req = urllib.request.Request(BOT + path, method=method, headers={"Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t


def load(sub, key):
    return {json.load(open(f))[key]: json.load(open(f)) for f in glob.glob(f"{EXP}/{sub}/*.json")}


cats, ms, cs, ts = load("categories", "slug"), load("merchants", "merchant_id"), load("customers", "customer_id"), load("triggers", "id")
call("POST", "/v1/teardown", {})  # start clean
print("healthz:", call("GET", "/v1/healthz")[1])
for scope, d in (("category", cats), ("merchant", ms), ("customer", cs), ("trigger", ts)):
    for cid, p in d.items():
        s, r, _ = call("POST", "/v1/context", {"scope": scope, "context_id": cid, "version": 1, "payload": p, "delivered_at": "2026-04-26T10:00:00Z"})
        assert s in (200, 409), (s, r)
print("after push:", call("GET", "/v1/healthz")[1]["contexts_loaded"])
print("re-push same version ->", call("POST", "/v1/context", {"scope": "category", "context_id": "dentists", "version": 1, "payload": cats["dentists"], "delivered_at": "x"})[:2])

pairs = json.load(open(f"{EXP}/test_pairs.json"))["pairs"]
convs = {}
for i in range(0, len(pairs), 5):
    batch = [p["trigger_id"] for p in pairs[i:i + 5]]
    s, r, dt = call("POST", "/v1/tick", {"now": "2026-04-26T10:30:00Z", "available_triggers": batch})
    print(f"\n=== tick {i // 5 + 1}: {len(r['actions'])} actions in {dt:.1f}s ===")
    for a in r["actions"]:
        convs[a["trigger_id"]] = a
        print(f"[{a['trigger_id']}] ({a['send_as']}, cta={a['cta']})\n  {a['body']}\n  why: {a['rationale']}")

print("\n=== re-tick same triggers (should be empty: already sent) ===")
print(call("POST", "/v1/tick", {"now": "2026-04-26T10:35:00Z", "available_triggers": [p["trigger_id"] for p in pairs[:5]]})[1])

print("\n=== next tick: leftover triggers for merchants that were already messaged this tick ===")
for a in call("POST", "/v1/tick", {"now": "2026-04-26T10:40:00Z", "available_triggers": ["trg_022_cde_webinar_dentists", "trg_001_research_digest_dentists"]})[1]["actions"]:
    convs[a["trigger_id"]] = a
    print(f"[{a['trigger_id']}]\n  {a['body']}")


def chat(trg, msgs, role="merchant"):
    a = convs[trg]
    print(f"\n--- conversation {a['conversation_id']} ---\nBOT: {a['body']}")
    for n, m in enumerate(msgs, 2):
        s, r, dt = call("POST", "/v1/reply", {"conversation_id": a["conversation_id"], "merchant_id": a["merchant_id"],
                                              "customer_id": a["customer_id"], "from_role": role, "message": m,
                                              "received_at": "2026-04-26T10:40:00Z", "turn_number": n})
        print(f"{role.upper()}: {m}\nBOT [{r['action']}] ({dt:.1f}s): {r.get('body', '')}  // {r['rationale']}")
        if r["action"] == "end":
            break

chat("trg_001_research_digest_dentists",
     ["Interesting. Is this relevant for diabetic patients too?", "Btw can you also help me file GST?", "haan theek hai, bhej do"])
chat("trg_023_competitor_opened_dentist", ["Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly."] * 4)
chat("trg_004_perf_dip_bharat", ["You people are useless", "can you help with my loan?", "stop"])
chat("trg_003_recall_due_priya", ["2"], role="customer")
chat("trg_013_corporate_thali_planning", ["busy right now, later"])
chat("trg_021_unverified_gbp_sunrise", ["mujhe magicpin judna hai"])
