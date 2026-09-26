"""
magicpin AI Challenge — Vera bot (HTTP server).

Run locally:   uvicorn bot:app --host 0.0.0.0 --port 8080
Endpoints:     GET /v1/healthz, GET /v1/metadata, POST /v1/context, POST /v1/tick,
               POST /v1/reply, POST /v1/teardown

Also exposes compose(category, merchant, trigger, customer) from composer.py
(challenge-brief §7.1).
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import llm
from composer import compose, compose_async  # noqa: F401  (compose re-exported for §7.1)
from conversation_handlers import new_state, respond

app = FastAPI(title="Vera bot")
START = time.time()


# Keep-alive: Render's free plan sleeps after 15 min without traffic (and wipes memory).
# Render sets RENDER_EXTERNAL_URL automatically, so the bot pings its own public URL
# every 10 minutes to stay awake. Does nothing when run locally.
async def _keep_alive():
    import httpx
    url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("KEEP_ALIVE_URL")
    if not url:
        return
    await asyncio.sleep(60)
    async with httpx.AsyncClient(timeout=20) as client:
        while True:
            try:
                await client.get(url.rstrip("/") + "/v1/healthz")
            except Exception as e:
                print(f"[keep-alive] ping failed: {e}")
            await asyncio.sleep(600)


@app.on_event("startup")
async def _start_keep_alive():
    asyncio.create_task(_keep_alive())

TICK_BUDGET_S = float(os.getenv("TICK_BUDGET_S", "11"))   # stay well under the judge's timeout
MAX_ACTIONS_PER_TICK = int(os.getenv("MAX_ACTIONS_PER_TICK", "20"))
VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

# ---------------------------------------------------------------------------
# In-memory state (fine for the test: judge says "don't restart between calls")
# ---------------------------------------------------------------------------
contexts: dict[tuple[str, str], dict] = {}          # (scope, id) -> {"version", "payload"}
conversations: dict[str, dict] = {}                 # conversation_id -> state
sent_suppression_keys: set[str] = set()
merchant_auto_replies: dict[str, int] = {}          # merchant_id -> auto-reply count (across conversations)
blocked: set[str] = set()                           # merchant/customer ids that opted out
snoozed_until: dict[str, datetime] = {}             # merchant_id -> datetime
bodies_by_recipient: dict[str, list[str]] = {}      # recipient -> bodies already sent (anti-repetition)


def get_ctx(scope: str, cid: Optional[str]) -> Optional[dict]:
    if not cid:
        return None
    row = contexts.get((scope, cid))
    return row["payload"] if row else None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(s: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# health + metadata
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"ok": True, "see": "/v1/healthz"}


@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.getenv("TEAM_NAME", "Vaibhavi Kansal"),
        "team_members": [m.strip() for m in os.getenv("TEAM_MEMBERS", "Vaibhavi Kansal").split(",")],
        "model": llm.MODEL if llm.enabled() else "template-only (no LLM key set)",
        "approach": ("Trigger-kind router -> deterministic fact-grounded template draft -> LLM polish (temp 0) "
                     "-> validator (no invented numbers/URLs/taboos, single CTA) with template fallback; "
                     "rule-first reply state machine (auto-reply, opt-out, intent->action, off-topic, per-turn language)."),
        "contact_email": os.getenv("CONTACT_EMAIL", "vaibhavikansal2323@gmail.com"),
        "version": "1.0.0",
        "submitted_at": os.getenv("SUBMITTED_AT", "2026-09-26T00:00:00Z"),
    }


# ---------------------------------------------------------------------------
# context push
# ---------------------------------------------------------------------------

@app.post("/v1/context")
async def push_context(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_json", "details": "body is not JSON"})
    scope, cid, version, payload = body.get("scope"), body.get("context_id"), body.get("version"), body.get("payload")
    if scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope", "details": str(scope)})
    if not cid or not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_payload", "details": "context_id and payload required"})
    try:
        version = int(version)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_version", "details": str(version)})

    cur = contexts.get((scope, cid))
    if cur and cur["version"] >= version:
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]})
    contexts[(scope, cid)] = {"version": version, "payload": payload}  # atomic replace
    return {"accepted": True, "ack_id": f"ack_{cid}_v{version}", "stored_at": utcnow().isoformat().replace("+00:00", "Z")}


# ---------------------------------------------------------------------------
# tick: decide what to send proactively
# ---------------------------------------------------------------------------

def _resolve(trg_id: str):
    trg = get_ctx("trigger", trg_id)
    if not trg:
        return None
    mid = trg.get("merchant_id") or (trg.get("payload") or {}).get("merchant_id")
    merchant = get_ctx("merchant", mid)
    if not merchant:
        return None
    slug = merchant.get("category_slug") or (trg.get("payload") or {}).get("category")
    category = get_ctx("category", slug) or {"slug": slug}
    cust_id = trg.get("customer_id") or (trg.get("payload") or {}).get("customer_id")
    customer = get_ctx("customer", cust_id) if cust_id else None
    if trg.get("scope") == "customer" and not customer:
        return None  # can't address a customer we know nothing about -> don't guess
    return trg, merchant, category, customer, mid, cust_id


@app.post("/v1/tick")
async def tick(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    now = parse_ts(body.get("now")) or utcnow()
    ids = body.get("available_triggers") or []

    candidates = []
    for tid in ids:
        r = _resolve(tid)
        if not r:
            continue
        trg, merchant, category, customer, mid, cust_id = r
        key = trg.get("suppression_key") or tid
        if key in sent_suppression_keys:
            continue                                   # already sent this exact nudge
        if mid in blocked or (cust_id and cust_id in blocked):
            continue                                   # opted out
        if not cust_id and mid in snoozed_until and now < snoozed_until[mid]:
            continue                                   # merchant asked us to wait
        candidates.append((int(trg.get("urgency") or 1), tid, trg, merchant, category, customer, mid, cust_id))

    # highest urgency first; max ONE merchant-facing message per merchant per tick
    candidates.sort(key=lambda x: -x[0])
    chosen, merchant_taken = [], set()
    for c in candidates:
        _, tid, trg, merchant, category, customer, mid, cust_id = c
        if not cust_id:
            if mid in merchant_taken:
                continue
            merchant_taken.add(mid)
        chosen.append(c)
        if len(chosen) >= MAX_ACTIONS_PER_TICK:
            break

    async def build(c):
        _, tid, trg, merchant, category, customer, mid, cust_id = c
        recipient = cust_id or mid
        res = await compose_async(category, merchant, trg, customer,
                                  previous_bodies=bodies_by_recipient.get(recipient, []),
                                  llm_timeout=min(9.0, TICK_BUDGET_S - 1))
        return c, res

    results = []
    if chosen:
        tasks = [asyncio.create_task(build(c)) for c in chosen]
        done, pending = await asyncio.wait(tasks, timeout=TICK_BUDGET_S)
        for p in pending:
            p.cancel()
        results = [t.result() for t in done if not t.exception()]
        # anything that didn't finish in time: fall back to the instant template path
        finished = {id(r[0]) for r in results}
        for c in chosen:
            if id(c) not in finished:
                _, tid, trg, merchant, category, customer, mid, cust_id = c
                res = await compose_async(category, merchant, trg, customer,
                                          previous_bodies=bodies_by_recipient.get(cust_id or mid, []), use_llm=False)
                results.append((c, res))

    actions = []
    for c, res in results:
        _, tid, trg, merchant, category, customer, mid, cust_id = c
        conv_id = f"conv_{mid}_{tid}" + (f"_{cust_id}" if cust_id else "")
        n = 2
        while conv_id in conversations:
            conv_id = f"conv_{mid}_{tid}_{n}"
            n += 1
        state = new_state(conv_id, mid, cust_id, tid, res["send_as"])
        state["bot_bodies"].append(res["body"])
        state["turns"].append({"from": "bot", "msg": res["body"]})
        conversations[conv_id] = state
        sent_suppression_keys.add(res["suppression_key"])
        bodies_by_recipient.setdefault(cust_id or mid, []).append(res["body"])
        actions.append({
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": cust_id,
            "send_as": res["send_as"],
            "trigger_id": tid,
            "template_name": res["template_name"],
            "template_params": res["template_params"],
            "body": res["body"],
            "cta": res["cta"],
            "suppression_key": res["suppression_key"],
            "rationale": res["rationale"],
        })
    return {"actions": actions}


# ---------------------------------------------------------------------------
# reply: merchant / customer answered
# ---------------------------------------------------------------------------

@app.post("/v1/reply")
async def reply(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"action": "end", "rationale": "invalid JSON"})
    conv_id = body.get("conversation_id") or "conv_unknown"
    from_role = body.get("from_role") or "merchant"
    message = str(body.get("message") or "")

    state = conversations.get(conv_id)
    if state is None:  # conversation we didn't start (e.g. judge replay) -> build from what we know
        state = new_state(conv_id, body.get("merchant_id"), body.get("customer_id"))
        conversations[conv_id] = state
    mid = state.get("merchant_id") or body.get("merchant_id")
    cust_id = state.get("customer_id") or body.get("customer_id")

    merchant = get_ctx("merchant", mid) or {"merchant_id": mid}
    trigger = get_ctx("trigger", state.get("trigger_id")) or {}
    category = get_ctx("category", merchant.get("category_slug")) or {}
    customer = get_ctx("customer", cust_id) if cust_id else None

    # merchant-level auto-reply memory (same canned text across different conversations)
    from conversation_handlers import classify
    if from_role == "merchant" and classify(message, state["merchant_msgs"]) == "auto_reply":
        merchant_auto_replies[mid] = merchant_auto_replies.get(mid, 0) + 1

    try:
        out = await asyncio.wait_for(
            respond(state, message, category, merchant, trigger, customer,
                    merchant_auto_count=merchant_auto_replies.get(mid, 0), from_role=from_role),
            timeout=20)
    except asyncio.TimeoutError:
        out = {"action": "wait", "wait_seconds": 600, "rationale": "Internal timeout; backing off briefly."}

    # side effects
    if state.get("opted_out"):
        blocked.add(cust_id if from_role == "customer" and cust_id else mid)
    if out["action"] == "wait" and mid and from_role == "merchant":
        now = parse_ts(body.get("received_at")) or utcnow()
        snoozed_until[mid] = datetime.fromtimestamp(now.timestamp() + int(out.get("wait_seconds", 0)), tz=timezone.utc)
    if out["action"] == "send":
        bodies_by_recipient.setdefault(cust_id or mid, []).append(out["body"])
    return out


@app.post("/v1/teardown")
async def teardown():
    for store in (contexts, conversations, merchant_auto_replies, snoozed_until, bodies_by_recipient):
        store.clear()
    sent_suppression_keys.clear()
    blocked.clear()
    return {"ok": True, "wiped": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
