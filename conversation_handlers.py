"""
Multi-turn reply handling (challenge-brief §7.4 + testing-brief §2.3).

Order of checks for every incoming merchant/customer message:
  1. explicit opt-out ("stop", "not interested")   -> end
  2. WhatsApp-Business auto-reply                  -> 1st: one short nudge for the owner,
                                                      2nd: wait 24h, 3rd+: end
  3. hostile / abusive (no opt-out)                -> one short apology + opt-out path, 2nd time -> end
  4. "busy / later"                                 -> wait
  5. off-topic ask (GST, loans ...)                 -> polite decline + back to the topic
  6. clear YES / "let's do it"                      -> ACTION mode immediately (no more questions)
  7. soft "no"                                      -> graceful close
  8. anything else (questions, details)            -> grounded LLM answer, template fallback
Language is re-detected on every turn (merchant can switch English <-> Hinglish).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import llm
from composer import Ctx, _number_universe, _numbers, date_label, first_sentence, humanize

MAX_BOT_TURNS = 5

AUTO_REPLY_PATTERNS = [
    r"thank(s| you) for (contacting|reaching|your message|messaging)", r"our team will", r"will get back",
    r"we will (respond|reply|contact)", r"respond(ing)? (shortly|soon)", r"get back (to you )?(shortly|soon)",
    r"automated (assistant|message|reply|response)", r"auto[- ]?reply", r"currently (unavailable|away|closed)",
    r"out of (the )?office", r"business hours", r"we have received your (message|query)",
    r"aapki jaankari ke liye", r"team tak pahuncha", r"jald hi (sampark|contact|reply)", r"main ek automated",
    r"this is an automated", r"do not reply to this",
]
OPT_OUT_PATTERNS = [
    r"\bstop\b", r"unsubscribe", r"not interested", r"no interest", r"don'?t (message|text|contact|send)",
    r"do not (message|text|contact|send)", r"stop (messaging|sending|texting)", r"no more messages",
    r"leave me alone", r"remove me", r"band karo", r"mat bhejo", r"message mat", r"nahi chahiye",
    r"interest nahi", r"block kar",
]
HOSTILE_PATTERNS = [
    r"useless", r"\bspam", r"idiot", r"stupid", r"nonsense", r"bakwas", r"pagal", r"\bfraud", r"\bscam",
    r"bother(ing)?", r"irritat", r"harass", r"shut up", r"\bbloody\b", r"\bdamn", r"\bf+u+c+k", r"\bshit",
    r"chutiya", r"bewakoof", r"waste of time", r"annoying", r"\bgo away",
]
LATER_PATTERNS = [
    r"\blater\b", r"\bbusy\b", r"baad (mein|me)", r"abhi nahi", r"not now", r"in a meeting",
    r"call (me )?(later|tomorrow)", r"\btomorrow\b", r"\bkal\b", r"after some time", r"thodi der",
]
OFF_TOPIC_PATTERNS = [
    r"\bgst\b", r"income tax", r"\bitr\b", r"\bloan\b", r"\bca\b", r"chartered accountant", r"accountant",
    r"passport", r"\bvisa\b", r"insurance", r"legal notice", r"lawyer", r"electricity bill", r"cricket score",
    r"stock market", r"share price", r"\bcrypto", r"job for my", r"marriage",
]
ACCEPT_PATTERNS = [
    r"^\s*(yes|yess+|yeah|yep|yup|ya|haan|han|ha|ji|ji haan|ok|okay|okk+|k|sure|done|confirm|confirmed|go|chalo|chalega|theek hai|thik hai|perfect|great|interested)\b",
    r"let'?s do (it|this)", r"lets do (it|this)", r"go ahead", r"please (do|proceed|go ahead|send|start)",
    r"\bdo it\b", r"send (it|me|the|now)", r"\bproceed\b", r"kar do", r"kardo", r"\bkaro\b", r"bhej do",
    r"\bbhejo\b", r"i want to (join|start|sign)", r"want to join", r"\bjudna\b", r"judrna", r"jud(na|ne) hai",
    r"sign me up", r"\bstart (it|now)\b", r"\byes please\b", r"what'?s next", r"whats next",
]
SOFT_NO_PATTERNS = [r"^\s*(no|nope|nah|nahi|na)\b", r"not (needed|required|right now)", r"no thanks", r"zaroorat nahi"]

HINGLISH_MARKERS = {
    "hai", "haan", "nahi", "kya", "karo", "kar", "mujhe", "aap", "aapka", "chahiye", "bhai", "ji", "kaise",
    "kyun", "kab", "abhi", "baad", "mein", "theek", "thik", "accha", "acha", "bhejo", "karna", "hoga", "wala",
    "judna", "batao", "bataiye", "kitna", "paisa", "dukaan", "hum", "humara", "mera", "meri",
    "ke", "liye", "ki", "ka", "yeh", "hoon", "hain", "bahut", "shukriya", "aapki", "aapke", "sabhi", "tak",
    "ko", "bhi", "rahi", "raha", "dijiye", "karein", "lagega", "iska", "nahin", "haa", "matlab", "kuch",
}

QUALIFYING_PHRASES = ["would you", "do you", "can you tell", "what if", "how about"]


def _match(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def detect_hinglish(text: str, default: bool) -> bool:
    if re.search(r"[ऀ-ॿ]", text):  # Devanagari
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = len(words & HINGLISH_MARKERS)
    if hits >= 2 or (hits == 1 and len(words) <= 4):
        return True
    if len(words) >= 4 and hits == 0:
        return False
    return default


def classify(text: str, prev_merchant_msgs: list[str]) -> str:
    t = text.strip().lower()
    if not t:
        return "empty"
    if _match(OPT_OUT_PATTERNS, t):
        return "opt_out"
    if _match(AUTO_REPLY_PATTERNS, t) or (prev_merchant_msgs and t in [p.strip().lower() for p in prev_merchant_msgs]):
        return "auto_reply"
    if _match(HOSTILE_PATTERNS, t):
        return "hostile"
    if _match(OFF_TOPIC_PATTERNS, t):
        return "off_topic"
    if _match(ACCEPT_PATTERNS, t) or re.fullmatch(r"[1-9]\.?", t):   # "2" = slot choice
        return "accept"
    if _match(LATER_PATTERNS, t):
        return "later"
    if _match(SOFT_NO_PATTERNS, t):
        return "soft_no"
    return "engaged"


# ---------------------------------------------------------------------------
# What "action mode" delivers for each trigger kind
# ---------------------------------------------------------------------------

DELIVERABLE = {
    "research_digest": ("the 2-min summary + a patient WhatsApp draft", "2-min summary + patient WhatsApp draft"),
    "regulation_change": ("your 5-point audit checklist", "aapki 5-point audit checklist"),
    "cde_opportunity": ("your seat booking", "aapki seat booking"),
    "perf_dip": ("the offer + a fresh Google post", "offer + fresh Google post"),
    "seasonal_perf_dip": ("the 4-week attendance challenge", "4-week attendance challenge"),
    "perf_spike": ("the follow-up Google post", "follow-up Google post"),
    "renewal_due": ("your renewal details", "aapki renewal details"),
    "winback_eligible": ("the restart plan + comeback offer", "restart plan + comeback offer"),
    "dormant_with_vera": ("the Google post", "Google post"),
    "festival_upcoming": ("the festive Google post + WhatsApp creative", "festive Google post + WhatsApp creative"),
    "curious_ask_due": ("the Google post + price-reply template", "Google post + price-reply template"),
    "competitor_opened": ("the counter post (reviews-led, no price war)", "counter post (reviews-led, bina price war)"),
    "review_theme_emerged": ("the public review reply + fix announcement", "public review reply + fix announcement"),
    "milestone_reached": ("the review-request WhatsApp", "review-request WhatsApp"),
    "active_planning_intent": ("the Google post + WhatsApp flyer", "Google post + WhatsApp flyer"),
    "ipl_match_today": ("the banner + Insta story", "banner + Insta story"),
    "category_seasonal": ("the shelf + restock checklist", "shelf + restock checklist"),
    "supply_alert": ("the customer WhatsApp + replacement-pickup note", "customer WhatsApp + replacement-pickup note"),
    "gbp_unverified": ("the verification steps", "verification steps"),
}


def _slots(ctx: Ctx) -> list[str]:
    opts = ctx.p.get("available_slots") or ctx.p.get("next_session_options") or []
    return [o.get("label") for o in opts if isinstance(o, dict) and o.get("label")]


def action_body(ctx: Ctx, hinglish: bool, customer_side: bool, message: str) -> str:
    if customer_side:
        slots = _slots(ctx)
        chosen = None
        m = re.search(r"\b([12])\b", message)
        if m and len(slots) >= int(m.group(1)):
            chosen = slots[int(m.group(1)) - 1]
        elif slots:
            chosen = slots[0]
        who = f" {ctx.cust_name}" if ctx.cust_name else ""
        if ctx.kind == "chronic_refill_due":
            return ("Confirmed ✅ Aapka order pack ho raha hai — dispatch ke baad update bhejenge. Dose mein koi change ho toh reply karein."
                    if hinglish else f"Confirmed{who} ✅ Your order is being packed — we'll message you once it's dispatched. Reply here if the dose has changed.")
        if chosen:
            return (f"Done{who} ✅ {chosen} aapke liye book ho gaya — {ctx.name}. Ek din pehle reminder bhejenge. Time badalna ho toh CHANGE reply karein."
                    if hinglish else f"Done{who} ✅ You're booked for {chosen} at {ctx.name}. We'll send a reminder a day before. Reply CHANGE if you need another time.")
        return (f"Done{who} ✅ Hum aapke liye slot hold kar rahe hain — agle message mein is hafte ke open times bhejenge."
                if hinglish else f"Done{who} ✅ We're holding a slot for you — next message will have this week's open times. Reply CHANGE anytime.")

    if ctx.kind not in DELIVERABLE:
        return ("Done ✅ Main abhi shuru kar rahi hoon — pehla draft 10 min mein yahin bhejungi.\n\n"
                "Next step: draft dekh ke CONFIRM reply karein, main live kar dungi."
                if hinglish else
                "Done ✅ Starting on it now — I'll share the first draft right here in 10 min.\n\n"
                "Next step: check it and reply CONFIRM, and I'll make it live.")
    en, hi = DELIVERABLE[ctx.kind]
    extra = ""
    if ctx.kind == "research_digest":
        item = ctx.digest_item(ctx.p.get("top_item_id"), kinds=("research",))
        if item:
            extra = f'\n\nPatient WhatsApp draft:\n"{item["title"]}. {first_sentence(item.get("actionable", "")).rstrip(".")}. Reply to book a quick check."'
    elif ctx.kind == "regulation_change":
        item = ctx.digest_item(ctx.p.get("top_item_id"), kinds=("compliance",))
        if item:
            extra = f"\n\nStep 1: {first_sentence(item.get('actionable', 'Audit your setup'))}"
    elif ctx.kind in ("perf_dip", "festival_upcoming", "competitor_opened") and (ctx.active_offers or ctx.catalog):
        o = ctx.active_offers[0] if ctx.active_offers else ctx.best_offer()
        extra = f'\n\nPost draft: "{o} at {ctx.name}, {ctx.locality}. Book on WhatsApp today."'
    if hinglish:
        return (f"Done ✅ {hi} abhi draft kar rahi hoon — 10 min mein ready.{extra}\n\n"
                f"Next step: CONFIRM reply karein aur main ise live kar dungi.")
    return (f"Done ✅ Drafting {en} now — ready in 10 min.{extra}\n\n"
            f"Next step: reply CONFIRM and I'll make it live.")


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def new_state(conversation_id: str, merchant_id: Optional[str], customer_id: Optional[str] = None,
              trigger_id: Optional[str] = None, send_as: str = "vera") -> dict:
    return {
        "conversation_id": conversation_id, "merchant_id": merchant_id, "customer_id": customer_id,
        "trigger_id": trigger_id, "send_as": send_as, "bot_bodies": [], "merchant_msgs": [],
        "turns": [], "auto_count": 0, "hostile_count": 0, "closing": False, "ended": False,
        "offtopic_count": 0,
    }


REPLY_SYSTEM = """You are Vera, magicpin's WhatsApp assistant for merchants, mid-conversation (do NOT introduce yourself).
Reply to the latest message. Rules:
- Use ONLY the facts provided; if you don't know something, say you'll check — never invent numbers, prices, names or dates.
- Answer the question directly first, then move the task forward. Stay on the topic of the conversation.
- If the person agreed to something, DO it (share the draft / confirm) — do not ask qualifying questions.
- One call-to-action, as the last sentence. No URLs. 1-4 short sentences.
- Language: {language}. Tone: {tone}.
- {role_note}
Return JSON only: {{"body": "...", "cta": "binary_yes_no|open_ended|binary_confirm_cancel|none", "rationale": "..."}}"""


async def respond(state: dict, message: str, category: dict, merchant: dict, trigger: dict,
                  customer: Optional[dict], merchant_auto_count: int, from_role: str = "merchant") -> dict:
    """Returns {"action": send|wait|end, ...}. Mutates `state`."""
    ctx = Ctx(category or {}, merchant or {}, trigger or {}, customer)
    ctx._last_msg = message
    customer_side = from_role == "customer"
    default_hi = ctx.cust_hinglish if customer_side else ctx.merchant_hinglish
    hinglish = detect_hinglish(message, default_hi)
    label = classify(message, state["merchant_msgs"])
    state["merchant_msgs"].append(message)
    state["turns"].append({"from": from_role, "msg": message, "label": label})

    def send(body: str, cta: str, rationale: str) -> dict:
        if body.strip() in [b.strip() for b in state["bot_bodies"]]:
            body += " 🙂" if not body.endswith("🙂") else " (just checking)"
        state["bot_bodies"].append(body)
        state["turns"].append({"from": "bot", "msg": body})
        return {"action": "send", "body": body, "cta": cta, "rationale": rationale}

    def end(rationale: str) -> dict:
        state["ended"] = True
        return {"action": "end", "rationale": rationale}

    if state.get("ended"):
        return end("Conversation already closed; not sending more.")

    if label == "empty":
        return {"action": "wait", "wait_seconds": 1800, "rationale": "Empty message; waiting."}

    if label == "opt_out":
        state["opted_out"] = True
        return end("Explicit opt-out ('stop' / 'not interested'). Closing and suppressing future sends to this contact.")

    if label == "auto_reply":
        state["auto_count"] += 1
        n = max(state["auto_count"], merchant_auto_count)
        if n >= 3:
            return end(f"Same WhatsApp-Business auto-reply seen {n}x — no human on the line. Closing to avoid wasting turns.")
        if n == 2:
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": "Auto-reply again — owner not at phone. Backing off 24h before one retry."}
        body = ("Lagta hai yeh auto-reply hai 🙂 Owner/manager jab dekhein, bas YES reply kar dein — baaki main sambhal lungi."
                if hinglish else "Looks like an auto-reply 🙂 When the owner sees this, just reply YES and I'll take it from there.")
        return send(body, "binary_yes_no", "Detected WhatsApp-Business auto-reply (canned phrasing). One short nudge flagged for the owner; no re-pitch.")

    if len(state["bot_bodies"]) >= MAX_BOT_TURNS:
        return end("Reached turn budget for this conversation; closing gracefully instead of over-messaging.")

    if label == "hostile":
        state["hostile_count"] += 1
        if state["hostile_count"] >= 2:
            return end("Repeated hostility — exiting gracefully.")
        body = ("Maaf kijiye agar messages zyada lage 🙏 Main sirf {n} ke Google listing aur customers badhane mein help karti hoon. "
                "Band karna ho toh STOP likhiye — main dobara message nahi karungi.").format(n=ctx.name) if hinglish else (
                f"Sorry if these felt like too much 🙏 I only help {ctx.name} with its Google listing and customers. "
                "Reply STOP and I won't message again.")
        return send(body, "none", "Hostile tone without explicit opt-out: one calm apology + clear opt-out path, no pitch.")

    if label == "later":
        wait = 86400 if re.search(r"tomorrow|\bkal\b", message.lower()) else 3600
        return {"action": "wait", "wait_seconds": wait, "rationale": f"Merchant asked for time; backing off {wait // 3600}h."}

    if label == "off_topic":
        state["offtopic_count"] += 1
        topic = _topic_line(ctx, hinglish)
        body = ("Yeh mere scope se bahar hai — iske liye aapke CA ya bank best rahenge. " + topic) if hinglish else (
                "That's outside what I can help with — your CA or bank is the right person for it. " + topic)
        return send(body, "binary_yes_no", "Out-of-scope ask politely declined; redirected to the original topic without losing the thread.")

    if label == "accept":
        body = action_body(ctx, hinglish, customer_side, message)
        if llm.enabled() and not customer_side and ctx.kind in ("active_planning_intent", "research_digest", "review_theme_emerged",
                                                                   "competitor_opened", "festival_upcoming", "perf_dip", "curious_ask_due"):
            improved = await _llm_reply(ctx, state, message, hinglish, customer_side, mode="action")
            if improved and not any(q in improved["body"].lower() for q in QUALIFYING_PHRASES):
                return send(improved["body"], improved.get("cta") or "binary_confirm_cancel",
                            "Merchant committed — switched to action mode and delivered the draft (no re-qualifying).")
        return send(body, "binary_confirm_cancel",
                    "Explicit intent detected — switched from pitch to action immediately; concrete next step + single CONFIRM.")

    if label == "soft_no":
        if state["closing"]:
            return end("Second decline — closing.")
        state["closing"] = True
        body = ("Koi baat nahi 👍 Jab zarurat ho, bas 'Hi Vera' likh dijiye." if hinglish
                else "No problem 👍 Whenever you need anything, just message 'Hi Vera'.")
        return send(body, "none", "Soft decline — graceful close, door left open.")

    # engaged: question / details
    out = await _llm_reply(ctx, state, message, hinglish, customer_side, mode="answer")
    if out:
        return send(out["body"], out.get("cta") or "open_ended", out.get("rationale") or "Answered the question from known facts and moved the task forward.")
    return send(_fallback_answer(ctx, hinglish, customer_side), "binary_yes_no",
                "Engaged reply; acknowledged and moved to a single concrete next step (template fallback).")


def _topic_line(ctx: Ctx, hinglish: bool) -> str:
    en, hi = DELIVERABLE.get(ctx.kind, ("the next step for your listing", "aapki listing ka next step"))
    return (f"Wapas apne topic pe — {hi} bhej doon? Reply YES." if hinglish
            else f"Coming back to our topic — shall I send {en}? Reply YES.")


def state_last_msg(ctx: Ctx) -> str:
    return getattr(ctx, "_last_msg", "")


def _fallback_answer(ctx: Ctx, hinglish: bool, customer_side: bool) -> str:
    if customer_side:
        return ("Shukriya! Main team se confirm karke abhi batati hoon. Tab tak slot hold karna ho toh YES reply karein."
                if hinglish else "Thanks! Let me check with the team and get back shortly. Reply YES if you'd like us to hold a slot meanwhile.")
    en, hi = DELIVERABLE.get(ctx.kind, ("a ready draft", "ready draft"))
    if "?" in state_last_msg(ctx):
        return (f"Accha sawaal — mere paas abhi iska confirmed data nahi hai, main check karke {hi} ke saath bhejti hoon. Bhej doon? Reply YES."
                if hinglish else f"Good question — I don't have verified data on that yet; I'll check and include it with {en}. Send it over? Reply YES.")
    return (f"Samajh gayi, noted 👍 Main ise dhyan mein rakh ke {hi} bana deti hoon — bas YES reply karein."
            if hinglish else f"Got it, noted 👍 I'll factor that in and prepare {en} for you — just reply YES to go ahead.")


async def _llm_reply(ctx: Ctx, state: dict, message: str, hinglish: bool, customer_side: bool, mode: str) -> Optional[dict]:
    if not llm.enabled():
        return None
    role_note = ("You are writing on behalf of the business to its customer; be warm and brief." if customer_side
                 else "You are talking to the business owner.")
    if mode == "action":
        role_note += " The merchant just said YES: deliver the actual draft now (e.g. the post text / plan), then ask them to reply CONFIRM to publish. No questions."
    system = REPLY_SYSTEM.format(
        language="Hinglish (Roman script)" if hinglish else "English",
        tone="clinical peer" if ctx.slug == "dentists" else "warm, practical",
        role_note=role_note,
    )
    facts = {
        "business": ctx.name, "owner": ctx.sal, "locality": ctx.locality, "active_offers": ctx.active_offers,
        "catalog": ctx.catalog[:6], "performance_30d": ctx.perf, "trigger": {"kind": ctx.kind, "payload": ctx.p},
    }
    user = json.dumps({"facts": facts, "conversation": state["turns"][-8:], "latest_message": message},
                      ensure_ascii=False, default=str)
    out = await llm.chat_json(system, user, max_tokens=400, timeout=8)
    if not out or not isinstance(out.get("body"), str) or len(out["body"].strip()) < 10:
        return None
    body = out["body"].strip()
    universe = _number_universe(facts, state["turns"]) | {"10"}
    bad = [n for n in _numbers(body) if n not in universe and not (n.isdigit() and int(n) <= 10)]
    if bad or "http" in body.lower() or len(body) > 1100:
        return None
    out["body"] = body
    return out
