"""
Message composer for the magicpin "Vera" challenge.

compose(category, merchant, trigger, customer) -> {body, cta, send_as, suppression_key, rationale, ...}

How it works (two layers):
  1. A deterministic TEMPLATE layer. One small function per trigger kind. It only
     uses facts that exist in the 4 contexts, so it can never hallucinate. This is
     also the fallback when the LLM is off, slow, or rate-limited.
  2. An LLM POLISH layer (Groq by default). It gets the template draft + the
     verified facts and rewrites the message to sound more natural and compelling.
     Its output is then VALIDATED (no URLs, no taboo words, no numbers that are
     not in the contexts, name present, not a repeat). If validation fails, the
     template draft is used instead.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any, Optional

import llm

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

REGIONAL_GREETING = {"ta": "Vanakkam", "kn": "Namaskara", "te": "Namaskaram", "mr": "Namaskar"}

CATEGORY_NOUN = {
    "dentists": "dental clinic", "salons": "salon", "restaurants": "restaurant",
    "gyms": "gym", "pharmacies": "pharmacy",
}
CUSTOMER_NOUN = {  # what the merchant calls its customers
    "dentists": "patients", "salons": "clients", "restaurants": "customers",
    "gyms": "members", "pharmacies": "customers",
}
EMOJI = {"dentists": "🦷", "salons": "✨", "restaurants": "🍽️", "gyms": "💪", "pharmacies": "💊"}


def g(d: Any, *path, default=None):
    """Safe nested get: g(merchant, 'identity', 'name')."""
    cur = d
    for p in path:
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list) and isinstance(p, int) and -len(cur) <= p < len(cur):
            cur = cur[p]
        else:
            return default
        if cur is None:
            return default
    return cur


def pct(x: Any, signed: bool = False) -> str:
    """0.18 -> '18%'. Values > 1 are treated as already-percent (e.g. 40 -> '40%')."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v) <= 1.5:
        v *= 100
    s = f"{abs(v):.0f}%" if abs(v) >= 1 else f"{abs(v):.1f}%"
    if signed:
        s = ("+" if v >= 0 else "-") + s
    return s


def money(x: Any) -> str:
    try:
        return f"₹{int(float(x)):,}"
    except (TypeError, ValueError):
        return str(x)


def num(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


def date_label(iso: Optional[str]) -> str:
    """'2026-11-12' or '2026-11-12T18:00:00+05:30' -> '12 Nov'."""
    if not iso:
        return ""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(iso))
    if not m:
        return str(iso)
    return f"{int(m.group(3))} {MONTHS[int(m.group(2)) - 1]}"


def time_label(iso: Optional[str]) -> str:
    m = re.search(r"T(\d{2}):(\d{2})", str(iso or ""))
    if not m:
        return ""
    h, mi = int(m.group(1)), int(m.group(2))
    suffix = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12}:{mi:02d}{suffix}" if mi else f"{h12}{suffix}"


def humanize(s: Any) -> str:
    return str(s or "").replace("_", " ").strip()


ABBREV = ("dr", "mr", "mrs", "ms", "st", "vs", "no", "approx", "e.g", "i.e")


def first_sentence(text: str) -> str:
    """First sentence, without breaking on 'Dr.' / 'vs.' / decimals."""
    text = (text or "").strip()
    for m in re.finditer(r"[.!?](\s+|$)", text):
        before = text[:m.start()].split()
        last = before[-1].lower().rstrip(".") if before else ""
        if last in ABBREV or (len(last) == 1 and last.isalpha()):
            continue
        return text[:m.start() + 1].strip()
    return text


# ---------------------------------------------------------------------------
# Context wrapper: every derived value the templates need, computed once.
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, category: dict, merchant: dict, trigger: dict, customer: Optional[dict]):
        self.cat = category or {}
        self.m = merchant or {}
        self.t = trigger or {}
        self.c = customer
        self.p = self.t.get("payload") or {}
        self.kind = self.t.get("kind", "generic")
        self.slug = self.cat.get("slug") or self.m.get("category_slug") or ""
        self.placeholder = bool(self.p.get("placeholder"))

        ident = self.m.get("identity") or {}
        self.name = ident.get("name") or "your business"
        self.owner = ident.get("owner_first_name") or ""
        self.locality = ident.get("locality") or ""
        self.city = ident.get("city") or ""
        langs = [str(x).lower() for x in (ident.get("languages") or ["en"])]
        self.merchant_hinglish = "hi" in langs

        if self.slug == "dentists" and self.owner:
            self.sal = f"Dr. {self.owner}"
        else:
            self.sal = self.owner or self.name

        self.perf = self.m.get("performance") or {}
        self.delta = self.perf.get("delta_7d") or {}
        self.peer = self.cat.get("peer_stats") or {}
        self.agg = self.m.get("customer_aggregate") or {}
        self.signals = self.m.get("signals") or []
        self.active_offers = [o.get("title") for o in (self.m.get("offers") or [])
                              if o.get("status") == "active" and o.get("title")]
        self.expired_offers = [o.get("title") for o in (self.m.get("offers") or [])
                               if o.get("status") != "active" and o.get("title")]
        self.catalog = [o.get("title") for o in (self.cat.get("offer_catalog") or []) if o.get("title")]
        self.taboos = [str(x).lower() for x in (g(self.cat, "voice", "vocab_taboo", default=[]) or [])]
        self.cust_noun = CUSTOMER_NOUN.get(self.slug, "customers")
        self.noun = CATEGORY_NOUN.get(self.slug, "business")

        # customer-side
        self.cust_hinglish = False
        self.cust_greet = "Hi"
        self.cust_name = ""
        self.child_name = ""
        if self.c:
            cid = self.c.get("identity") or {}
            pref = str(cid.get("language_pref") or "en").lower()
            self.cust_hinglish = pref.startswith("hi")
            code = pref[:2]
            if self.cust_hinglish:
                self.cust_greet = "Namaste"
            elif code in REGIONAL_GREETING:
                self.cust_greet = REGIONAL_GREETING[code]
            raw = str(cid.get("name") or "")
            pm = re.match(r"\s*([^()]+?)\s*\(parent:\s*([^)]+)\)", raw)
            if pm:  # "Aanya (parent: Sneha)" -> talk to Sneha about Aanya
                self.child_name, self.cust_name = pm.group(1).strip(), pm.group(2).strip()
            elif raw and "no profile" not in raw and "walk-in" not in raw:
                self.cust_name = raw.strip()

    # --- helpers used by several templates ---
    def digest_item(self, *ids, kinds=()) -> Optional[dict]:
        digest = self.cat.get("digest") or []
        for i in ids:
            if i:
                for d in digest:
                    if d.get("id") == i:
                        return d
        for k in kinds:
            for d in digest:
                if d.get("kind") == k:
                    return d
        return None

    def cohort_anchor(self) -> str:
        a = self.agg
        if a.get("high_risk_adult_count"):
            return f"your {num(a['high_risk_adult_count'])} high-risk adult patients"
        if a.get("chronic_rx_count"):
            return f"your {num(a['chronic_rx_count'])} chronic-Rx customers"
        if a.get("total_active_members"):
            return f"your {num(a['total_active_members'])} active members"
        if a.get("total_unique_ytd"):
            return f"your {num(a['total_unique_ytd'])} {self.cust_noun} this year"
        return f"your {self.cust_noun}"

    def best_offer(self, keywords=()) -> Optional[str]:
        """Prefer the merchant's own active offer; else a service+price item from the category catalog."""
        pools = [self.active_offers, self.catalog]
        for pool in pools:
            for kw in keywords:
                for o in pool:
                    if kw.lower() in o.lower():
                        return o
        if self.active_offers:
            return self.active_offers[0]
        for o in self.catalog:  # service @ price beats "% off"
            if "@" in o and "%" not in o:
                return o
        return self.catalog[0] if self.catalog else None

    def perf_line(self) -> str:
        bits = []
        if self.perf.get("views") is not None:
            bits.append(f"{num(self.perf['views'])} views")
        if self.perf.get("calls") is not None:
            bits.append(f"{num(self.perf['calls'])} calls")
        if self.perf.get("directions") is not None:
            bits.append(f"{num(self.perf['directions'])} direction requests")
        return ", ".join(bits)

    def last_merchant_msg(self) -> Optional[str]:
        for turn in reversed(self.m.get("conversation_history") or []):
            if turn.get("from") == "merchant" and turn.get("body"):
                return turn["body"]
        return None

    def signal_value(self, prefix: str) -> Optional[str]:
        for s in self.signals:
            if str(s).startswith(prefix):
                return str(s).split(":", 1)[1] if ":" in str(s) else ""
        return None


def M(ctx: Ctx, en: str, hi: str) -> str:
    """Pick English or Hinglish for merchant-facing text."""
    return hi if ctx.merchant_hinglish else en


def C(ctx: Ctx, en: str, hi: str) -> str:
    """Pick English or Hinglish for customer-facing text."""
    return hi if ctx.cust_hinglish else en


# ---------------------------------------------------------------------------
# Merchant-facing templates. Each returns (body, cta, rationale, template_params).
# ---------------------------------------------------------------------------

def t_research(ctx: Ctx):
    item = ctx.digest_item(ctx.p.get("top_item_id"), ctx.p.get("digest_item_id"), kinds=("research", "trend", "tech"))
    if not item:
        return t_generic(ctx)
    trial = f" ({num(item['trial_n'])}-patient trial)" if item.get("trial_n") else ""
    anchor = ctx.cohort_anchor()
    body = M(ctx,
             f"{ctx.sal}, {item.get('source', 'this week’s digest')} has one item relevant to {anchor}: "
             f"{item['title']}{trial}. {first_sentence(item.get('summary', ''))} "
             f"Want me to pull a 2-min summary + draft a WhatsApp note you can share with {ctx.cust_noun}? Reply YES.",
             f"{ctx.sal}, {item.get('source', 'is hafte ke digest')} mein ek item {anchor.replace('your ', 'aapke ', 1)} ke liye relevant hai: "
             f"{item['title']}{trial}. {first_sentence(item.get('summary', ''))} "
             f"Main 2-min summary + {ctx.cust_noun} ke liye WhatsApp note draft kar doon? Bas YES reply karein.")
    return body, "binary_yes_no", f"External digest item '{item['title']}' ({item.get('source')}) tied to {anchor}; reciprocity + low-friction YES.", [ctx.sal, item.get("source", ""), item["title"]]


def t_regulation(ctx: Ctx):
    item = ctx.digest_item(ctx.p.get("top_item_id"), ctx.p.get("digest_item_id"), kinds=("compliance",))
    if not item:
        return t_generic(ctx)
    deadline = date_label(ctx.p.get("deadline_iso")) or ""
    dl = f" Deadline: {deadline}." if deadline else ""
    body = M(ctx,
             f"{ctx.sal}, compliance heads-up — {item['title']} ({item.get('source', '')}). "
             f"{item.get('summary', '')}{dl} Want a 5-point checklist to audit your setup this week? Reply YES.",
             f"{ctx.sal}, compliance heads-up — {item['title']} ({item.get('source', '')}). "
             f"{item.get('summary', '')}{dl} Main aapke setup ke liye 5-point audit checklist bhej doon? Reply YES.")
    return body, "binary_yes_no", f"Regulation change with a hard deadline ({deadline or 'n/a'}); loss-aversion + effort externalization via checklist.", [ctx.sal, item["title"], deadline]


def t_cde(ctx: Ctx):
    item = ctx.digest_item(ctx.p.get("digest_item_id"), ctx.p.get("top_item_id"), kinds=("event", "cde"))
    credits = ctx.p.get("credits")
    fee = humanize(ctx.p.get("fee"))
    if not item:
        return t_generic(ctx)
    extra = ", ".join(x for x in [f"{credits} CDE credits" if credits else "", fee] if x)
    body = M(ctx,
             f"{ctx.sal}, {item['title']} ({item.get('source', '')}){' — ' + extra if extra else ''}. "
             f"{first_sentence(item.get('summary', ''))} Want me to save you a seat? Reply YES.",
             f"{ctx.sal}, {item['title']} ({item.get('source', '')}){' — ' + extra if extra else ''}. "
             f"{first_sentence(item.get('summary', ''))} Aapke liye seat book kar doon? Reply YES.")
    return body, "binary_yes_no", "Professional-development opportunity from the category digest; free credits = easy yes.", [ctx.sal, item["title"], extra]


def _pick_delta(ctx: Ctx, want_negative: bool):
    metric = ctx.p.get("metric")
    delta = ctx.p.get("delta_pct")
    if metric and delta is not None:
        return metric, delta
    best = None
    for k, v in ctx.delta.items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if best is None or (v < best[1] if want_negative else v > best[1]):
            best = (k.replace("_pct", ""), v)
    return best if best else (None, None)


def t_perf_dip(ctx: Ctx):
    metric, delta = _pick_delta(ctx, want_negative=True)
    base = ctx.p.get("vs_baseline")
    parts = []
    if metric and delta is not None and float(delta) < 0:
        w = humanize(ctx.p.get("window") or "7d")
        parts.append(f"your {metric} are down {pct(delta)} ({w})" + (f" vs a baseline of ~{base}" if base else ""))
    peer_calls = ctx.peer.get("avg_calls_30d")
    if ctx.perf.get("calls") is not None and peer_calls and ctx.perf["calls"] < peer_calls:
        parts.append(f"30-day calls {ctx.perf['calls']} vs peer avg {peer_calls}")
    if ctx.perf.get("ctr") and ctx.peer.get("avg_ctr") and ctx.perf["ctr"] < ctx.peer["avg_ctr"]:
        parts.append(f"CTR {pct(ctx.perf['ctr'])} vs peer {pct(ctx.peer['avg_ctr'])}")
    headline = "; ".join(parts) or f"your listing numbers slipped this week ({ctx.perf_line()})"
    offer = ctx.best_offer()
    if not ctx.active_offers and offer:
        fix = M(ctx, f"You have no live offer right now — a service+price hook like “{offer}” usually brings calls back.",
                f"Abhi koi live offer nahi hai — “{offer}” jaisa service+price hook calls wapas laata hai.")
    elif ctx.signal_value("stale_posts") is not None:
        fix = M(ctx, f"Your last Google post was {ctx.signal_value('stale_posts')} ago — a fresh post is the quickest lift.",
                f"Aapki last Google post {ctx.signal_value('stale_posts')} purani hai — fresh post sabse quick lift hai.")
    else:
        fix = M(ctx, f"Pushing “{offer}” in a fresh Google post is the quickest lift." if offer else "A fresh Google post is the quickest lift.",
                f"“{offer}” ko fresh Google post mein push karna sabse quick lift hai." if offer else "Ek fresh Google post sabse quick lift hai.")
    body = M(ctx,
             f"{ctx.sal}, quick flag — {headline}. {fix} Want me to set it up today? Reply YES.",
             f"{ctx.sal}, quick flag — {headline}. {fix} Aaj hi set kar doon? Reply YES.")
    return body, "binary_yes_no", "Performance dip with concrete numbers + peer benchmark (loss aversion); one ready fix, YES to execute.", [ctx.sal, headline, offer or ""]


def t_perf_spike(ctx: Ctx):
    metric, delta = _pick_delta(ctx, want_negative=False)
    driver = humanize(ctx.p.get("likely_driver"))
    if metric and delta is not None and float(delta) > 0:
        head = f"your {metric} are up {pct(delta)} this week" + (f" (vs a baseline of ~{ctx.p['vs_baseline']})" if ctx.p.get("vs_baseline") else "")
    else:
        head = f"your listing had a strong month — {ctx.perf_line()}"
    drv = M(ctx, f" Looks driven by your {driver}." if driver else "", f" Lagta hai aapki {driver} se aaya hai." if driver else "")
    body = M(ctx,
             f"{ctx.sal}, good news — {head}.{drv} Momentum fades in ~7 days unless we follow up. Want me to draft a follow-up post to ride it? Reply YES.",
             f"{ctx.sal}, good news — {head}.{drv} Follow-up na ho toh momentum ~7 din mein fade hota hai. Ek follow-up post draft kar doon? Reply YES.")
    return body, "binary_yes_no", "Positive spike — reinforce with a timely follow-up; curiosity + effort externalization.", [ctx.sal, head, driver]


def t_seasonal_dip(ctx: Ctx):
    metric, delta = _pick_delta(ctx, want_negative=True)
    note = humanize(ctx.p.get("season_note"))
    beat = next((b for b in (ctx.cat.get("seasonal_beats") or []) if "Apr" in str(b.get("month_range", ""))), None)
    head = f"{metric} down {pct(delta)} this week" if metric and delta is not None else "a softer week"
    ctxline = f" This is the expected {note}" + (f" ({beat['note']})" if beat else "") + "." if note or beat else ""
    members = ctx.agg.get("total_active_members")
    focus = f" Better use of this window: keep your {members} members engaged." if members else " Better use of this window: retention over acquisition."
    body = M(ctx,
             f"{ctx.sal}, {head} — don’t panic.{ctxline}{focus} Want me to draft a 4-week attendance challenge? Reply YES.",
             f"{ctx.sal}, {head} — ghabraiye mat.{ctxline}{focus} 4-week attendance challenge draft kar doon? Reply YES.")
    return body, "binary_yes_no", "Seasonal (expected) dip — reframe to prevent panic spend, redirect to retention.", [ctx.sal, head, note]


def t_renewal(ctx: Ctx):
    days = ctx.p.get("days_remaining", g(ctx.m, "subscription", "days_remaining"))
    plan = ctx.p.get("plan", g(ctx.m, "subscription", "plan", default=""))
    amount = ctx.p.get("renewal_amount")
    amt = f" ({money(amount)}, about {money(round(float(amount) / 30))}/day)" if amount else ""
    # one concrete win the renewal unlocks, from the merchant's own data
    lapsed = ctx.agg.get("lapsed_180d_plus") or ctx.agg.get("lapsed_90d_plus")
    calls_d = ctx.delta.get("calls_pct")
    hook_en = hook_hi = ""
    if lapsed:
        hook_en = f" First thing I’d run after renewal: a win-back message to your {lapsed} lapsed {ctx.cust_noun}."
        hook_hi = f" Renewal ke baad pehla kaam: aapke {lapsed} lapsed {ctx.cust_noun} ko win-back message."
    elif calls_d is not None and float(calls_d) < 0:
        hook_en = f" Calls are down {pct(calls_d)} this week, so this is the wrong time to lose visibility."
        hook_hi = f" Is hafte calls {pct(calls_d)} gire hain, visibility khone ka yeh sahi time nahi hai."
    body = M(ctx,
             f"{ctx.sal}, your {plan} plan renews in {days} days{amt}. Last 30 days it brought {ctx.name} {ctx.perf_line()}.{hook_en} "
             f"Want me to send the renewal details now? Reply YES.",
             f"{ctx.sal}, aapka {plan} plan {days} din mein renew hona hai{amt}. Pichhle 30 din mein isse {ctx.name} ko {ctx.perf_line()} mile.{hook_hi} "
             f"Renewal details abhi bhej doon? Reply YES.")
    return body, "binary_yes_no", f"Renewal due in {days} days; value delivered (30-day numbers), per-day cost, and a concrete next win from their own data.", [ctx.sal, plan, str(days)]


def t_winback(ctx: Ctx):
    d = ctx.p.get("days_since_expiry", g(ctx.m, "subscription", "days_since_expiry"))
    dip = ctx.p.get("perf_dip_pct")
    lapsed = ctx.p.get("lapsed_customers_added_since_expiry")
    bits = []
    if dip is not None:
        bits.append(f"views/calls are down {pct(dip)}")
    if lapsed:
        bits.append(f"{lapsed} more {ctx.cust_noun} have gone inactive")
    what = " and ".join(bits) if bits else f"your listing is at {ctx.perf_line()}"
    body = M(ctx,
             f"{ctx.sal}, it’s been {d} days since {ctx.name}’s plan expired — since then {what}. "
             f"I can restart it and send a comeback offer to lapsed {ctx.cust_noun} the same day. Want the details? Reply YES.",
             f"{ctx.sal}, {ctx.name} ka plan expire hue {d} din ho gaye — tab se {what}. "
             f"Main restart karke usi din lapsed {ctx.cust_noun} ko comeback offer bhej sakti hoon. Details bhejun? Reply YES.")
    return body, "binary_yes_no", "Win-back: quantified loss since expiry + effort externalization.", [ctx.sal, str(d), what]


def t_dormant(ctx: Ctx):
    days = ctx.p.get("days_since_last_merchant_message")
    trend = (ctx.cat.get("trend_signals") or [None])[0]
    hook = ""
    if trend:
        hook = f"'{trend.get('query')}' searches are up {pct(trend.get('delta_yoy'))} YoY."
    gap = f"It’s been {days} days since we spoke. " if days else ""
    gap_hi = f"{days} din ho gaye baat kiye. " if days else ""
    body = M(ctx,
             f"{ctx.sal}, {gap}One thing I spotted for {ctx.locality or ctx.city} {ctx.noun}s: {hook} "
             f"Quick question — what’s your most-asked service this week? I’ll turn it into a Google post for you.",
             f"{ctx.sal}, {gap_hi}{ctx.locality or ctx.city} ke {ctx.noun}s ke liye ek cheez dikhi: {hook} "
             f"Ek quick sawaal — is hafte sabse zyada kaunsi service poochi gayi? Main uska Google post bana dungi.")
    return body, "open_ended", "Dormant merchant — re-open with a fresh verifiable trend + an easy question (asking-the-merchant lever).", [ctx.sal, hook]


def t_festival(ctx: Ctx):
    fest = ctx.p.get("festival")
    offer = ctx.best_offer()
    if fest:
        when = f"{fest} is on {date_label(ctx.p.get('date'))}" + (f" ({ctx.p['days_until']} days away)" if ctx.p.get("days_until") else "")
    else:
        when = "the festive season is coming up"
    o = f" “{offer}” is a strong hook for it." if offer else ""
    o_hi = f" “{offer}” iske liye strong hook hai." if offer else ""
    body = M(ctx,
             f"{ctx.sal}, {when}. Festive searches start 2-3 weeks early, so listings that post first get seen first.{o} "
             f"Want me to draft a festive Google post + WhatsApp creative for {ctx.name}? Reply YES.",
             f"{ctx.sal}, {when}. Festive searches 2-3 hafte pehle shuru ho jaati hain — jo pehle post karta hai woh pehle dikhta hai.{o_hi} "
             f"{ctx.name} ke liye festive Google post + WhatsApp creative draft kar doon? Reply YES.")
    return body, "binary_yes_no", "Festival window — early-mover framing + merchant's own offer as hook.", [ctx.sal, when, offer or ""]


def t_curious(ctx: Ctx):
    offers = ", ".join(ctx.active_offers[:2])
    cur = f" Is it still {offers}, or something new?" if offers else ""
    cur_hi = f" Abhi bhi {offers}, ya kuch naya?" if offers else ""
    trend = (ctx.cat.get("trend_signals") or [None])[0]
    if trend and trend.get("query") and trend.get("delta_yoy") is not None:
        cur += f" (For context: '{trend['query']}' searches are up {pct(trend['delta_yoy'])} YoY.)"
        cur_hi += f" (Context: '{trend['query']}' searches {pct(trend['delta_yoy'])} YoY badhi hain.)"
    body = M(ctx,
             f"Hi {ctx.sal}! Quick one — which service was most asked-for at {ctx.name} this week?{cur} "
             f"Reply with just the name — I’ll turn it into a Google post + a ready WhatsApp reply for price questions. 5 min, zero effort from you.",
             f"Hi {ctx.sal}! Ek quick sawaal — is hafte {ctx.name} pe sabse zyada kaunsi service poochi gayi?{cur_hi} "
             f"Bas naam reply kar dijiye — main uska Google post + price queries ke liye ready WhatsApp reply bana dungi. 5 min ka kaam.")
    return body, "open_ended", "Weekly curious-ask — low-stakes question + reciprocity (post + reply draft).", [ctx.sal, ctx.name]


def t_competitor(ctx: Ctx):
    comp = ctx.p.get("competitor_name")
    mine = ctx.active_offers[0] if ctx.active_offers else None
    if comp:
        dist = f" {ctx.p.get('distance_km')} km away" if ctx.p.get("distance_km") else " nearby"
        their = f", leading with “{ctx.p['their_offer']}”" if ctx.p.get("their_offer") else ""
        opened = f" (opened {date_label(ctx.p.get('opened_date'))})" if ctx.p.get("opened_date") else ""
        head = f"{comp} opened{dist}{opened}{their}"
    else:
        head = f"a new {ctx.noun} listing has come up near you in {ctx.locality or ctx.city}"
    vs = f" Your live offer is “{mine}”." if mine else " You have no live offer right now."
    body = M(ctx,
             f"{ctx.sal}, heads-up — {head}.{vs} Your last 30 days: {ctx.perf_line()}. "
             f"New entrants usually pull price-shoppers in the first 60 days. Want me to draft a counter that leans on your reviews, not a price war? Reply YES.",
             f"{ctx.sal}, heads-up — {head}.{vs} Aapke last 30 din: {ctx.perf_line()}. "
             f"Naye players pehle 60 din price-shoppers kheenchte hain. Price war ke bina, reviews pe based counter draft kar doon? Reply YES.")
    return body, "binary_yes_no", "Competitor event — loss aversion + a non-price counter-move drafted for them.", [ctx.sal, head, mine or ""]


def t_review_theme(ctx: Ctx):
    theme = ctx.p.get("theme")
    occ = ctx.p.get("occurrences_30d")
    quote = ctx.p.get("common_quote")
    trend = ctx.p.get("trend")
    if not theme:
        neg = [r for r in (ctx.m.get("review_themes") or []) if r.get("sentiment") == "neg"] or (ctx.m.get("review_themes") or [])
        if neg:
            theme, occ, quote = neg[0].get("theme"), neg[0].get("occurrences_30d"), neg[0].get("common_quote")
    if not theme:
        body = M(ctx,
                 f"{ctx.sal}, a new pattern is showing up in {ctx.name}’s recent reviews. Want me to pull the reviews and draft polite public replies? Reply YES.",
                 f"{ctx.sal}, {ctx.name} ke recent reviews mein ek naya pattern dikh raha hai. Reviews nikaal ke polite public replies draft kar doon? Reply YES.")
        return body, "binary_yes_no", "Review theme trigger without detail — offer to pull reviews, no invented specifics.", [ctx.sal]
    q = f" — e.g. “{quote}”" if quote else ""
    tr = f" and it’s {trend}" if trend else ""
    body = M(ctx,
             f"{ctx.sal}, {occ or 'several'} reviews in the last 30 days mention {humanize(theme)}{q}{tr}. "
             f"Unanswered, this starts showing up in Google’s review summary. Want me to draft a polite public reply + one fix you can announce? Reply YES.",
             f"{ctx.sal}, pichhle 30 din mein {occ or 'kai'} reviews {humanize(theme)} ke baare mein hain{q}. "
             f"Jawab na diya toh yeh Google ke review summary mein dikhne lagta hai. Polite public reply + ek fix announce karne ka draft bana doon? Reply YES.")
    return body, "binary_yes_no", f"Emerging review theme '{theme}' with count + real quote; loss aversion + drafted response.", [ctx.sal, humanize(theme), str(occ)]


def t_milestone(ctx: Ctx):
    now_v, target = ctx.p.get("value_now"), ctx.p.get("milestone_value")
    metric = humanize(ctx.p.get("metric") or "")
    if now_v and target:
        gap = int(target) - int(now_v)
        unit = "reviews" if "review" in metric else (metric or "")
        head = (f"{ctx.name} is at {now_v} {unit} — just {gap} away from {target}"
                if gap > 0 else f"{ctx.name} just crossed {target} {unit}")
        ask = M(ctx, f"Want me to send a review-request WhatsApp to your recent happy {ctx.cust_noun} to close the gap this week? Reply YES.",
                f"Recent happy {ctx.cust_noun} ko review-request WhatsApp bhej doon taaki is hafte gap close ho jaye? Reply YES.")
    else:
        head = f"{ctx.name} hit {num(ctx.perf.get('views'))} profile views in the last 30 days" if ctx.perf.get("views") else f"{ctx.name} hit a new milestone"
        ask = M(ctx, "Want me to turn this into a ‘thank you’ Google post? Reply YES.",
                "Ise ek ‘thank you’ Google post mein badal doon? Reply YES.")
    body = f"{ctx.sal}, 🎉 {head}. {ask}"
    return body, "binary_yes_no", "Milestone — celebrate + a concrete action to lock it in (social proof).", [ctx.sal, head]


def t_planning(ctx: Ctx):
    topic = humanize(ctx.p.get("intent_topic") or "the idea you mentioned")
    words = [w for w in topic.split() if len(w) > 3]
    related = [o for o in ctx.active_offers + ctx.catalog if any(w.lower() in o.lower() for w in words)]
    base = related[0] if related else (ctx.active_offers[0] if ctx.active_offers else None)
    lines = [f"• Name: {topic.title()} — {ctx.name}, {ctx.locality}".rstrip(", ")]
    if base:
        lines.append(f"• Anchor price: built on your “{base}”")
    lines.append("• Booking: WhatsApp the day before; confirm slots in one reply")
    lines.append(f"• Launch: Google post + WhatsApp broadcast to your {ctx.cust_noun}")
    draft = "\n".join(lines)
    body = M(ctx,
             f"{ctx.sal}, here’s a starter version of the {topic} — edit anything:\n{draft}\n"
             f"Reply YES and I’ll publish it as a Google post + WhatsApp flyer, or tell me what to change.",
             f"{ctx.sal}, {topic} ka starter version yeh raha — kuch bhi edit kar sakte hain:\n{draft}\n"
             f"YES reply karein toh main Google post + WhatsApp flyer publish kar dungi, ya bataiye kya badalna hai.")
    return body, "binary_yes_no", "Merchant already asked for this — go straight to a concrete draft (action mode, no re-qualifying).", [ctx.sal, topic]


def t_ipl(ctx: Ctx):
    match, venue = ctx.p.get("match"), ctx.p.get("venue")
    t = time_label(ctx.p.get("match_time_iso"))
    item = ctx.digest_item("d_2026W17_ipl_window", kinds=("seasonal",))
    insight = ""
    if item and "ipl" in item.get("title", "").lower():
        insight = " " + first_sentence(item.get("summary", ""))
    offer = ctx.active_offers[0] if ctx.active_offers else ctx.best_offer(("match", "combo"))
    weekend = ctx.p.get("is_weeknight") is False
    tip = (f" So tonight, push “{offer}” as a delivery special rather than a dine-in promo." if weekend and offer
           else f" Good night to push “{offer}”." if offer else "")
    body = M(ctx,
             f"{ctx.sal}, {match} at {venue} tonight, {t}.{insight}{tip} Want me to draft the banner + an Insta story? Live in 10 min — reply YES.",
             f"{ctx.sal}, aaj raat {t} {venue} mein {match}.{insight}{tip} Banner + Insta story draft kar doon? 10 min mein live — YES reply karein.")
    return body, "binary_yes_no", "Same-day local event with a data-backed tip from the digest + merchant's own offer.", [ctx.sal, str(match), t]


def t_category_seasonal(ctx: Ctx):
    trends = ctx.p.get("trends") or []
    pretty = []
    for tr in trends[:4]:
        m = re.match(r"(.+?)_demand_([+-]?\d+)", str(tr))
        pretty.append(f"{m.group(1).replace('_', ' ')} {int(m.group(2)):+d}%" if m else humanize(tr))
    lst = ", ".join(pretty)
    body = M(ctx,
             f"{ctx.sal}, summer demand shift has started: {lst}. Pharmacies that move the rising items to counter-level first catch the impulse buys. "
             f"Want a shelf + restock checklist for {ctx.name}? Reply YES.",
             f"{ctx.sal}, summer demand shift shuru ho gaya hai: {lst}. Jo pharmacies badhte items counter pe pehle rakhti hain woh impulse buys pakadti hain. "
             f"{ctx.name} ke liye shelf + restock checklist bhej doon? Reply YES.")
    return body, "binary_yes_no", "Seasonal category shift with exact % moves; actionable checklist offer.", [ctx.sal, lst]


def t_supply_alert(ctx: Ctx):
    item = ctx.digest_item(ctx.p.get("alert_id"), kinds=("alert",))
    batches = ", ".join(ctx.p.get("affected_batches") or [])
    mol = ctx.p.get("molecule", "")
    mfr = ctx.p.get("manufacturer", "")
    why = first_sentence(item.get("summary", "")) if item else ""
    rx = ctx.agg.get("chronic_rx_count")
    rxl = f" You have {rx} chronic-Rx customers on file — some may have these batches." if rx else ""
    body = M(ctx,
             f"{ctx.sal}, urgent: recall on {mol} batches {batches} ({mfr}). {why}{rxl} "
             f"Want me to draft the customer WhatsApp + replacement-pickup note? Reply YES.",
             f"{ctx.sal}, urgent: {mol} ke batches {batches} ({mfr}) pe recall aaya hai. {why}{rxl} "
             f"Customers ke liye WhatsApp + replacement-pickup note draft kar doon? Reply YES.")
    return body, "binary_yes_no", "Urgency-5 safety/supply alert with exact batch numbers; drafted customer comms.", [ctx.sal, mol, batches]


def t_gbp_unverified(ctx: Ctx):
    path = humanize(ctx.p.get("verification_path") or "postcard or phone call")
    up = ctx.p.get("estimated_uplift_pct")
    upl = f" Verified listings typically see ~{pct(up)} more visibility." if up else ""
    body = M(ctx,
             f"{ctx.sal}, {ctx.name}’s Google profile is still unverified — so Google holds back your edits and ranking.{upl} "
             f"Verification is via {path}; I can walk you through it in 5 min. Start now? Reply YES.",
             f"{ctx.sal}, {ctx.name} ka Google profile abhi unverified hai — isliye Google aapke edits aur ranking rok ke rakhta hai.{upl} "
             f"Verification {path} se hota hai; main 5 min mein karwa dungi. Abhi shuru karein? Reply YES.")
    return body, "binary_yes_no", "Unverified GBP — clear loss + 5-min effort framing.", [ctx.sal, path]


def t_generic(ctx: Ctx):
    body = M(ctx,
             f"{ctx.sal}, quick update on {ctx.name}: last 30 days — {ctx.perf_line()}. "
             f"I have one idea to lift this week’s numbers. Want it? Reply YES.",
             f"{ctx.sal}, {ctx.name} ka quick update: pichhle 30 din — {ctx.perf_line()}. "
             f"Is hafte numbers badhane ka ek idea hai. Bhejun? Reply YES.")
    return body, "binary_yes_no", f"Trigger '{ctx.kind}' with limited payload — anchored on merchant's real 30-day numbers.", [ctx.sal, ctx.perf_line()]


# ---------------------------------------------------------------------------
# Customer-facing templates (sent as merchant_on_behalf)
# ---------------------------------------------------------------------------

def _cust_open(ctx: Ctx) -> str:
    who = f" {ctx.cust_name}" if ctx.cust_name else ""
    return f"{ctx.cust_greet}{who}, {ctx.name} here {EMOJI.get(ctx.slug, '')}".rstrip()


def _slot_cta(ctx: Ctx, slots: list) -> tuple[str, str]:
    labels = [s.get("label") for s in slots if s.get("label")][:2]
    if len(labels) == 2:
        return (C(ctx, f"We’ve kept 2 slots for you: {labels[0]} or {labels[1]}. Reply 1 or 2, or tell us a time that works.",
                  f"Aapke liye 2 slots rakhe hain: {labels[0]} ya {labels[1]}. 1 ya 2 reply karein, ya apna time bataiye."), "multi_choice_slot")
    if len(labels) == 1:
        return (C(ctx, f"We’ve kept {labels[0]} for you. Reply YES to confirm.",
                  f"Aapke liye {labels[0]} rakha hai. Confirm karne ke liye YES reply karein."), "binary_yes_no")
    return (C(ctx, "Reply YES and we’ll share this week’s open slots.",
              "YES reply karein, hum is hafte ke slots bhej denge."), "binary_yes_no")


def c_recall(ctx: Ctx):
    due = date_label(ctx.p.get("due_date"))
    last = date_label(ctx.p.get("last_service_date") or g(ctx.c, "relationship", "last_visit"))
    default_service = {"dentists": "check-up", "salons": "next appointment", "gyms": "next session",
                       "pharmacies": "refill", "restaurants": "next visit"}.get(ctx.slug, "next visit")
    service = humanize(ctx.p.get("service_due") or default_service).replace("6 month", "6-month")
    kw = service.split()[-1].lower() if ctx.p.get("service_due") else None
    offer = next((o for o in ctx.active_offers + ctx.catalog if kw and kw in o.lower()), None)
    price = f" {offer}." if offer and "@" in offer else ""
    cta, cta_type = _slot_cta(ctx, ctx.p.get("available_slots") or [])
    last_l = C(ctx, f"Your last visit was on {last}", f"Aapki last visit {last} ko thi") if last else ""
    due_l = C(ctx, f" — your {service} is due{' by ' + due if due else ''}.", f" — aapka {service} due hai{' (' + due + ' tak)' if due else ''}.")
    body = f"{_cust_open(ctx)}. {last_l}{due_l}{price} {cta}"
    return body, cta_type, "Customer recall — real last-visit date, catalog price, real slots; language pref honored.", [ctx.cust_name, ctx.name, service]


def c_appointment(ctx: Ctx):
    when = ctx.p.get("slot_label") or time_label(ctx.p.get("appointment_iso")) or ""
    w = f" at {when}" if when else ""
    body = C(ctx,
             f"{_cust_open(ctx)}. Reminder: your appointment at {ctx.name}{', ' + ctx.locality if ctx.locality else ''} is tomorrow{w}. "
             f"Reply 1 to confirm or 2 to reschedule.",
             f"{_cust_open(ctx)}. Reminder: kal{(' ' + when) if when else ''} {ctx.name}{', ' + ctx.locality if ctx.locality else ''} mein aapka appointment hai. "
             f"Confirm ke liye 1, reschedule ke liye 2 reply karein.")
    return body, "multi_choice_slot", "Appointment-tomorrow reminder; no invented time when payload lacks it.", [ctx.cust_name, ctx.name]


def c_lapsed(ctx: Ctx):
    last = date_label(g(ctx.c, "relationship", "last_visit"))
    visits = g(ctx.c, "relationship", "visits_total")
    days = ctx.p.get("days_since_last_visit")
    focus = humanize(ctx.p.get("previous_focus"))
    offer = ctx.active_offers[0] if ctx.active_offers else None
    since = (C(ctx, f"It’s been {days} days since your last visit", f"Aapki last visit ko {days} din ho gaye") if days
             else C(ctx, f"We last saw you on {last}", f"Aap last {last} ko aaye the") if last else C(ctx, "It’s been a while", "Kaafi time ho gaya"))
    f = C(ctx, f" — no pressure, happens to everyone. Your {focus} goal is still very doable", f" — koi baat nahi, sabke saath hota hai. Aapka {focus} goal abhi bhi possible hai") if focus else ""
    o = C(ctx, f" We have “{offer}” running right now.", f" Abhi “{offer}” chal raha hai.") if offer else ""
    ask = C(ctx, " Want us to hold a slot for you this week? Reply YES.", " Is hafte aapke liye slot rakh dein? YES reply karein.")
    body = f"{_cust_open(ctx)}. {since}{f}.{o}{ask}"
    return body, "binary_yes_no", f"Lapsed customer (visits={visits}) — warm, no-guilt win-back using a real live offer only.", [ctx.cust_name, ctx.name]


def c_refill(ctx: Ctx):
    mols = ctx.p.get("molecule_list") or []
    runs = date_label(ctx.p.get("stock_runs_out_iso"))
    senior = g(ctx.c, "identity", "senior_citizen")
    sen_offer = next((o for o in ctx.active_offers if "senior" in o.lower()), None)
    deliv = next((o for o in ctx.active_offers if "delivery" in o.lower()), None)
    extra = []
    if senior and sen_offer:
        extra.append(sen_offer)
    if deliv:
        extra.append(deliv)
    ex = (" " + " + ".join(extra) + ".") if extra else ""
    who = f"{ctx.cust_name}" if ctx.cust_name else C(ctx, "your", "aapki")
    if mols:
        med = ", ".join(mols)
        body = C(ctx,
                 f"{_cust_open(ctx)}. {who}’s monthly medicines ({med}) run out on {runs or 'soon'}. Same pack is ready.{ex} Reply CONFIRM to dispatch, or tell us if the dose changed.",
                 f"{_cust_open(ctx)}. {who} ki monthly medicines ({med}) {runs or 'jaldi'} ko khatam hongi. Same pack ready hai.{ex} Dispatch ke liye CONFIRM reply karein, ya dose badla ho toh bataiye.")
    else:
        body = C(ctx,
                 f"{_cust_open(ctx)}. Your regular refill / follow-up is due.{ex} Reply CONFIRM and we’ll keep it ready, or tell us if anything changed.",
                 f"{_cust_open(ctx)}. Aapka regular refill / follow-up due hai.{ex} CONFIRM reply karein, hum ready rakhenge — ya kuch badla ho toh bataiye.")
    return body, "binary_confirm_cancel", "Chronic refill — exact molecules + run-out date; applicable real offers only.", [ctx.cust_name, ctx.name]


def c_trial(ctx: Ctx):
    tdate = date_label(ctx.p.get("trial_date"))
    who = ctx.child_name or C(ctx, "you", "aap")
    cta, cta_type = _slot_cta(ctx, ctx.p.get("next_session_options") or [])
    t = C(ctx, f" after the trial on {tdate}", f" {tdate} ke trial ke baad") if tdate else ""
    body = C(ctx,
             f"{_cust_open(ctx)}. Hope {who} enjoyed it{t}! {cta}",
             f"{_cust_open(ctx)}. Umeed hai {who} ko{t} maza aaya! {cta}")
    return body, cta_type, "Trial follow-up — real trial date + real next session options.", [ctx.cust_name, ctx.name]


def c_wedding(ctx: Ctx):
    wd = date_label(ctx.p.get("wedding_date"))
    days = ctx.p.get("days_to_wedding")
    step = humanize(ctx.p.get("next_step_window_open"))
    offer = ctx.best_offer(("bridal", "skin", "facial"))
    o = f" {offer}." if offer else ""
    body = C(ctx,
             f"{_cust_open(ctx)} 💍 {days} days to your wedding ({wd}) — this is the right window to start the {step} before bridal bookings fill up.{o} Want us to block your first session next week? Reply YES.",
             f"{_cust_open(ctx)} 💍 Shaadi mein {days} din ({wd}) — {step} shuru karne ka sahi time abhi hai, bridal bookings bharne se pehle.{o} Next week pehla session block kar dein? YES reply karein.")
    return body, "binary_yes_no", "Bridal follow-up — wedding-date countdown + next program step.", [ctx.cust_name, wd]


MERCHANT_TEMPLATES = {
    "research_digest": t_research, "category_research_digest_release": t_research,
    "category_trend_movement": t_research,
    "regulation_change": t_regulation, "compliance": t_regulation,
    "cde_opportunity": t_cde,
    "perf_dip": t_perf_dip, "seasonal_perf_dip": t_seasonal_dip, "perf_spike": t_perf_spike,
    "renewal_due": t_renewal, "winback_eligible": t_winback,
    "dormant_with_vera": t_dormant, "festival_upcoming": t_festival,
    "curious_ask_due": t_curious, "scheduled_recurring": t_curious,
    "competitor_opened": t_competitor, "review_theme_emerged": t_review_theme,
    "milestone_reached": t_milestone, "active_planning_intent": t_planning,
    "ipl_match_today": t_ipl, "local_news_event": t_generic,
    "category_seasonal": t_category_seasonal, "supply_alert": t_supply_alert,
    "gbp_unverified": t_gbp_unverified,
}
CUSTOMER_TEMPLATES = {
    "recall_due": c_recall, "customer_lapsed_soft": c_lapsed, "customer_lapsed_hard": c_lapsed,
    "appointment_tomorrow": c_appointment, "chronic_refill_due": c_refill,
    "trial_followup": c_trial, "wedding_package_followup": c_wedding,
}


def template_compose(ctx: Ctx) -> dict:
    is_customer = bool(ctx.c) or ctx.t.get("scope") == "customer"
    if is_customer and ctx.c:
        fn = CUSTOMER_TEMPLATES.get(ctx.kind, c_lapsed)
        send_as = "merchant_on_behalf"
    else:
        fn = MERCHANT_TEMPLATES.get(ctx.kind, t_generic)
        send_as = "vera"
    body, cta, rationale, params = fn(ctx)
    body = re.sub(r"[ \t]+", " ", body).replace(" .", ".").replace("..", ".").strip()
    return {
        "body": body, "cta": cta, "send_as": send_as, "rationale": rationale,
        "template_name": f"{'merchant' if send_as == 'merchant_on_behalf' else 'vera'}_{ctx.kind}_v1",
        "template_params": [str(p) for p in params if p is not None],
    }


# ---------------------------------------------------------------------------
# LLM polish + validation
# ---------------------------------------------------------------------------

TONE_HINTS = {
    "dentists": "clinical peer (doctor-to-doctor). Technical terms OK. No hype.",
    "salons": "warm, practical, friendly.",
    "restaurants": "fellow operator, busy and practical.",
    "gyms": "energetic coach, disciplined.",
    "pharmacies": "trustworthy, precise neighbourhood pharmacist.",
}

SYSTEM_PROMPT = """You write ONE WhatsApp message for magicpin's merchant assistant "Vera".
You receive a DRAFT (already factually correct) plus the verified FACTS. Rewrite the draft so it is sharper and more compelling.

HARD RULES
1. Use ONLY facts given. Never invent numbers, names, prices, dates, studies, competitors or offers. Every number you write must already appear in DRAFT or FACTS.
2. Keep the key facts from the draft (numbers, names, dates, sources).
3. Open with the salutation given. No preamble ("Hope you're well"), no self-introduction.
4. Exactly ONE call-to-action and it is the LAST sentence. Keep the CTA type of the draft (e.g. "Reply YES", or slot choice 1/2).
5. No URLs. No hashtags. No ALL-CAPS hype. Avoid these taboo words: {taboos}.
6. Language: {language}.
7. Tone: {tone}
8. Length: 2-4 short sentences, under 480 characters (planning drafts with bullet lines may be longer).
9. Use service+price offers exactly as written (e.g. "Haircut @ ₹99"), never generic "% off" unless that is the actual offer.
10. Use at least one lever: loss aversion, social proof, curiosity, effort externalization ("I've drafted it"), or asking a question.

Return JSON only: {{"body": "...", "rationale": "one line: why this message now and which levers"}}"""


def _language_instruction(ctx: Ctx, customer_facing: bool) -> str:
    if customer_facing:
        return ("natural Hindi-English code-mix (Hinglish, Roman script)" if ctx.cust_hinglish
                else f"simple English (you may open with '{ctx.cust_greet}')")
    return ("natural Hindi-English code-mix (Hinglish in Roman script), mostly English with Hindi connectors"
            if ctx.merchant_hinglish else "English")


def _compact_facts(ctx: Ctx) -> dict:
    facts = {
        "merchant": {
            "name": ctx.name, "salutation": ctx.sal, "locality": ctx.locality, "city": ctx.city,
            "subscription": ctx.m.get("subscription"), "performance_30d": ctx.perf,
            "active_offers": ctx.active_offers, "signals": ctx.signals,
            "customer_aggregate": ctx.agg, "review_themes": ctx.m.get("review_themes"),
            "last_merchant_message": ctx.last_merchant_msg(),
        },
        "category": {"slug": ctx.slug, "peer_stats": ctx.peer, "offer_catalog": ctx.catalog[:8]},
        "trigger": {"kind": ctx.kind, "urgency": ctx.t.get("urgency"), "payload": ctx.p},
    }
    ids = {ctx.p.get(k) for k in ("top_item_id", "digest_item_id", "alert_id")} - {None}
    items = [d for d in (ctx.cat.get("digest") or []) if d.get("id") in ids]
    if items:
        facts["digest_item"] = items[0]
    if ctx.c:
        facts["customer"] = {
            "name_to_address": ctx.cust_name, "child": ctx.child_name,
            "language_pref": g(ctx.c, "identity", "language_pref"),
            "relationship": ctx.c.get("relationship"), "state": ctx.c.get("state"),
            "preferences": ctx.c.get("preferences"),
        }
    return facts


def _numbers(text: str) -> set[str]:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _number_universe(*objs) -> set[str]:
    out: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, bool) or o is None:
            return
        elif isinstance(o, (int, float)):
            out.add(str(o))
            out.add(str(int(o)) if float(o).is_integer() else str(o))
            if abs(o) <= 1.5:  # fraction -> percent forms
                out.add(f"{abs(o) * 100:.0f}")
                out.add(f"{abs(o) * 100:.1f}".rstrip("0").rstrip("."))
            out.add(str(abs(int(o))))
        else:
            out.update(_numbers(str(o)))

    for o in objs:
        walk(o)
    return out


def validate(body: str, ctx: Ctx, universe: set[str], previous: list[str], customer_facing: bool) -> Optional[str]:
    """Return None if OK, else the reason it failed."""
    if not body or len(body) < 40:
        return "too short"
    if len(body) > 1100:
        return "too long"
    low = body.lower()
    if "http" in low or "www." in low:
        return "url"
    for t in ctx.taboos:
        t = re.sub(r"\(.*?\)", "", t).strip()
        if t and t in low:
            return f"taboo:{t}"
    for n in _numbers(body):
        if n not in universe and not (n.isdigit() and int(n) <= 10):
            return f"unverified number {n}"
    anchor = (ctx.cust_name or ctx.name) if customer_facing else (ctx.owner or ctx.name)
    if anchor and anchor.split()[0].lower() not in low:
        return "missing name"
    if body.strip() in [p.strip() for p in previous]:
        return "repeat"
    return None


async def compose_async(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None,
                        previous_bodies: Optional[list[str]] = None, use_llm: bool = True,
                        llm_timeout: Optional[float] = None) -> dict:
    ctx = Ctx(category, merchant, trigger, customer)
    draft = template_compose(ctx)
    previous = previous_bodies or []
    result = dict(draft)
    result["suppression_key"] = trigger.get("suppression_key") or f"{ctx.kind}:{merchant.get('merchant_id')}"
    result["composer"] = "template"

    customer_facing = draft["send_as"] == "merchant_on_behalf"
    if use_llm and llm.enabled():
        system = SYSTEM_PROMPT.format(
            taboos=", ".join(ctx.taboos[:8]) or "none",
            language=_language_instruction(ctx, customer_facing),
            tone=(f"customer-facing, from the business itself, warm and short. {TONE_HINTS.get(ctx.slug, '')}"
                  if customer_facing else TONE_HINTS.get(ctx.slug, "peer, practical.")),
        )
        user = json.dumps({
            "send_as": draft["send_as"],
            "salutation": (f"{ctx.cust_greet} {ctx.cust_name}".strip() if customer_facing else ctx.sal),
            "why_now": f"trigger kind = {ctx.kind}",
            "DRAFT": draft["body"],
            "FACTS": _compact_facts(ctx),
            "do_not_repeat": previous[-3:],
        }, ensure_ascii=False, default=str)
        out = await llm.chat_json(system, user, max_tokens=450, timeout=llm_timeout)
        if out and isinstance(out.get("body"), str):
            body = out["body"].strip()
            universe = _number_universe(category, merchant, trigger, customer or {}) | _numbers(draft["body"])
            why = validate(body, ctx, universe, previous, customer_facing)
            if why is None:
                result["body"] = body
                result["composer"] = "llm"
                if out.get("rationale"):
                    result["rationale"] = f"{out['rationale']} (trigger={ctx.kind})"
            else:
                result["rationale"] += f" [LLM draft rejected: {why}; used verified template]"

    if result["body"] in previous:  # never send the same body twice
        result["body"] += M(ctx, " (Following up on this one.)", " (Is par follow-up.)")
    return result


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict:
    """Sync entry point matching challenge-brief §7.1."""
    r = asyncio.run(compose_async(category, merchant, trigger, customer))
    return {k: r[k] for k in ("body", "cta", "send_as", "suppression_key", "rationale")}
