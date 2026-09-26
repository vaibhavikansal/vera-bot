# Vera bot — magicpin AI Challenge

**Name:** Vaibhavi Kansal  **Model:** Groq `llama-3.3-70b-versatile` at temperature 0, with a template-only fallback.

## Approach

1. **Router by trigger kind.** Each of the 26 trigger kinds in the dataset has its own small composer function. There are 19 merchant-facing and 7 customer-facing ones.
2. **Fact-grounded draft first.** Each function builds a complete message using only fields that exist in the 4 contexts. That includes the digest item by id, the merchant's real 30-day numbers, peer stats, active offers, real slots and real dates. When a trigger payload is only a placeholder, the draft falls back to the merchant's real numbers. It never invents a festival, a competitor name or a time.
3. **LLM polish.** The LLM gets the draft plus a compact fact sheet and rewrites it for voice and compulsion. The prompt carries category tone, taboo words, language and a single-CTA rule.
4. **Validator.** It rejects any LLM output that has a URL, a taboo word, a number not found in the contexts, a missing name, or a repeated body. In that case the verified draft is sent instead. This keeps specificity high and hallucination at zero.
5. **Language.** Merchants with `hi` in `languages` get natural Hinglish. Customers follow `language_pref`, with regional greetings for ta/kn/te/mr. The reply language is re-detected on every turn.

## Conversation handling (`conversation_handlers.py`)

The bot checks rules first, then uses the LLM, so the replay tests are fast and predictable.

| Merchant says | Bot does |
|---|---|
| "stop" / "not interested" | `end` and blocks future ticks to that contact |
| WhatsApp-Business auto-reply (pattern match or repeated verbatim; counted per merchant across conversations) | 1st: one short owner nudge. 2nd: `wait` 24h. 3rd: `end` |
| Abuse without opt-out | One calm apology plus a STOP path. A 2nd time: `end` |
| "busy / later / kal" | `wait` 1h, or 24h for "tomorrow" |
| Off-topic (GST, loan…) | Polite decline, then back to the original topic |
| "yes / let's do it / judna hai / 2" | **Action mode**: delivers the draft or booking right away with one CONFIRM. No qualifying questions |
| A question | Grounded LLM answer. If the data is missing, it says so honestly |

## Operational safety

- Parallel composition per tick with an 11s budget. Anything slower falls back to the instant template, so ticks never time out. This was tested with a hanging LLM: 20 actions came back in 4s.
- At most one merchant-facing message per merchant per tick, highest urgency first. Suppression keys are never re-sent. Snoozed and opted-out contacts are skipped.
- Context versions are idempotent: a stale or same version returns 409 and a higher version replaces the old one atomically. `/v1/teardown` wipes all state.

## Tradeoffs

- Templates cap the downside but can read slightly formulaic when the LLM is unavailable.
- The number validator is strict. It sometimes rejects a good LLM rewrite that rounds a figure, and in that case the template is used.
- State is in memory, which the brief allows. A restart during the test would lose conversations.

## What extra context would help most

- Real open appointment slots and business hours for each merchant. Many customer triggers have no time data.
- Review count and rating on MerchantContext, needed for milestone and social-proof claims.
- Locality-level peer stats, for example "3 dentists in Lajpat Nagar did X", to unlock the social-proof lever safely.
- Which WhatsApp templates are already approved, so `template_name` maps to real ones.
