"""
Tiny LLM client. Default = Groq (free, fast, OpenAI-compatible API).

Env vars:
  GROQ_API_KEY   your Groq key (https://console.groq.com/keys)
  LLM_MODEL      default "openai/gpt-oss-120b"
  LLM_BASE_URL   default Groq; any OpenAI-compatible endpoint works
  LLM_API_KEY    use instead of GROQ_API_KEY for other providers
  LLM_TIMEOUT    seconds per call (default 9)

If no key is set, or the call fails / times out / is rate-limited, callers
get None and fall back to the deterministic template composer. The bot
therefore never crashes and never misses the 30s budget because of the LLM.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Optional

import httpx

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "9"))


def api_key() -> str:
    return os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or ""


def enabled() -> bool:
    return bool(api_key())


_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=TIMEOUT)
    return _client


# Free-tier Groq allows only ~8,000 tokens/minute per model. To stay inside it:
#  - at most LLM_CONCURRENCY calls run at once (others fall back to templates fast),
#  - after a 429 the LLM is paused for a short cool-down instead of hammering it.
_sem: Optional[asyncio.Semaphore] = None
_cooldown_until = 0.0


def _semaphore() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(int(os.getenv("LLM_CONCURRENCY", "2")))
    return _sem


async def chat_json(system: str, user: str, max_tokens: int = 500,
                    timeout: Optional[float] = None) -> Optional[dict]:
    """Call the LLM and parse a JSON object from the reply. Returns None on any failure."""
    global _cooldown_until
    if not enabled() or time.time() < _cooldown_until:
        return None
    reasoning = "gpt-oss" in MODEL or "qwen3" in MODEL
    payload = {
        "model": MODEL,
        "temperature": 0,          # deterministic
        "seed": 7,
        "max_tokens": max_tokens + (700 if reasoning else 0),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if reasoning:
        # keep thinking short (speed + token budget); JSON is parsed from the text
        payload["reasoning_effort"] = os.getenv("LLM_REASONING", "low")
    else:
        payload["response_format"] = {"type": "json_object"}
    sem = _semaphore()
    try:  # wait briefly for a free slot; if still busy, use the template instead
        await asyncio.wait_for(sem.acquire(), timeout=float(os.getenv("LLM_QUEUE_WAIT_S", "4")))
    except asyncio.TimeoutError:
        return None
    if time.time() < _cooldown_until:  # a 429 arrived while we waited
        sem.release()
        return None
    try:
        try:
            r = await _get_client().post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key()}"},
                json=payload,
                timeout=timeout or TIMEOUT,
            )
        finally:
            sem.release()
        if r.status_code == 429:
            _cooldown_until = time.time() + float(os.getenv("LLM_COOLDOWN_S", "20"))
            print("[llm] rate-limited (429); cooling down, templates in use")
            return None
        if r.status_code != 200:
            print(f"[llm] HTTP {r.status_code}: {r.text[:200]}")
            return None
        text = r.json()["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{[\s\S]*\}", text)
        return json.loads(m.group()) if m else None
    except Exception as e:  # timeout, network, bad JSON
        print(f"[llm] error: {type(e).__name__}: {e}")
        return None
