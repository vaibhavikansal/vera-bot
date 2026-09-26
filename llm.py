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

import json
import os
import re
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


async def chat_json(system: str, user: str, max_tokens: int = 500,
                    timeout: Optional[float] = None) -> Optional[dict]:
    """Call the LLM and parse a JSON object from the reply. Returns None on any failure."""
    if not enabled():
        return None
    payload = {
        "model": MODEL,
        "temperature": 0,          # deterministic
        "seed": 7,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if "gpt-oss" in MODEL or "qwen3" in MODEL:
        # reasoning models: keep thinking short (speed) and leave room for the answer
        payload["reasoning_effort"] = os.getenv("LLM_REASONING", "low")
        payload["max_tokens"] = max_tokens + 1500
    try:
        r = await _get_client().post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key()}"},
            json=payload,
            timeout=timeout or TIMEOUT,
        )
        if r.status_code != 200:
            print(f"[llm] HTTP {r.status_code}: {r.text[:200]}")
            return None
        text = r.json()["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{[\s\S]*\}", text)
        return json.loads(m.group()) if m else None
    except Exception as e:  # timeout, network, bad JSON
        print(f"[llm] error: {type(e).__name__}: {e}")
        return None
