"""
ARIES — backend/services/llm_provider.py
Provider-agnostic LLM call layer.
Swap between Anthropic and Gemini via the LLM_PROVIDER env variable.

Environment variables:
    LLM_PROVIDER      "anthropic" (default) | "gemini"
    ANTHROPIC_API_KEY  Required when LLM_PROVIDER=anthropic
    ANTHROPIC_MODEL    Optional Anthropic model override
    GEMINI_API_KEY     Required when LLM_PROVIDER=gemini
    GEMINI_MODEL       Optional Gemini model override
"""

import logging
import os

import httpx

logger = logging.getLogger("aries.llm_provider")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Anthropic
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Gemini
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_TOKENS = 256
REQUEST_TIMEOUT = 20.0


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

async def _call_anthropic(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            ANTHROPIC_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
        )

    if response.status_code != 200:
        logger.error("Anthropic error: status=%s body=%s", response.status_code, response.text[:300])
        raise RuntimeError(f"Anthropic API error (status {response.status_code}).")

    data = response.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"].strip()

    raise RuntimeError("Anthropic API returned an empty response.")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

async def _call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    logger.debug(
        "Gemini request config: provider=%s model=%s api_key_exists=%s",
        "gemini",
        model,
        bool(api_key),
    )

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    api_url = f"{GEMINI_API_BASE_URL}/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": MAX_TOKENS},
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            api_url,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
        )

    logger.debug("Gemini response status_code=%s", response.status_code)

    if response.status_code != 200:
        logger.error("Gemini error: status=%s body=%s", response.status_code, response.text[:500])
        raise RuntimeError(f"Gemini API error (status {response.status_code}).")

    data = response.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts).strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError("Gemini API returned an unexpected response structure.") from exc

    if not text:
        raise RuntimeError("Gemini API returned an empty response.")

    return text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def call_llm(prompt: str) -> str:
    """
    Calls the configured LLM provider and returns the raw text response.
    Switch providers by setting LLM_PROVIDER=anthropic|gemini in your environment.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    logger.debug("call_llm: provider=%s", provider)

    if provider == "anthropic":
        return await _call_anthropic(prompt)
    elif provider == "gemini":
        return await _call_gemini(prompt)
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            "Set LLM_PROVIDER to 'anthropic' or 'gemini'."
        )
