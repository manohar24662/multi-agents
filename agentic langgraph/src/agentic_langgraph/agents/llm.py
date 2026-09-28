
"""Shared, lazy Groq client configuration for all workflow agents."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_FALLBACK_MODEL = "openai/gpt-oss-120b"


class EmptyModelResponseError(RuntimeError):
    """Raised when Groq completes a request without user-visible text."""


def _max_tokens() -> int:
    """Read a safe completion limit from the environment."""
    try:
        return max(256, int(os.getenv("GROQ_MAX_TOKENS", "4096")))
    except ValueError:
        logger.warning("GROQ_MAX_TOKENS is invalid; using 4096.")
        return 4096


def _response_text(response: Any) -> str:
    """Extract plain text from a LangChain response."""
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts).strip()
    return str(content).strip() if content else ""


@lru_cache
def get_llm() -> ChatGroq:
    """Build the primary Groq client only when an agent first needs it."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or export it in your shell.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        api_key=api_key,
        temperature=0,
        max_tokens=_max_tokens(),
        max_retries=0,
    )


@lru_cache
def get_fallback_llm() -> ChatGroq:
    """Build the fallback Groq client after the primary client fails.

    GROQ_FALLBACK_API_KEY is optional. If it is not set, the primary key is reused
    for the fallback model so a second key is not required.
    """
    api_key = os.getenv("GROQ_FALLBACK_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or export it in your shell.")
    return ChatGroq(
        model=os.getenv("GROQ_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
        api_key=api_key,
        temperature=0,
        max_tokens=_max_tokens(),
        max_retries=0,
    )


def _invoke_client(
    client: ChatGroq,
    prompt: Any,
    *,
    agent_name: str,
    client_name: str,
    attempts: int,
) -> tuple[str | None, str, Exception | None]:
    """Invoke one Groq client, retrying empty or transiently failed completions."""
    details = "no response"
    last_error: Exception | None = None

    for attempt in range(1, max(1, attempts) + 1):
        try:
            response = client.invoke(prompt)
        except Exception as error:
            last_error = error
            details = f"request failed: {type(error).__name__}"
            logger.warning(
                "%s %s failed (attempt %s/%s): %s",
                agent_name,
                client_name,
                attempt,
                max(1, attempts),
                type(error).__name__,
            )
            continue

        if text := _response_text(response):
            return text, "", None

        details = "completion contained no visible text"
        logger.warning(
            "%s %s returned no visible text (attempt %s/%s).",
            agent_name,
            client_name,
            attempt,
            max(1, attempts),
        )

    return None, details, last_error


def invoke_for_text(prompt: Any, *, agent_name: str, attempts: int = 1) -> str:
    """Return Groq text, falling back to the configured secondary Groq model."""
    text, primary_details, primary_error = _invoke_client(
        get_llm(),
        prompt,
        agent_name=agent_name,
        client_name="primary Groq model",
        attempts=attempts,
    )
    if text:
        return text

    logger.info("%s is trying the fallback Groq model.", agent_name)
    fallback_text, fallback_details, fallback_error = _invoke_client(
        get_fallback_llm(),
        prompt,
        agent_name=agent_name,
        client_name="fallback Groq model",
        attempts=1,
    )
    if fallback_text:
        return fallback_text

    if primary_error and fallback_error:
        raise RuntimeError(
            f"Both Groq models failed for {agent_name}: "
            f"primary={type(primary_error).__name__}, "
            f"fallback={type(fallback_error).__name__}."
        ) from fallback_error

    raise EmptyModelResponseError(
        f"{agent_name} received no visible text from Groq "
        f"(primary: {primary_details}; fallback: {fallback_details})."
    )
