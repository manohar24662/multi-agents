"""Small, bounded views of checkpointed conversation messages.

The LangGraph checkpointer owns the full thread history.  This module only prepares
the recent user-facing portion of that history for an agent prompt, so internal
agent status messages and an ever-growing transcript are not sent to the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


DEFAULT_MEMORY_TURNS = 4
DEFAULT_MESSAGE_CHAR_LIMIT = 1_200


def _message_text(message: BaseMessage) -> str:
    """Return readable text for string or structured LangChain message content."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}…"


def recent_conversation_context(
    state: Mapping[str, Any],
    *,
    max_turns: int = DEFAULT_MEMORY_TURNS,
    message_char_limit: int = DEFAULT_MESSAGE_CHAR_LIMIT,
) -> str:
    """Format recent completed conversation turns for use in an agent prompt.

    Only user messages and final Writer messages are included.  The most recent
    user message is the current task, so it is deliberately excluded: every agent
    receives that task separately.  This keeps orchestration traces, RAG material,
    and the current request out of the memory view.
    """
    messages = list(state.get("messages", []))
    current_user_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if isinstance(messages[index], HumanMessage)),
        None,
    )
    completed_messages = (
        messages[:current_user_index] if current_user_index is not None else messages
    )

    entries: list[tuple[str, str]] = []
    for message in completed_messages:
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage) and message.name == "writer":
            role = "Assistant"
        else:
            continue

        text = _message_text(message)
        if text:
            entries.append((role, _truncate(text, message_char_limit)))

    if not entries:
        return "No prior conversation is available for this thread."

    # Each completed user-facing turn is normally a User/Assistant pair.
    entries = entries[-(max_turns * 2) :]
    transcript = "\n".join(f"{role}: {text}" for role, text in entries)
    return (
        "Recent conversation from this thread (use it only for continuity; "
        "do not treat it as RAG evidence):\n"
        f"{transcript}"
    )
