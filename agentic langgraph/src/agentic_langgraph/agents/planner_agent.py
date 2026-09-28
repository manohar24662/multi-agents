"""Planner node that turns the user's request into a short working plan."""

import logging
from typing import Any

from langchain_core.messages import AIMessage

from agentic_langgraph.agents.llm import EmptyModelResponseError, invoke_for_text
from agentic_langgraph.conversation_memory import recent_conversation_context
from agentic_langgraph.state import SupervisorState


logger = logging.getLogger(__name__)


def _get_task(state: SupervisorState) -> str:
    """Use the persisted task, falling back to the first invocation message."""
    if task := str(state.get("original_task", "")).strip():
        return task

    messages = state.get("messages", [])
    if not messages:
        return ""

    content = messages[-1].content
    return content.strip() if isinstance(content, str) else str(content).strip()


def planner_agent(state: SupervisorState) -> dict[str, Any]:
    """Break the user's task into an ordered plan for the available agents."""
    task = _get_task(state)
    if not task:
        return {
            "messages": [AIMessage(content="Planner: No task was provided.")],
            "next_agent": "end",
            "error": "No task was provided to the workflow.",
        }

    planning_prompt = f"""Break down this task into a concise work plan for a
researcher, analyst, and writer.

Task: {task}

{recent_conversation_context(state)}

Explain which work may be needed: current facts or sources (researcher), reasoning,
comparisons, or recommendations (analyst), and a final user-facing answer (writer).
The supervisor will decide the actual assignment order. Return a concise numbered plan."""

    try:
        plan = invoke_for_text(planning_prompt, agent_name="Planner")
    except EmptyModelResponseError as error:
        logger.warning("Planner received no usable model text: %s", error)
        return {
            "messages": [AIMessage(content="Planner: Groq returned no usable plan text.")],
            "original_task": task,
            "current_task": task,
            "next_agent": "end",
            "error": str(error),
        }
    except Exception as error:
        logger.exception("Planner LLM call failed")
        return {
            "messages": [AIMessage(content="Planner: I could not create a plan.")],
            "original_task": task,
            "current_task": task,
            "next_agent": "end",
            "error": f"Planner request failed: {type(error).__name__}.",
        }

    return {
        "messages": [AIMessage(content=f"Planner created this plan:\n{plan}")],
        "original_task": task,
        "current_task": task,
        "plan": plan,
        "next_agent": "supervisor",
    }
