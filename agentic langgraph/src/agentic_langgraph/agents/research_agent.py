"""Research worker node."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agentic_langgraph.agents.llm import EmptyModelResponseError, invoke_for_text
from agentic_langgraph.conversation_memory import recent_conversation_context
from agentic_langgraph.RAG_pipline.services.retrieval.rag_context import (
    get_rag_context_for_state,
)
from agentic_langgraph.state import SupervisorState


logger = logging.getLogger(__name__)


def researcher_agent(state: SupervisorState) -> dict[str, Any]:
    """Gather research for the original user task."""
    task = str(state.get("original_task") or state.get("current_task") or "").strip()
    if not task:
        return {
            "messages": [AIMessage(content="Researcher: No task was available.")],
            "next_agent": "end",
            "error": "The researcher did not receive a task.",
        }

    rag_context, rag_state = get_rag_context_for_state(state, task)
    rag_instruction = (
        f"""
Private document context (use it when relevant; preserve the source names and do not
claim information that is not present in it):
{rag_context}
"""
        if rag_context
        else ""
    )

    prompt = f"""As a research specialist, provide concise, reliable information about:
{task}

{recent_conversation_context(state)}

Include key facts, current developments, relevant data, and notable examples.
{rag_instruction}"""

    try:
        research_data = invoke_for_text([HumanMessage(content=prompt)], agent_name="Researcher")
    except EmptyModelResponseError as error:
        logger.warning("Researcher received no usable model text: %s", error)
        return {
            "messages": [AIMessage(content="Researcher: Groq returned no usable research text.")],
            "next_agent": "end",
            "error": str(error),
        }
    except Exception as error:
        logger.exception("Researcher LLM call failed")
        return {
            "messages": [AIMessage(content="Researcher: Research could not be completed.")],
            "next_agent": "end",
            "error": f"Researcher request failed: {type(error).__name__}.",
        }

    completed_agents = [*state.get("completed_agents", []), "researcher"]
    return {
        **rag_state,
        "messages": [AIMessage(content=f"Researcher completed research for: {task}")],
        "research_data": research_data,
        "completed_agents": completed_agents,
        "next_agent": "supervisor",
    }
