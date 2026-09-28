"""Analysis worker node."""

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


def analyst_agent(state: SupervisorState) -> dict[str, Any]:
    """Analyze the task, using research when the supervisor assigned it first."""
    task = str(state.get("original_task") or state.get("current_task") or "").strip()
    research_context = str(state.get("research_data", "")).strip()
    if not research_context:
        research_context = "No separate research was assigned; analyze the task directly."
    rag_context, rag_state = get_rag_context_for_state(state, task)

    if rag_context:
        knowledge_context = rag_context
    else:
        knowledge_context = "No relevant information was found in the private RAG knowledge base."

    prompt = f"""As a data analyst, analyze this task: {task}

{recent_conversation_context(state)}

Private knowledge from RAG:
{knowledge_context}

Research context:
{research_context}

Use the private RAG information when it is relevant.
Use the research context when available.

Provide key patterns, implications, actionable insights, and risks."""

    try:
        analysis = invoke_for_text([HumanMessage(content=prompt)], agent_name="Analyst")
    except EmptyModelResponseError as error:
        logger.warning("Analyst received no usable model text: %s", error)
        return {
            "messages": [AIMessage(content="Analyst: Groq returned no usable analysis text.")],
            "next_agent": "end",
            "error": str(error),
        }
    except Exception as error:
        logger.exception("Analyst LLM call failed")
        return {
            "messages": [AIMessage(content="Analyst: Analysis could not be completed.")],
            "next_agent": "end",
            "error": f"Analyst request failed: {type(error).__name__}.",
        }

    completed_agents = [*state.get("completed_agents", []), "analyst"]
    return {
        **rag_state,
        "messages": [AIMessage(content=f"Analyst completed analysis for: {task}")],
        "analysis": analysis,
        "completed_agents": completed_agents,
        "next_agent": "supervisor",
    }
