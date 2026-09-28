"""Writing worker node."""

import logging
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agentic_langgraph.agents.llm import EmptyModelResponseError, invoke_for_text
from agentic_langgraph.conversation_memory import recent_conversation_context
from agentic_langgraph.RAG_pipline.services.retrieval.rag_context import (
    get_rag_context_for_state,
)
from agentic_langgraph.state import SupervisorState


logger = logging.getLogger(__name__)


def writer_agent(state: SupervisorState) -> dict[str, Any]:
    """Create a final response, with any available research and analysis as context."""
    task = str(state.get("original_task") or state.get("current_task") or "").strip()
    research_data = str(state.get("research_data", "")).strip()
    analysis = str(state.get("analysis", "")).strip()

    supporting_material = []
    if research_data:
        supporting_material.append(f"Research findings:\n{research_data}")
    if analysis:
        supporting_material.append(f"Analysis:\n{analysis}")
    context = "\n\n".join(supporting_material) or "No prior agent output was needed for this task."
    rag_context, rag_state = get_rag_context_for_state(state, task)
    knowledge_context = rag_context or "No relevant private RAG context was found."
    prompt = f"""Create the best final user-facing response for this task:
{task}

{recent_conversation_context(state)}

Private knowledge from RAG (use it when relevant and preserve source names):
{knowledge_context}

Available supporting material:
{context}

Use the private RAG information when it is relevant.
Use the supporting research and analysis when available.
Do not invent information that is not supported by the available context.

Match the format to the task. For a question, give a clear answer; for a drafting task,
produce the requested draft; for a report, use a structured report."""

    try:
        report = invoke_for_text([HumanMessage(content=prompt)], agent_name="Writer")
    except EmptyModelResponseError as error:
        logger.warning("Writer received no usable model text: %s", error)
        return {
            "messages": [AIMessage(content="Writer: Groq returned no usable final text.")],
            "next_agent": "end",
            "error": str(error),
        }
    except Exception as error:
        logger.exception("Writer LLM call failed")
        return {
            "messages": [AIMessage(content="Writer: The final response could not be generated.")],
            "next_agent": "end",
            "error": f"Writer request failed: {type(error).__name__}.",
        }

    final_report = f"""FINAL RESPONSE
{'=' * 50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Task: {task}
{'=' * 50}

{report}

{'=' * 50}
Response compiled by the Multi-Agent AI System powered by Groq
"""

    completed_agents = [*state.get("completed_agents", []), "writer"]
    return {
        **rag_state,
        # This is the only internal message retained as user-facing conversation
        # memory.  Other node status messages stay out of future model prompts.
        "messages": [AIMessage(content=report, name="writer")],
        "final_report": final_report,
        "completed_agents": completed_agents,
        "next_agent": "end",
        "task_complete": True,
    }
