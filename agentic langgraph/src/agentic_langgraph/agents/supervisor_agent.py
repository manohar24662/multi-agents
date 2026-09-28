"""Supervisor node that assigns the next agent from the user's task and plan."""

import logging
import os
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from agentic_langgraph.agents.llm import get_llm
from agentic_langgraph.state import SupervisorState


logger = logging.getLogger(__name__)

AGENT_NAMES = ("researcher", "analyst", "writer")

def _llm_routing_enabled() -> bool:
    """Optionally use Groq to choose the next worker before local routing."""
    value = os.getenv("SUPERVISOR_USE_LLM", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def create_supervisor_chain():
    """Create the LLM chain that selects the next agent for this task."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a task-routing supervisor. Choose the single best next agent.

Agents:
- researcher: current facts, sources, background, or evidence
- analyst: comparisons, reasoning, risks, recommendations, or data interpretation
- writer: direct drafting, rewriting, summarising, or the final user-facing answer

Use the user's query and planner's plan. Choose only an available agent and never
repeat a completed one. A simple writing or translation request can go straight to
writer. An analysis can go directly to analyst when research is unnecessary. Reply
with exactly one word: researcher, analyst, writer, or done. Reply done only after
a final report exists.""",
            ),
            (
                "human",
                """User query: {task}
Planner plan: {plan}
Completed agents: {completed_agents}
Available agents: {available_agents}
Has research: {has_research}
Has analysis: {has_analysis}
Has final report: {has_report}""",
            ),
        ]
    )
    return prompt | get_llm()


def _fallback_agent(task: str, available_agents: list[str]) -> str:
    """Choose a sensible route if the supervisor model is unavailable or invalid."""
    task_lower = task.lower()
    writing_terms = ("write", "rewrite", "draft", "translate", "email", "message", "poem", "story")
    analysis_terms = (
        "analyze",
        "analyse",
        "compare",
        "comparison",
        "pros and cons",
        "risk",
        "strategy",
        "recommend",
        "decision",
    )

    if "writer" in available_agents and any(term in task_lower for term in writing_terms):
        return "writer"
    if "analyst" in available_agents and any(term in task_lower for term in analysis_terms):
        return "analyst"
    if "researcher" in available_agents:
        return "researcher"
    if "analyst" in available_agents:
        return "analyst"
    if "writer" in available_agents:
        return "writer"
    return "end"


def _valid_choice(content: object, available_agents: list[str], has_report: bool) -> str | None:
    """Validate the supervisor's one-word response before routing the graph."""
    choice = str(content).strip().lower().strip(".`'\"")
    if choice == "done":
        return "end" if has_report else None
    return choice if choice in available_agents else None


def supervisor_agent(state: SupervisorState) -> dict[str, Any]:
    """Ask the supervisor which currently available agent best matches the task."""
    task = str(state.get("original_task") or state.get("current_task") or "").strip()
    iteration_count = int(state.get("iteration_count", 0))
    max_iterations = int(state.get("max_iterations", 10))

    if state.get("error"):
        return {
            "messages": [AIMessage(content="Supervisor: Workflow stopped because a step failed.")],
            "next_agent": "end",
        }

    if state.get("final_report"):
        return {
            "messages": [AIMessage(content="Supervisor: The final report is ready.")],
            "next_agent": "end",
        }

    if iteration_count >= max_iterations:
        logger.warning("Maximum iteration count reached: %s", max_iterations)
        return {
            "messages": [AIMessage(content="Supervisor: Maximum iterations reached.")],
            "next_agent": "end",
            "iteration_count": iteration_count + 1,
            "error": f"Workflow stopped after {max_iterations} supervisor decisions.",
        }

    completed_agents = [
        agent for agent in state.get("completed_agents", []) if agent in AGENT_NAMES
    ]
    available_agents = [agent for agent in AGENT_NAMES if agent not in completed_agents]
    if not available_agents:
        return {
            "messages": [AIMessage(content="Supervisor: No agents remain to complete the task.")],
            "next_agent": "end",
            "error": "The workflow ended before the writer produced a final report.",
        }

    has_research = bool(state.get("research_data"))
    has_analysis = bool(state.get("analysis"))
    has_report = bool(state.get("final_report"))
    next_agent: str | None = None

    # A single available worker is a deterministic choice, so do not make an API call.
    if len(available_agents) == 1:
        next_agent = available_agents[0]

    if next_agent is None and _llm_routing_enabled():
        try:
            decision = create_supervisor_chain().invoke(
                {
                    "task": task,
                    "plan": state.get("plan", ""),
                    "completed_agents": ", ".join(completed_agents) or "none",
                    "available_agents": ", ".join(available_agents),
                    "has_research": has_research,
                    "has_analysis": has_analysis,
                    "has_report": has_report,
                }
            )
            next_agent = _valid_choice(decision.content, available_agents, has_report)
            if next_agent is None:
                logger.warning("Supervisor returned an unavailable or invalid route: %r", decision.content)
        except Exception:
            logger.exception("Supervisor decision failed; using local routing")

    next_agent = next_agent or _fallback_agent(task, available_agents)
    return {
        "messages": [AIMessage(content=f"Supervisor: Assigning the task to {next_agent}.")],
        "next_agent": next_agent,
        "iteration_count": iteration_count + 1,
    }
