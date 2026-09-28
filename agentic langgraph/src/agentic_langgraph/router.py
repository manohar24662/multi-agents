from typing import Literal
from langgraph.graph import END

from agentic_langgraph.state import SupervisorState


def router(state: SupervisorState) -> Literal["supervisor", "researcher", "analyst", "writer", "__end__"]:
    """Routes to next agent based on state"""

    next_agent = state.get("next_agent", "supervisor")

    if state.get("error") or next_agent == "end" or state.get("task_complete", False):
        return END

    if next_agent in ["supervisor", "researcher", "analyst", "writer"]:
        return next_agent

    return "supervisor"


def worker_router(state: SupervisorState) -> Literal["supervisor", "__end__"]:
    """Return workers to the supervisor unless they completed or failed."""
    if state.get("error") or state.get("task_complete") or state.get("next_agent") == "end":
        return END
    return "supervisor"