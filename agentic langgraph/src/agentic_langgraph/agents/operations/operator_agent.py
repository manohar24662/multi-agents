"""Operation worker node.

This node is deliberately a safe placeholder until specific operation tools are
implemented and explicitly allowlisted in ``tools.py``.
"""

from typing import Any

from langchain_core.messages import AIMessage

from agentic_langgraph.state import SupervisorState


def operator_agent(state: SupervisorState) -> dict[str, Any]:
    """Record that no operation ran while the tool registry is still empty."""
    task = str(state.get("original_task") or state.get("current_task") or "").strip()
    previous_result = str(state.get("operation_result", "")).strip()
    operation_pass = state.get("completed_agents", []).count("operator") + 1
    completed_agents = [*state.get("completed_agents", []), "operator"]
    operation_result = (
        f"Operation pass {operation_pass}: No operation was executed because the operator "
        "has no configured tools yet. "
        "Add explicit, allowlisted tools in agents/operations/tools.py before enabling "
        "system actions."
    )
    combined_result = "\n\n".join(filter(None, [previous_result, operation_result]))

    return {
        "messages": [AIMessage(content=f"Operator: No configured tools are available for: {task}")],
        "operation_result": combined_result,
        "completed_agents": completed_agents,
        "next_agent": "supervisor",
    }
