import logging
from typing import Any

from langgraph.graph import END, StateGraph

from agentic_langgraph.agents.analysist_agent import analyst_agent
from agentic_langgraph.agents.planner_agent import planner_agent
from agentic_langgraph.agents.research_agent import researcher_agent
from agentic_langgraph.agents.supervisor_agent import supervisor_agent
from agentic_langgraph.agents.writer_agent import writer_agent
from agentic_langgraph.router import router, worker_router
from agentic_langgraph.state import SupervisorState

logger = logging.getLogger(__name__)


def create_graph(*, checkpointer: Any | None = None):
    """Build and compile the multi-agent supervisor graph.

    A caller can supply a LangGraph checkpointer to retain state per ``thread_id``.
    Leaving it unset preserves the lightweight, stateless import-time graph used by
    library consumers and existing tests.
    """

    graph = StateGraph(SupervisorState)

    # Register nodes
    graph.add_node("planner", planner_agent)
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("writer", writer_agent)

    # Entry point: planner runs first, then hands off to supervisor
    graph.set_entry_point("planner")
    graph.add_edge("planner", "supervisor")

    # Supervisor decides who goes next, dynamically
    graph.add_conditional_edges(
        "supervisor",
        router,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "supervisor": "supervisor",
            END: END
        }
    )

    # Successful intermediate work returns to the supervisor; terminal states stop.
    for worker in ("researcher", "analyst", "writer"):
        graph.add_conditional_edges(
            worker,
            worker_router,
            {"supervisor": "supervisor", END: END},
        )

    compiled_graph = graph.compile(checkpointer=checkpointer)
    logger.info("Multi-agent graph compiled successfully.")

    return compiled_graph


# A reusable graph for applications that import this module.
app = create_graph()
