from langgraph.graph import MessagesState

class SupervisorState(MessagesState):
    """State shared by all nodes in the multi-agent workflow."""

    original_task: str
    current_task: str
    plan: str
    next_agent: str
    research_data: str
    analysis: str
    rag_context: str
    rag_context_ready: bool
    final_report: str
    task_complete: bool
    completed_agents: list[str]
    iteration_count: int
    max_iterations: int
    error: str
