"""Command-line entry point for the research, analysis, and writing workflow."""

import argparse
import logging
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver


DEFAULT_CHECKPOINT_DB = ".langgraph_checkpoints.sqlite"
DEFAULT_THREAD_FILE = ".langgraph_default_thread"


def create_sqlite_checkpointer(checkpoint_db: str | Path) -> SqliteSaver:
    """Create the local persistent checkpointer used by the CLI.

    The database is intentionally local and thread-scoped through LangGraph's
    ``configurable.thread_id``.  SQLite is a practical default for one local CLI
    process; a shared production deployment should use a database checkpointer such
    as LangGraph's Postgres saver instead.
    """
    database_path = Path(checkpoint_db).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer


def build_turn_input(task: str, max_iterations: int) -> dict[str, Any]:
    """Build state for one new user turn without carrying workflow output forward.

    ``messages`` is merged by ``MessagesState``/``add_messages``.  The other keys
    are deliberately reset so research, analysis, RAG context, and completion state
    from the previous task cannot be mistaken for work on the new task.
    """
    return {
        "messages": [HumanMessage(content=task)],
        "original_task": task,
        "current_task": task,
        "plan": "",
        "next_agent": "supervisor",
        "research_data": "",
        "analysis": "",
        "rag_context": "",
        "rag_context_ready": False,
        "final_report": "",
        "completed_agents": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "task_complete": False,
        "error": "",
    }


def resolve_thread_id(
    supplied_thread_id: str | None,
    *,
    new_thread: bool = False,
    thread_file: str | Path = DEFAULT_THREAD_FILE,
) -> tuple[str, bool]:
    """Return the selected thread and whether it is the local default session.

    A CLI process cannot retain an in-memory variable after it exits.  The small
    thread-ID file therefore makes ordinary one-shot commands behave like one
    continuing local conversation.  Callers who need isolation can either provide
    an explicit ``--thread-id`` or begin a fresh default session with
    ``--new-thread``.
    """
    if supplied_thread_id:
        return supplied_thread_id.strip(), False

    path = Path(thread_file)
    if not new_thread and path.is_file():
        existing_thread_id = path.read_text(encoding="utf-8").strip()
        if existing_thread_id:
            return existing_thread_id, True

    thread_id = str(uuid.uuid4())
    path.write_text(f"{thread_id}\n", encoding="utf-8")
    return thread_id, True


def run_turn(graph: Any, task: str, max_iterations: int, config: dict[str, Any]) -> int:
    """Run one user turn and print its final response."""
    result = graph.invoke(build_turn_input(task, max_iterations), config=config)

    if error := result.get("error"):
        print(f"Workflow failed: {error}")
        return 1

    print(result.get("final_report", "Workflow finished without a final report."))
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run the agentic LangGraph workflow.")
    parser.add_argument(
        "task",
        nargs="?",
        help="The topic or question the agents should work on.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum supervisor decisions before the workflow stops (default: 10).",
    )
    parser.add_argument(
        "--thread-id",
        help=(
            "Conversation identifier. Reuse it to recall the same conversation "
            "across commands."
        ),
    )
    parser.add_argument(
        "--new-thread",
        action="store_true",
        help="Start a new default local conversation instead of resuming the previous one.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep one thread open and accept multiple user messages until /exit.",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=DEFAULT_CHECKPOINT_DB,
        help=(
            "SQLite checkpoint database path "
            f"(default: {DEFAULT_CHECKPOINT_DB})."
        ),
    )
    args = parser.parse_args()

    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if not args.task and not args.interactive:
        parser.error("provide a task or use --interactive")
    if args.thread_id and args.new_thread:
        parser.error("use either --thread-id or --new-thread, not both")

    from agentic_langgraph.graph import create_graph

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    thread_id, is_default_thread = resolve_thread_id(args.thread_id, new_thread=args.new_thread)
    checkpointer = create_sqlite_checkpointer(args.checkpoint_db)
    graph = create_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    thread_label = "default local session" if is_default_thread else "explicit thread"
    print(f"Conversation thread: {thread_id} ({thread_label})")
    exit_code = 0
    if args.task:
        exit_code = run_turn(graph, args.task, args.max_iterations, config)
        if not args.interactive:
            return exit_code

    print("Interactive conversation started. Type /exit to finish.")
    try:
        while True:
            task = input("You> ").strip()
            if task.lower() in {"/exit", "/quit", "exit", "quit"}:
                break
            if not task:
                continue
            exit_code = run_turn(graph, task, args.max_iterations, config) or exit_code
    except (EOFError, KeyboardInterrupt):
        print()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
