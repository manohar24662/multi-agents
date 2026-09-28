"""Checkpointed conversation-memory tests with no live model or RAG calls."""

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from agentic_langgraph.agents import analysist_agent, planner_agent, research_agent, writer_agent
from agentic_langgraph.graph import create_graph
from agentic_langgraph.main import build_turn_input, create_sqlite_checkpointer


def _fake_model_response(prompt, *, agent_name: str) -> str:
    """A deterministic Writer that proves it received bounded prior context."""
    text = prompt[0].content if isinstance(prompt, list) else str(prompt)
    if agent_name == "Planner":
        return "Use the available agents and produce a direct final answer."
    if agent_name == "Researcher":
        return "No external research is needed for this conversation-memory test."
    if agent_name == "Analyst":
        return "Use the conversation context supplied with the current task."
    if agent_name == "Writer":
        if "What is my name?" in text:
            return (
                "Your name is Manohar."
                if "User: My name is Manohar." in text
                else "I do not know your name from this conversation."
            )
        if "What framework am I using?" in text:
            return (
                "You are using LangGraph."
                if "User: I am building an AI agent with LangGraph." in text
                else "I do not know which framework you are using."
            )
        if "My name is Manohar." in text:
            return "Nice to meet you, Manohar."
        if "I am building an AI agent with LangGraph." in text:
            return "That is a great LangGraph project."
    raise AssertionError(f"Unexpected {agent_name} prompt: {text}")


def _empty_rag_context(*_args, **_kwargs):
    return "", {"rag_context": "", "rag_context_ready": True}


def _run(graph, thread_id: str, task: str) -> dict:
    return graph.invoke(
        build_turn_input(task, max_iterations=10),
        config={"configurable": {"thread_id": thread_id}},
    )


class ConversationMemoryTests(TestCase):
    def test_thread_remembers_user_facts_and_framework_but_not_across_threads(self):
        """Exercises the three requested memory scenarios in one graph session."""
        with (
            patch.dict(
                os.environ,
                {"SUPERVISOR_USE_LLM": "false"},
            ),
            patch.object(planner_agent, "invoke_for_text", side_effect=_fake_model_response),
            patch.object(research_agent, "invoke_for_text", side_effect=_fake_model_response),
            patch.object(analysist_agent, "invoke_for_text", side_effect=_fake_model_response),
            patch.object(writer_agent, "invoke_for_text", side_effect=_fake_model_response),
            patch.object(research_agent, "get_rag_context_for_state", side_effect=_empty_rag_context),
            patch.object(analysist_agent, "get_rag_context_for_state", side_effect=_empty_rag_context),
            patch.object(writer_agent, "get_rag_context_for_state", side_effect=_empty_rag_context),
        ):
            graph = create_graph(checkpointer=InMemorySaver())

            _run(graph, "manohar-thread", "My name is Manohar.")
            name_answer = _run(graph, "manohar-thread", "What is my name?")
            self.assertIn("Your name is Manohar.", name_answer["final_report"])

            _run(graph, "framework-thread", "I am building an AI agent with LangGraph.")
            framework_answer = _run(graph, "framework-thread", "What framework am I using?")
            self.assertIn("You are using LangGraph.", framework_answer["final_report"])

            new_thread_answer = _run(graph, "new-thread", "What is my name?")
            self.assertIn("I do not know your name", new_thread_answer["final_report"])

    def test_sqlite_checkpointer_restores_a_thread_after_rebuilding_the_graph(self):
        """Verify the CLI's local SQLite checkpointer is durable across graph instances."""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"SUPERVISOR_USE_LLM": "false"},
            ),
            patch.object(planner_agent, "invoke_for_text", side_effect=_fake_model_response),
            patch.object(research_agent, "invoke_for_text", side_effect=_fake_model_response),
            patch.object(analysist_agent, "invoke_for_text", side_effect=_fake_model_response),
            patch.object(writer_agent, "invoke_for_text", side_effect=_fake_model_response),
            patch.object(research_agent, "get_rag_context_for_state", side_effect=_empty_rag_context),
            patch.object(analysist_agent, "get_rag_context_for_state", side_effect=_empty_rag_context),
            patch.object(writer_agent, "get_rag_context_for_state", side_effect=_empty_rag_context),
        ):
            database = Path(directory) / "checkpoints.sqlite"
            first_checkpointer = create_sqlite_checkpointer(database)
            first_graph = create_graph(checkpointer=first_checkpointer)
            _run(first_graph, "durable-thread", "My name is Manohar.")
            first_checkpointer.conn.close()

            second_checkpointer = create_sqlite_checkpointer(database)
            second_graph = create_graph(checkpointer=second_checkpointer)
            answer = _run(second_graph, "durable-thread", "What is my name?")
            second_checkpointer.conn.close()

        self.assertIn("Your name is Manohar.", answer["final_report"])
