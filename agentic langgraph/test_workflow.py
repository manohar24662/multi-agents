
"""Workflow tests that run locally without calling Groq or Qdrant."""

import os
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agentic_langgraph.agents import (
    analysist_agent,
    planner_agent,
    research_agent,
    supervisor_agent,
    writer_agent,
)
from agentic_langgraph.graph import create_graph


def _initial_state(task: str) -> dict:
    return {
        "messages": [HumanMessage(content=task)],
        "original_task": task,
        "current_task": task,
        "completed_agents": [],
        "iteration_count": 0,
        "max_iterations": 10,
        "task_complete": False,
        "error": "",
    }


def _supervisor_with_choices(*choices: str) -> Mock:
    chain = Mock()
    chain.invoke.side_effect = [AIMessage(content=choice) for choice in choices]
    return chain


class WorkflowTests(TestCase):
    def test_researcher_includes_retrieved_private_context_in_the_llm_prompt(self):
        private_context = "Source: resume.pdf, page 1\nDurga Manohar is a software engineer."
        with (
            patch.object(
                research_agent,
                "get_rag_context_for_state",
                return_value=(
                    private_context,
                    {"rag_context": private_context, "rag_context_ready": True},
                ),
            ),
            patch.object(research_agent, "invoke_for_text", return_value="Research findings") as invoke,
        ):
            result = research_agent.researcher_agent(_initial_state("Tell me about Durga Manohar"))

        prompt = invoke.call_args.args[0][0].content
        self.assertIn(private_context, prompt)
        self.assertIn("preserve the source names", prompt)
        self.assertEqual(result["research_data"], "Research findings")
        self.assertTrue(result["rag_context_ready"])

    def test_research_analysis_writer_route_completes(self):
        supervisor_chain = _supervisor_with_choices("researcher", "analyst", "writer")
        with (
            patch.dict(
                os.environ,
                {"SUPERVISOR_USE_LLM": "true"},
            ),
            patch.object(planner_agent, "invoke_for_text", return_value="Research then analyze"),
            patch.object(research_agent, "invoke_for_text", return_value="Research findings"),
            patch.object(
                research_agent,
                "get_rag_context_for_state",
                return_value=("", {"rag_context": "", "rag_context_ready": True}),
            ),
            patch.object(analysist_agent, "invoke_for_text", return_value="Analysis findings"),
            patch.object(analysist_agent, "get_rag_context_for_state", return_value=("", {})),
            patch.object(writer_agent, "invoke_for_text", return_value="Final report body"),
            patch.object(writer_agent, "get_rag_context_for_state", return_value=("", {})),
            patch.object(supervisor_agent, "create_supervisor_chain", return_value=supervisor_chain),
        ):
            result = create_graph().invoke(_initial_state("Compare two deployment options"))

        self.assertIn("Research findings", result["research_data"])
        self.assertIn("Analysis findings", result["analysis"])
        self.assertIn("Final report body", result["final_report"])
        self.assertEqual(result["completed_agents"], ["researcher", "analyst", "writer"])
        self.assertTrue(result["task_complete"])
        self.assertFalse(result["error"])

    def test_direct_writing_task_runs_writer_only(self):
        supervisor_chain = _supervisor_with_choices("writer")
        with (
            patch.object(planner_agent, "invoke_for_text", return_value="Draft a welcome email"),
            patch.object(writer_agent, "invoke_for_text", return_value="Welcome email"),
            patch.object(writer_agent, "get_rag_context_for_state", return_value=("", {})),
            patch.object(supervisor_agent, "create_supervisor_chain", return_value=supervisor_chain),
        ):
            result = create_graph().invoke(_initial_state("Write a welcome email for new customers"))

        self.assertEqual(result["completed_agents"], ["writer"])
        self.assertFalse(result.get("research_data"))
        self.assertIn("Welcome email", result["final_report"])

    def test_supervisor_failure_uses_a_local_writer_route(self):
        unavailable_chain = Mock()
        unavailable_chain.invoke.side_effect = RuntimeError("network unavailable")
        with (
            patch.dict(
                os.environ,
                {"SUPERVISOR_USE_LLM": "true"},
            ),
            patch.object(planner_agent, "invoke_for_text", return_value="Draft a greeting"),
            patch.object(writer_agent, "invoke_for_text", return_value="Hello!"),
            patch.object(writer_agent, "get_rag_context_for_state", return_value=("", {})),
            patch.object(supervisor_agent, "create_supervisor_chain", return_value=unavailable_chain),
            self.assertLogs(supervisor_agent.logger, level="ERROR"),
        ):
            result = create_graph().invoke(_initial_state("Write a greeting"))

        self.assertEqual(result["completed_agents"], ["writer"])
        self.assertIn("Hello!", result["final_report"])

    def test_planner_failure_ends_without_a_fake_report(self):
        with (
            patch.object(planner_agent, "invoke_for_text", side_effect=RuntimeError("network unavailable")),
            self.assertLogs(planner_agent.logger, level="ERROR"),
        ):
            result = create_graph().invoke(_initial_state("Any task"))

        self.assertEqual(result["error"], "Planner request failed: RuntimeError.")
        self.assertFalse(result.get("final_report"))
        self.assertFalse(result["task_complete"])
