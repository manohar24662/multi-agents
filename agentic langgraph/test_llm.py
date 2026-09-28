
"""Tests for the primary and fallback Groq LLM clients without network calls."""

import os
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage

from agentic_langgraph.agents import llm


class LlmConfigurationTests(TestCase):
    def tearDown(self):
        llm.get_llm.cache_clear()
        llm.get_fallback_llm.cache_clear()

    @patch.object(llm, "ChatGroq")
    def test_primary_client_uses_groq_environment_variables(self, chat_groq):
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test-primary-key",
                "GROQ_MODEL": "test-primary-model",
                "GROQ_MAX_TOKENS": "900",
            },
            clear=True,
        ):
            client = llm.get_llm()

        self.assertIs(client, chat_groq.return_value)
        chat_groq.assert_called_once_with(
            model="test-primary-model",
            api_key="test-primary-key",
            temperature=0,
            max_tokens=900,
            max_retries=0,
        )

    @patch.object(llm, "ChatGroq")
    def test_fallback_reuses_primary_key_when_no_second_key_is_configured(self, chat_groq):
        with patch.dict(
            os.environ,
            {
                "GROQ_API_KEY": "test-primary-key",
                "GROQ_FALLBACK_MODEL": "test-fallback-model",
            },
            clear=True,
        ):
            client = llm.get_fallback_llm()

        self.assertIs(client, chat_groq.return_value)
        chat_groq.assert_called_once_with(
            model="test-fallback-model",
            api_key="test-primary-key",
            temperature=0,
            max_tokens=4096,
            max_retries=0,
        )

    def test_empty_primary_response_uses_the_fallback_model(self):
        primary = Mock()
        primary.invoke.return_value = AIMessage(content="")
        fallback = Mock()
        fallback.invoke.return_value = AIMessage(content="Fallback answer")

        with (
            patch.object(llm, "get_llm", return_value=primary),
            patch.object(llm, "get_fallback_llm", return_value=fallback),
            self.assertLogs(llm.logger, level="INFO"),
        ):
            result = llm.invoke_for_text("Prompt", agent_name="Planner", attempts=1)

        self.assertEqual(result, "Fallback answer")
        primary.invoke.assert_called_once_with("Prompt")
        fallback.invoke.assert_called_once_with("Prompt")
