"""Tests for CLI thread selection without running LLM or RAG services."""

import tempfile
from pathlib import Path
from unittest import TestCase

from agentic_langgraph.main import resolve_thread_id


class CliThreadSelectionTests(TestCase):
    def test_default_cli_session_reuses_its_thread_id(self):
        with tempfile.TemporaryDirectory() as directory:
            thread_file = Path(directory) / "thread-id"

            first_thread, is_default = resolve_thread_id(None, thread_file=thread_file)
            second_thread, is_default_again = resolve_thread_id(None, thread_file=thread_file)

        self.assertTrue(is_default)
        self.assertTrue(is_default_again)
        self.assertEqual(first_thread, second_thread)

    def test_new_default_thread_and_explicit_thread_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            thread_file = Path(directory) / "thread-id"
            original_thread, _ = resolve_thread_id(None, thread_file=thread_file)
            replacement_thread, _ = resolve_thread_id(
                None, new_thread=True, thread_file=thread_file
            )
            explicit_thread, is_default = resolve_thread_id(
                "private-thread", thread_file=thread_file
            )

        self.assertNotEqual(original_thread, replacement_thread)
        self.assertEqual(explicit_thread, "private-thread")
        self.assertFalse(is_default)
