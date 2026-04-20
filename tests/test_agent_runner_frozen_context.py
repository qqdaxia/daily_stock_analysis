# -*- coding: utf-8 -*-
"""Verify runner._execute_tools propagates ContextVar state into worker threads.

PR #1071 residual blocker (Phase B): the ReAct executor spawns tool handlers
through a ``ThreadPoolExecutor``; Python does not automatically inherit the
parent thread's ``contextvars`` in worker threads, so without
``contextvars.copy_context().run(...)`` the pipeline-frozen
``_agent_frozen_target_date`` ContextVar seen inside ``_analyze_with_agent``
would be invisible to the tool handlers, and cache-layer freshness checks
would silently fall back to wall-clock time — reproducing the #1066 bug.
"""

from __future__ import annotations

import threading
import unittest
from datetime import date
from typing import List

from src.agent.llm_adapter import ToolCall
from src.agent.runner import _execute_tools
from src.agent.tools.registry import ToolDefinition, ToolParameter, ToolRegistry
from src.services.stock_history_cache import (
    get_agent_frozen_target_date,
    reset_agent_frozen_target_date,
    set_agent_frozen_target_date,
)


class ExecuteToolsFrozenContextTestCase(unittest.TestCase):
    """``_execute_tools`` must surface the main thread's ContextVar to workers."""

    def _build_registry_capturing_contextvar(
        self,
        captured: List[tuple],
        thread_names: List[str],
    ) -> ToolRegistry:
        registry = ToolRegistry()

        def handler(stock_code: str) -> dict:
            captured.append((stock_code, get_agent_frozen_target_date()))
            thread_names.append(threading.current_thread().name)
            return {"code": stock_code, "ok": True}

        registry.register(
            ToolDefinition(
                name="capture_tool",
                description="capture contextvar for tests",
                parameters=[
                    ToolParameter(
                        name="stock_code",
                        type="string",
                        description="stock code",
                    )
                ],
                handler=handler,
                category="data",
            )
        )
        return registry

    def _make_tool_call(self, tc_id: str, code: str) -> ToolCall:
        return ToolCall(id=tc_id, name="capture_tool", arguments={"stock_code": code})

    def test_single_tool_submit_propagates_frozen_target_date(self) -> None:
        frozen = date(2026, 4, 16)
        captured: List[tuple] = []
        thread_names: List[str] = []
        registry = self._build_registry_capturing_contextvar(captured, thread_names)

        token = set_agent_frozen_target_date(frozen)
        try:
            _execute_tools(
                tool_calls=[self._make_tool_call("tc-1", "600519")],
                tool_registry=registry,
                step=1,
                progress_callback=None,
                tool_calls_log=[],
                # >0 forces the single-tool pool.submit path with copy_context
                tool_wait_timeout_seconds=5.0,
            )
        finally:
            reset_agent_frozen_target_date(token)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], ("600519", frozen))
        # Confirm the handler actually ran on a worker thread, not on the caller.
        self.assertNotEqual(thread_names[0], threading.current_thread().name)

    def test_multi_tool_submit_propagates_frozen_target_date(self) -> None:
        frozen = date(2026, 4, 16)
        captured: List[tuple] = []
        thread_names: List[str] = []
        registry = self._build_registry_capturing_contextvar(captured, thread_names)

        calls = [
            self._make_tool_call(f"tc-{idx}", code)
            for idx, code in enumerate(("600519", "000001", "AAPL", "hk00700"))
        ]

        token = set_agent_frozen_target_date(frozen)
        try:
            _execute_tools(
                tool_calls=calls,
                tool_registry=registry,
                step=2,
                progress_callback=None,
                tool_calls_log=[],
                tool_wait_timeout_seconds=5.0,
            )
        finally:
            reset_agent_frozen_target_date(token)

        self.assertEqual(len(captured), 4)
        for _code, observed_frozen in captured:
            self.assertEqual(observed_frozen, frozen)
        # All handlers ran on distinct worker threads.
        self.assertTrue(all(name != threading.current_thread().name for name in thread_names))

    def test_inline_single_tool_path_keeps_caller_contextvar(self) -> None:
        """Timeout-less single-tool path runs inline on the caller thread and
        thus naturally sees the caller's ContextVar.
        """
        frozen = date(2026, 4, 16)
        captured: List[tuple] = []
        thread_names: List[str] = []
        registry = self._build_registry_capturing_contextvar(captured, thread_names)

        token = set_agent_frozen_target_date(frozen)
        try:
            _execute_tools(
                tool_calls=[self._make_tool_call("tc-inline", "600519")],
                tool_registry=registry,
                step=3,
                progress_callback=None,
                tool_calls_log=[],
                tool_wait_timeout_seconds=None,
            )
        finally:
            reset_agent_frozen_target_date(token)

        self.assertEqual(captured, [("600519", frozen)])
        self.assertEqual(thread_names[0], threading.current_thread().name)


if __name__ == "__main__":
    unittest.main()
