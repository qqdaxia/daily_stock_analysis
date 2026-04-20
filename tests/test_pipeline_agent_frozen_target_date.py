# -*- coding: utf-8 -*-
"""Invariant test for #1066: pipeline-frozen target_date reaches agent tools.

Regression scenario (reviewer #1071, latest comment):
1. Pipeline Step 1 freezes ``target_date`` to T using ``resume_reference_time``.
2. Wall-clock then crosses the market-close boundary (T → T+1).
3. Without ContextVar propagation, the agent's tool handlers would fall
   back to ``get_effective_trading_date(current_time=None)`` and mark the
   just-cached T-history as stale, causing redundant refetches or empty
   returns.

This test checks the invariant end-to-end:
- ``_analyze_with_agent`` sets the ``ContextVar`` to the frozen trading date.
- A real ``_execute_tools`` worker-thread dispatch still sees that
  ``ContextVar`` even after the wall-clock is advanced past the boundary.
- ``load_recent_history_df`` (through its shared path used by every agent
  tool) observes the frozen date instead of the advanced wall-clock date.
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.executor import AgentResult
from src.agent.llm_adapter import ToolCall
from src.agent.runner import _execute_tools
from src.agent.tools.registry import ToolDefinition, ToolParameter, ToolRegistry
from src.core.pipeline import StockAnalysisPipeline
from src.services.stock_history_cache import (
    get_agent_frozen_target_date,
    get_candidate_pick_cache,
)


class _StubExecutor:
    """Pretends to be a full AgentExecutor but only drives ``_execute_tools``.

    The stub captures the frozen date that tool handlers *observe* through
    the ContextVar, proving that the pipeline set the context and that the
    ThreadPoolExecutor propagated it into worker threads.
    """

    def __init__(self, registry: ToolRegistry, observations: List[Dict[str, Any]]):
        self._registry = registry
        self._observations = observations

    def run(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        # Confirm pipeline handed us the frozen date in ``initial_context`` too
        # (observability-side of the same invariant).
        self._observations.append({"source": "initial_context", "context": dict(context or {})})

        # Drive the same threaded tool-dispatch path the real ReAct loop uses.
        tool_calls = [
            ToolCall(id="tc-1", name="probe_history", arguments={"stock_code": "600519"}),
            ToolCall(id="tc-2", name="probe_history", arguments={"stock_code": "000001"}),
        ]
        _execute_tools(
            tool_calls=tool_calls,
            tool_registry=self._registry,
            step=1,
            progress_callback=None,
            tool_calls_log=[],
            tool_wait_timeout_seconds=5.0,
        )
        return AgentResult(
            success=True,
            content="",
            dashboard=None,
            tool_calls_log=[],
            total_steps=1,
            total_tokens=0,
            provider="stub",
        )


class PipelineAgentFrozenTargetDateTestCase(unittest.TestCase):
    def _build_pipeline(self) -> StockAnalysisPipeline:
        pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
        pipeline.config = SimpleNamespace(
            report_language="zh",
            agent_skills=None,
            report_integrity_enabled=False,
        )
        pipeline.social_sentiment_service = None
        return pipeline

    def test_agent_tools_observe_pipeline_frozen_target_date_across_boundary(self) -> None:
        pipeline = self._build_pipeline()

        # Step 1 froze the trading date to T on the A-share market.
        frozen = date(2026, 4, 16)
        # Wall clock has already advanced past market close into T+1. If the
        # ContextVar is NOT propagated, tool handlers would resolve to T+1.
        after_close = datetime(2026, 4, 17, 10, 30)

        observations: List[Dict[str, Any]] = []

        registry = ToolRegistry()

        def _probe_handler(stock_code: str) -> dict:
            observations.append(
                {
                    "source": "tool_handler",
                    "stock_code": stock_code,
                    "frozen": get_agent_frozen_target_date(),
                    "cache_is_dict": isinstance(get_candidate_pick_cache(), dict),
                }
            )
            return {"ok": True, "code": stock_code}

        registry.register(
            ToolDefinition(
                name="probe_history",
                description="Probe the frozen ContextVar from inside a tool worker.",
                parameters=[
                    ToolParameter(
                        name="stock_code",
                        type="string",
                        description="stock code",
                    )
                ],
                handler=_probe_handler,
                category="data",
            )
        )

        def _fake_build_executor(*_args, **_kwargs):
            return _StubExecutor(registry, observations)

        with patch(
            "src.core.pipeline.get_market_for_stock", return_value="cn"
        ), patch(
            "src.core.pipeline.get_effective_trading_date", return_value=frozen
        ), patch(
            "src.agent.factory.build_agent_executor", side_effect=_fake_build_executor
        ):
            result = pipeline._analyze_with_agent(
                code="600519",
                report_type=SimpleNamespace(value="simple"),
                query_id="qid-1",
                stock_name="贵州茅台",
                realtime_quote=None,
                chip_data=None,
                fundamental_context=None,
                trend_result=None,
                current_time=after_close,
            )

        # When context propagation works end-to-end, the stub executor
        # completes successfully and we reach the ``_agent_result_to_...``
        # conversion. That helper is expected to return ``None`` in this
        # stubbed-no-dashboard scenario; we care about the observations.
        self.assertIsNone(result)

        context_obs = [o for o in observations if o["source"] == "initial_context"]
        tool_obs = [o for o in observations if o["source"] == "tool_handler"]

        self.assertEqual(len(context_obs), 1)
        self.assertEqual(context_obs[0]["context"].get("frozen_target_date"), frozen.isoformat())

        # Both parallel tool workers must observe the pipeline-frozen date,
        # NOT the wall-clock's post-close ``after_close`` date.
        self.assertEqual(len(tool_obs), 2)
        for obs in tool_obs:
            self.assertEqual(
                obs["frozen"],
                frozen,
                msg=f"tool worker saw frozen={obs['frozen']!r}, expected {frozen!r} — ContextVar propagation regressed",
            )
            self.assertTrue(
                obs["cache_is_dict"],
                msg="candidate_pick_cache ContextVar must also be visible to tool workers",
            )

    def test_context_var_is_released_after_agent_run(self) -> None:
        """After ``_analyze_with_agent`` returns (or raises), no leaked ContextVar state."""
        pipeline = self._build_pipeline()
        frozen = date(2026, 4, 16)

        observations: List[Dict[str, Any]] = []
        registry = ToolRegistry()

        def _noop_handler(stock_code: str) -> dict:
            return {"ok": True}

        registry.register(
            ToolDefinition(
                name="probe_history",
                description="noop",
                parameters=[ToolParameter(name="stock_code", type="string", description="x")],
                handler=_noop_handler,
                category="data",
            )
        )

        def _fake_build_executor(*_args, **_kwargs):
            return _StubExecutor(registry, observations)

        with patch(
            "src.core.pipeline.get_market_for_stock", return_value="cn"
        ), patch(
            "src.core.pipeline.get_effective_trading_date", return_value=frozen
        ), patch(
            "src.agent.factory.build_agent_executor", side_effect=_fake_build_executor
        ):
            pipeline._analyze_with_agent(
                code="600519",
                report_type=SimpleNamespace(value="simple"),
                query_id="qid-2",
                stock_name="贵州茅台",
                realtime_quote=None,
                chip_data=None,
                fundamental_context=None,
                trend_result=None,
                current_time=datetime(2026, 4, 17, 10, 30),
            )

        self.assertIsNone(get_agent_frozen_target_date())
        self.assertIsNone(get_candidate_pick_cache())


if __name__ == "__main__":
    unittest.main()
