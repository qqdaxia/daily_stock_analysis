# -*- coding: utf-8 -*-
"""Regression tests for Taiwan stock market support."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()
if "newspaper" not in sys.modules:
    mock_np = MagicMock()
    mock_np.Article = MagicMock()
    mock_np.Config = MagicMock()
    sys.modules["newspaper"] = mock_np

from data_provider.base import DataFetcherManager, canonical_stock_code, normalize_stock_code
from data_provider.yfinance_fetcher import YfinanceFetcher
from src.core.trading_calendar import MARKET_EXCHANGE, get_market_for_stock
from src.market_context import detect_market, get_market_guidelines
from src.search_service import SearchResponse, SearchResult, SearchService


def _search_response(query: str) -> SearchResponse:
    return SearchResponse(
        query=query,
        results=[
            SearchResult(
                title="news",
                snippet="snippet",
                url="https://example.com/news",
                source="example.com",
                published_date="2026-03-29",
            )
        ],
        provider="Mock",
        success=True,
    )


class TaiwanCodeNormalizationTestCase(unittest.TestCase):
    def test_base_normalize_stock_code_accepts_bare_tw_code(self) -> None:
        self.assertEqual(normalize_stock_code("2330"), "2330.TW")
        self.assertEqual(normalize_stock_code("2330.tw"), "2330.TW")

    def test_base_canonical_stock_code_promotes_bare_tw_code(self) -> None:
        self.assertEqual(canonical_stock_code("2330"), "2330.TW")
        self.assertEqual(canonical_stock_code("2330.tw"), "2330.TW")

    def test_yfinance_convert_stock_code_accepts_bare_tw_code(self) -> None:
        fetcher = YfinanceFetcher()
        self.assertEqual(fetcher._convert_stock_code("2330"), "2330.TW")
        self.assertEqual(fetcher._convert_stock_code("2330.TW"), "2330.TW")


class TaiwanRoutingTestCase(unittest.TestCase):
    def test_data_fetcher_manager_routes_tw_daily_data_to_yfinance(self) -> None:
        calls: list[str] = []

        def _unexpected_call(*args, **kwargs):
            raise AssertionError("non-YFinance fetcher should not be used for TW stocks")

        yfinance_fetcher = SimpleNamespace(
            name="YfinanceFetcher",
            priority=9,
            get_daily_data=lambda stock_code, start_date=None, end_date=None, days=30: (
                calls.append(stock_code) or pd.DataFrame({"close": [123.0]})
            ),
        )
        manager = DataFetcherManager(
            fetchers=[
                SimpleNamespace(name="EfinanceFetcher", priority=0, get_daily_data=_unexpected_call),
                SimpleNamespace(name="AkshareFetcher", priority=1, get_daily_data=_unexpected_call),
                yfinance_fetcher,
            ]
        )

        df, source = manager.get_daily_data("2330", days=5)

        self.assertFalse(df.empty)
        self.assertEqual(source, "YfinanceFetcher")
        self.assertEqual(calls, ["2330.TW"])


class TaiwanMarketContextTestCase(unittest.TestCase):
    def test_market_context_detects_tw(self) -> None:
        self.assertEqual(detect_market("2330"), "tw")
        self.assertEqual(detect_market("2330.TW"), "tw")
        self.assertIn("台股", get_market_guidelines("2330.TW"))

    def test_trading_calendar_detects_tw_market(self) -> None:
        self.assertEqual(MARKET_EXCHANGE["tw"], "XTAI")
        self.assertEqual(get_market_for_stock("2330"), "tw")
        self.assertEqual(get_market_for_stock("2330.TW"), "tw")


class TaiwanSearchQueryTestCase(unittest.TestCase):
    def test_search_stock_news_uses_tw_query_template(self) -> None:
        service = SearchService(
            bocha_keys=["dummy_key"],
            searxng_public_instances_enabled=False,
            news_max_age_days=3,
            news_strategy_profile="short",
        )
        mock_search = MagicMock(return_value=_search_response("台积电 2330 台股 最新消息"))
        service._providers[0].search = mock_search

        service.search_stock_news("2330.TW", "台积电", max_results=3)

        self.assertEqual(mock_search.call_args.args[0], "台积电 2330 台股 最新消息")

    def test_search_comprehensive_intel_uses_tw_dimensions(self) -> None:
        service = SearchService(
            bocha_keys=["dummy_key"],
            searxng_public_instances_enabled=False,
            news_max_age_days=3,
            news_strategy_profile="short",
        )
        mock_search = MagicMock(
            side_effect=[
                _search_response("latest"),
                _search_response("market"),
            ]
        )
        service._providers[0].search = mock_search

        with patch("src.search_service.time.sleep"):
            service.search_comprehensive_intel("2330.TW", "台积电", max_searches=2)

        first_query = mock_search.call_args_list[0].args[0]
        second_query = mock_search.call_args_list[1].args[0]
        self.assertIn("台股", first_query)
        self.assertIn("法说会", second_query)


if __name__ == "__main__":
    unittest.main()
