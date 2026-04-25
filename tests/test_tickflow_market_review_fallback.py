# -*- coding: utf-8 -*-
"""Regression tests for TickFlow market-review manager fallback."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.base import DataFetcherManager


class _DummyFetcher:
    def __init__(self, name, indices=None, stats=None, hot_stocks=None):
        self.name = name
        self.priority = 1
        self.indices = indices
        self.stats = stats
        self.hot_stocks = hot_stocks
        self.index_calls = 0
        self.stats_calls = 0
        self.hot_stock_calls = 0

    def get_main_indices(self, region="cn"):
        self.index_calls += 1
        return self.indices

    def get_market_stats(self):
        self.stats_calls += 1
        return self.stats

    def get_hot_stocks(self, n=5):
        self.hot_stock_calls += 1
        return self.hot_stocks


class _DummyTickFlowFetcher:
    def __init__(self, indices=None, stats=None, error=None):
        self.indices = indices
        self.stats = stats
        self.error = error
        self.closed = False

    def get_main_indices(self, region="cn"):
        if self.error is not None:
            raise self.error
        return self.indices

    def get_market_stats(self):
        if self.error is not None:
            raise self.error
        return self.stats

    def close(self):
        self.closed = True


class TestTickFlowMarketReviewFallback(unittest.TestCase):
    def test_manager_prefers_tickflow_indices_when_available(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher("AkshareFetcher", indices=[{"code": "fallback"}])
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: _DummyTickFlowFetcher(
            indices=[{"code": "000001"}]
        )

        data = DataFetcherManager.get_main_indices(manager, region="cn")

        self.assertEqual(data, [{"code": "000001"}])
        self.assertEqual(fallback.index_calls, 0)

    def test_manager_falls_back_when_tickflow_indices_fail(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher("AkshareFetcher", indices=[{"code": "fallback"}])
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: _DummyTickFlowFetcher(
            error=RuntimeError("tickflow down")
        )

        data = DataFetcherManager.get_main_indices(manager, region="cn")

        self.assertEqual(data, [{"code": "fallback"}])
        self.assertEqual(fallback.index_calls, 1)

    def test_manager_falls_back_when_tickflow_indices_missing(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher("AkshareFetcher", indices=[{"code": "fallback"}])
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: _DummyTickFlowFetcher(
            indices=None
        )

        data = DataFetcherManager.get_main_indices(manager, region="cn")

        self.assertEqual(data, [{"code": "fallback"}])
        self.assertEqual(fallback.index_calls, 1)

    def test_manager_skips_tickflow_for_non_cn_indices(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher("YfinanceFetcher", indices=[{"code": "^GSPC"}])
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: self.fail(
            "TickFlow should not be called for non-CN indices"
        )

        data = DataFetcherManager.get_main_indices(manager, region="us")

        self.assertEqual(data, [{"code": "^GSPC"}])
        self.assertEqual(fallback.index_calls, 1)

    def test_manager_falls_back_when_tickflow_market_stats_fails(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher(
            "AkshareFetcher",
            stats={"up_count": 1, "down_count": 2, "flat_count": 3},
        )
        manager._fetchers = [fallback]
        manager._get_tickflow_fetcher = lambda: _DummyTickFlowFetcher(
            error=RuntimeError("tickflow down")
        )

        data = DataFetcherManager.get_market_stats(manager)

        self.assertEqual(data["up_count"], 1)
        self.assertEqual(fallback.stats_calls, 1)

    @patch("src.config.get_config")
    def test_manager_skips_tickflow_without_api_key(self, mock_get_config):
        mock_get_config.return_value = SimpleNamespace(tickflow_api_key=None)

        manager = DataFetcherManager.__new__(DataFetcherManager)
        fallback = _DummyFetcher(
            "AkshareFetcher",
            stats={"up_count": 2, "down_count": 1, "flat_count": 0},
        )
        manager._fetchers = [fallback]

        data = DataFetcherManager.get_market_stats(manager)

        self.assertEqual(data["up_count"], 2)
        self.assertEqual(fallback.stats_calls, 1)

    def test_manager_close_releases_tickflow_fetcher(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        tickflow_fetcher = _DummyTickFlowFetcher(indices=[{"code": "000001"}])
        manager._tickflow_fetcher = tickflow_fetcher
        manager._tickflow_api_key = "tf-secret"
        manager._tickflow_lock = None

        DataFetcherManager.close(manager)

        self.assertTrue(tickflow_fetcher.closed)
        self.assertIsNone(manager._tickflow_fetcher)
        self.assertIsNone(manager._tickflow_api_key)

    def test_manager_prefers_hot_stocks_when_available(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        primary = _DummyFetcher(
            "EfinanceFetcher",
            hot_stocks=[{"code": "300308", "name": "中际旭创", "change_pct": 9.9, "amount": 2e9}],
        )
        fallback = _DummyFetcher(
            "AkshareFetcher",
            hot_stocks=[{"code": "000001", "name": "平安银行", "change_pct": 3.1, "amount": 1e9}],
        )
        manager._fetchers = [primary, fallback]

        data = DataFetcherManager.get_hot_stocks(manager, n=3)

        self.assertEqual(data[0]["code"], "300308")
        self.assertEqual(primary.hot_stock_calls, 1)
        self.assertEqual(fallback.hot_stock_calls, 0)

    def test_manager_falls_back_when_hot_stocks_missing(self):
        manager = DataFetcherManager.__new__(DataFetcherManager)
        primary = _DummyFetcher("EfinanceFetcher", hot_stocks=[])
        fallback = _DummyFetcher(
            "AkshareFetcher",
            hot_stocks=[{"code": "000001", "name": "平安银行", "change_pct": 3.1, "amount": 1e9}],
        )
        manager._fetchers = [primary, fallback]

        data = DataFetcherManager.get_hot_stocks(manager, n=3)

        self.assertEqual(data[0]["code"], "000001")
        self.assertEqual(primary.hot_stock_calls, 1)
        self.assertEqual(fallback.hot_stock_calls, 1)


if __name__ == "__main__":
    unittest.main()
