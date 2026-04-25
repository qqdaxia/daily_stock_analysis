import os
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.efinance_fetcher import EfinanceFetcher


class _DummyCircuitBreaker:
    def __init__(self):
        self.failures = []

    def record_failure(self, source: str, error=None) -> None:
        self.failures.append((source, error))

    def record_success(self, source: str) -> None:
        return None

    def is_available(self, source: str) -> bool:
        return True


class TestEfinanceMainIndices(unittest.TestCase):
    def test_get_main_indices_prefers_jinkai_column_for_open_price(self):
        fetcher = EfinanceFetcher()
        fake_df = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "最新价": [3200.0],
                "涨跌幅": [0.63],
                "涨跌额": [20.0],
                "今开": [3188.0],
                "开盘": [0.0],
                "最高": [3215.0],
                "最低": [3170.0],
                "成交量": [123456789],
                "成交额": [9876543210.0],
                "振幅": [1.2],
            }
        )
        fake_efinance = types.SimpleNamespace(
            stock=types.SimpleNamespace(get_realtime_quotes=lambda *args, **kwargs: fake_df)
        )

        with patch.dict(sys.modules, {"efinance": fake_efinance}):
            with patch.object(fetcher, "_set_random_user_agent", return_value=None), patch.object(
                fetcher, "_enforce_rate_limit", return_value=None
            ):
                data = fetcher.get_main_indices(region="cn")

        self.assertIsNotNone(data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["code"], "sh000001")
        self.assertEqual(data[0]["name"], "上证指数")
        self.assertAlmostEqual(data[0]["open"], 3188.0)
        self.assertAlmostEqual(data[0]["current"], 3200.0)

    def test_get_main_indices_falls_back_to_kaipan_when_jinkai_is_missing(self):
        fetcher = EfinanceFetcher()
        fake_df = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "最新价": [3200.0],
                "涨跌幅": [0.63],
                "涨跌额": [20.0],
                "今开": [""],
                "开盘": [3186.0],
                "最高": [3215.0],
                "最低": [3170.0],
                "成交量": [123456789],
                "成交额": [9876543210.0],
                "振幅": [1.2],
            }
        )
        fake_efinance = types.SimpleNamespace(
            stock=types.SimpleNamespace(get_realtime_quotes=lambda *args, **kwargs: fake_df)
        )

        with patch.dict(sys.modules, {"efinance": fake_efinance}):
            with patch.object(fetcher, "_set_random_user_agent", return_value=None), patch.object(
                fetcher, "_enforce_rate_limit", return_value=None
            ):
                data = fetcher.get_main_indices(region="cn")

        self.assertIsNotNone(data)
        self.assertEqual(len(data), 1)
        self.assertAlmostEqual(data[0]["open"], 3186.0)

    def test_get_realtime_quote_records_circuit_breaker_failure_on_parse_error(self):
        fetcher = EfinanceFetcher()
        breaker = _DummyCircuitBreaker()
        malformed_df = pd.DataFrame({"unexpected": ["value"]})

        with patch("data_provider.efinance_fetcher.get_realtime_circuit_breaker", return_value=breaker):
            with patch.object(fetcher, "_get_stock_realtime_dataframe", return_value=malformed_df):
                quote = fetcher.get_realtime_quote("000001")

        self.assertIsNone(quote)
        self.assertEqual(len(breaker.failures), 1)
        self.assertEqual(breaker.failures[0][0], "efinance")
        self.assertIn("code", str(breaker.failures[0][1]))

    def test_get_hot_stocks_records_circuit_breaker_failure_on_schema_drift(self):
        fetcher = EfinanceFetcher()
        breaker = _DummyCircuitBreaker()
        malformed_df = pd.DataFrame({"pct_chg": [9.9], "amount": [1_000_000.0]})

        with patch("data_provider.efinance_fetcher.get_realtime_circuit_breaker", return_value=breaker):
            with patch.object(fetcher, "_get_stock_realtime_dataframe", return_value=malformed_df):
                hot_stocks = fetcher.get_hot_stocks()

        self.assertIsNone(hot_stocks)
        self.assertEqual(len(breaker.failures), 1)
        self.assertEqual(breaker.failures[0][0], "efinance")
        self.assertIn("hot_stocks schema missing required columns", str(breaker.failures[0][1]))
        self.assertIn("code", str(breaker.failures[0][1]))


if __name__ == "__main__":
    unittest.main()
