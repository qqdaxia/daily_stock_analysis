# -*- coding: utf-8 -*-
"""
Regression tests for the 5-digit stock code ambiguity fix (Issue #946).

Bug: inputting '02714' (intended as A-share 002714 牧原股份) was routed to
HK stock path and returned wrong data (蒙牛乳业 HK:02319).

Covered acceptance criteria:
1. '002714' → '002714'  (6-digit A-share, unchanged)
2. '02714'  → '02714'   (5-digit, kept as-is; manager 在港股路径无数据时可尝试补零到 '002714')
3. '00700'  → '00700'   (known HK, kept as-is)
4. 'HK02714' → 'HK02714' (explicit HK prefix, stays HK)
5. '02714.HK' → 'HK02714' (explicit HK suffix, stays HK)
6. '02319'  → '02319'   (real HK code not in old STOCK_NAME_MAP, must not be padded)
"""

import sys
import pandas as pd
import unittest
from unittest.mock import MagicMock

# Lightweight stubs for optional LLM runtime deps
if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

try:
    from data_provider.base import DataFetchError, DataFetcherManager, normalize_stock_code, _KNOWN_HK_BARE_CODES
    _IMPORTS_OK = True
    _IMPORT_ERROR = ""
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-03-01", "2026-03-02"],
            "open": [1.0, 1.1],
            "high": [1.2, 1.2],
            "low": [0.9, 1.0],
            "close": [1.1, 1.05],
            "volume": [1000, 1000],
            "amount": [10000, 10000],
            "pct_chg": [0.5, 0.5],
        }
    )


class _MockDailyFetcher:
    def __init__(self, name: str, priority: int, result_provider):
        self.name = name
        self.priority = priority
        self._result_provider = result_provider
        self.calls = []

    def get_daily_data(self, stock_code: str, start_date: str | None = None, end_date: str | None = None, days: int = 30):
        self.calls.append(stock_code)
        return self._result_provider(stock_code=stock_code, start_date=start_date, end_date=end_date, days=days)


@unittest.skipIf(not _IMPORTS_OK, f"imports failed: {_IMPORT_ERROR}")
class TestFiveDigitAmbiguity(unittest.TestCase):
    """
    Tests for the bare 5-digit code auto-pad heuristic in normalize_stock_code.
    """

    # --- Acceptance criteria ---

    def test_six_digit_ashare_unchanged(self):
        """6-digit A-share codes must not be modified."""
        self.assertEqual(normalize_stock_code("002714"), "002714")
        self.assertEqual(normalize_stock_code("000001"), "000001")
        self.assertEqual(normalize_stock_code("600519"), "600519")

    def test_five_digit_02714_kept_as_is(self):
        """'02714' must be kept as-is; upstream routing decides whether to pad."""
        self.assertEqual(normalize_stock_code("02714"), "02714")

    def test_five_digit_tencent_kept_as_hk(self):
        """'00700' (腾讯控股) is a known HK code and must stay '00700'."""
        self.assertEqual(normalize_stock_code("00700"), "00700")

    def test_explicit_hk_prefix_routes_to_hk(self):
        """'HK02714' has explicit HK prefix → stays 'HK02714'."""
        self.assertEqual(normalize_stock_code("HK02714"), "HK02714")

    def test_explicit_hk_suffix_routes_to_hk(self):
        """'02714.HK' has explicit HK suffix → normalizes to 'HK02714'."""
        self.assertEqual(normalize_stock_code("02714.HK"), "HK02714")

    # --- Other known HK whitelist codes must be preserved ---

    def test_meituan_kept_as_hk(self):
        """'03690' (美团) is a known HK code and must stay '03690'."""
        self.assertEqual(normalize_stock_code("03690"), "03690")

    def test_xiaomi_kept_as_hk(self):
        """'01810' (小米集团) is a known HK code and must stay '01810'."""
        self.assertEqual(normalize_stock_code("01810"), "01810")

    def test_kuaishou_kept_as_hk(self):
        """'01024' (快手) is a known HK code and must stay '01024'."""
        self.assertEqual(normalize_stock_code("01024"), "01024")

    def test_lixiang_kept_as_hk(self):
        """'02015' (理想汽车) is a known HK code and must stay '02015'."""
        self.assertEqual(normalize_stock_code("02015"), "02015")

    def test_hsbc_kept_as_hk(self):
        """'00005' (汇丰控股) is a known HK code and must stay '00005'."""
        self.assertEqual(normalize_stock_code("00005"), "00005")

    # --- 5-digit codes outside the 000-003 pad range are not touched ---

    def test_five_digit_alibaba_unchanged(self):
        """'09988' (阿里巴巴) does not fall in the 000-003 pad range → unchanged."""
        self.assertEqual(normalize_stock_code("09988"), "09988")

    def test_five_digit_outside_pad_range_unchanged(self):
        """'05500' padded → '005500' starts with '005', outside 000-003 → unchanged."""
        self.assertEqual(normalize_stock_code("05500"), "05500")

    def test_five_digit_02319_kept_as_hk(self):
        """'02319' (蒙牛乳业) is a real HK code absent from old STOCK_NAME_MAP;
        must NOT be padded to '002319' (which would be a wrong A-share code)."""
        self.assertEqual(normalize_stock_code("02319"), "02319")

    # --- _KNOWN_HK_BARE_CODES sanity check ---

    def test_known_hk_bare_codes_contains_tencent(self):
        """'00700' must be in the HK whitelist."""
        self.assertIn("00700", _KNOWN_HK_BARE_CODES)

    def test_known_hk_bare_codes_contains_02714(self):
        """'02714' must now be in the HK whitelist (all 5-digit codes are protected)."""
        self.assertIn("02714", _KNOWN_HK_BARE_CODES)

    def test_known_hk_bare_codes_contains_02319(self):
        """'02319' (蒙牛乳业, not in old STOCK_NAME_MAP) must be in the HK whitelist."""
        self.assertIn("02319", _KNOWN_HK_BARE_CODES)

    def test_manager_fallback_to_ashare_after_hk_empty(self):
        """港股路径无数据时会补零重试 A 股候选码。"""

        def result_provider(stock_code: str, **_):
            if stock_code == "002714":
                return _sample_df()
            return None

        fetcher = _MockDailyFetcher("AkshareFetcher", 1, result_provider)
        manager = DataFetcherManager(fetchers=[fetcher])
        df, source = manager.get_daily_data("02714", start_date="2026-03-01", end_date="2026-03-02")
        self.assertEqual(source, "AkshareFetcher")
        self.assertEqual(fetcher.calls, ["02714", "002714"])
        self.assertEqual(len(df), 2)

    def test_manager_does_not_retry_padded_for_explicit_hk_code(self):
        """显式 HK 代码不会触发 6 位补零兜底。"""

        fetcher = _MockDailyFetcher("AkshareFetcher", 1, lambda **_: None)
        manager = DataFetcherManager(fetchers=[fetcher])
        with self.assertRaises(DataFetchError):
            manager.get_daily_data("HK02714", start_date="2026-03-01", end_date="2026-03-02")
        self.assertEqual(fetcher.calls, ["HK02714"])

    # --- Other normalize_stock_code behaviours must be preserved ---

    def test_sh_prefix_stripped(self):
        """SH prefix is stripped as before."""
        self.assertEqual(normalize_stock_code("SH600519"), "600519")

    def test_sz_prefix_stripped(self):
        """SZ prefix is stripped as before."""
        self.assertEqual(normalize_stock_code("SZ000001"), "000001")

    def test_hk_suffix_to_prefix(self):
        """1810.HK → HK01810 as before."""
        self.assertEqual(normalize_stock_code("1810.HK"), "HK01810")

    def test_hk_prefix_short_padded(self):
        """HK700 → HK00700 as before."""
        self.assertEqual(normalize_stock_code("HK700"), "HK00700")


if __name__ == "__main__":
    unittest.main()
