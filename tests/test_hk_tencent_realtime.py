# -*- coding: utf-8 -*-
"""
Regression tests for the HK realtime quote path that fetches from the
Tencent single-stock endpoint (`http://qt.gtimg.cn/q=hkXXXXX`).

The Tencent payload uses a tilde-separated layout that mostly matches the
A-share format up to ``fields[37]``, but starting at ``fields[44]`` the
field semantics diverge (HK has no turnover_rate/PB/limit-up/limit-down at
those positions). These tests pin down the HK-specific field mapping and
guard against accidental reuse of the A-share parser.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

from data_provider.akshare_fetcher import AkshareFetcher, _to_hk_tx_symbol


# Real captured payload from `curl http://qt.gtimg.cn/q=hk00700`. Trimmed
# to keep the fixture readable but preserves all 78 fields.
TENCENT_HK_PAYLOAD_00700 = (
    'v_hk00700="100~腾讯控股~00700~479.200~473.800~474.600~19321018.0~0~0~'
    '479.200~0~0~0~0~0~0~0~0~0~479.200~0~0~0~0~0~0~0~0~0~19321018.0~'
    '2026/04/29 16:08:43~5.400~1.14~480.600~474.600~479.200~19321018.0~'
    '9231869057.790~0~17.57~~0~0~1.27~43729.8639~43729.8639~TENCENT~0.95~'
    '683.000~464.900~0.71~-43.20~0~0~0~0~0~17.57~3.45~0.21~100~-20.00~'
    '-4.92~GP~19.48~11.27~-3.97~-2.88~-21.05~9125597636.00~9125597636.00~'
    '17.57~4.531~477.815~-12.87~HKD~1~50";'
)


def _mock_response(text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.encoding = "gbk"
    return resp


class TestToHkTxSymbol(unittest.TestCase):
    """Pure-function input normalization for the Tencent/Sina HK symbol."""

    def test_plain_5_digit(self):
        self.assertEqual(_to_hk_tx_symbol("00700"), "hk00700")

    def test_4_digit_pads_to_5(self):
        self.assertEqual(_to_hk_tx_symbol("0700"), "hk00700")

    def test_uppercase_hk_prefix(self):
        self.assertEqual(_to_hk_tx_symbol("HK00700"), "hk00700")

    def test_dot_hk_suffix(self):
        self.assertEqual(_to_hk_tx_symbol("0700.HK"), "hk00700")

    def test_mixed_case_and_whitespace(self):
        self.assertEqual(_to_hk_tx_symbol(" Hk00700 "), "hk00700")


class TestHkTencentRealtimeParsing(unittest.TestCase):
    """End-to-end parsing of the Tencent HK payload."""

    def setUp(self):
        self.fetcher = AkshareFetcher()
        # Stub out side effects unrelated to the parsing under test.
        self.fetcher._set_random_user_agent = MagicMock()
        self.fetcher._enforce_rate_limit = MagicMock()

    @patch("data_provider.akshare_fetcher.requests.get")
    def test_parses_full_hk_quote(self, mock_get):
        mock_get.return_value = _mock_response(TENCENT_HK_PAYLOAD_00700)

        quote = self.fetcher._get_hk_realtime_quote_from_tencent("00700", "00700")

        self.assertIsNotNone(quote)
        self.assertEqual(quote.code, "00700")
        self.assertEqual(quote.name, "腾讯控股")
        self.assertAlmostEqual(quote.price, 479.200)
        self.assertAlmostEqual(quote.pre_close, 473.800)
        self.assertAlmostEqual(quote.open_price, 474.600)
        self.assertAlmostEqual(quote.high, 480.600)
        self.assertAlmostEqual(quote.low, 474.600)
        self.assertAlmostEqual(quote.change_amount, 5.400)
        self.assertAlmostEqual(quote.change_pct, 1.14)
        # HK volume is reported in shares, not lots — must NOT be multiplied by 100.
        self.assertEqual(quote.volume, 19321018)
        self.assertAlmostEqual(quote.amount, 9231869057.790)
        self.assertAlmostEqual(quote.pe_ratio, 17.57)
        # 总市值 / 流通市值 are reported in 亿 (1e8); converted to 元.
        self.assertAlmostEqual(quote.total_mv, 43729.8639 * 1e8)
        self.assertAlmostEqual(quote.circ_mv, 43729.8639 * 1e8)
        self.assertAlmostEqual(quote.high_52w, 683.000)
        self.assertAlmostEqual(quote.low_52w, 464.900)

    @patch("data_provider.akshare_fetcher.requests.get")
    def test_returns_none_on_http_500(self, mock_get):
        mock_get.return_value = _mock_response("", status_code=500)
        self.assertIsNone(
            self.fetcher._get_hk_realtime_quote_from_tencent("00700", "00700")
        )

    @patch("data_provider.akshare_fetcher.requests.get")
    def test_returns_none_on_empty_payload(self, mock_get):
        mock_get.return_value = _mock_response('v_hk00700="";')
        self.assertIsNone(
            self.fetcher._get_hk_realtime_quote_from_tencent("00700", "00700")
        )

    @patch("data_provider.akshare_fetcher.requests.get")
    def test_returns_none_on_short_field_count(self, mock_get):
        mock_get.return_value = _mock_response('v_hk00700="100~腾讯~00700~479.2";')
        self.assertIsNone(
            self.fetcher._get_hk_realtime_quote_from_tencent("00700", "00700")
        )

    @patch("data_provider.akshare_fetcher.requests.get")
    def test_returns_none_when_price_is_zero(self, mock_get):
        # Replace the price field (index 3) with 0 — should be treated as no data.
        zero_price_payload = TENCENT_HK_PAYLOAD_00700.replace(
            "00700~479.200~", "00700~0~"
        )
        mock_get.return_value = _mock_response(zero_price_payload)
        self.assertIsNone(
            self.fetcher._get_hk_realtime_quote_from_tencent("00700", "00700")
        )

    @patch("data_provider.akshare_fetcher.requests.get")
    def test_request_exception_returns_none(self, mock_get):
        mock_get.side_effect = ConnectionError("Connection aborted.")
        self.assertIsNone(
            self.fetcher._get_hk_realtime_quote_from_tencent("00700", "00700")
        )

    @patch("data_provider.akshare_fetcher.requests.get")
    def test_quote_code_uses_normalized_form(self, mock_get):
        """quote.code must be the 5-digit normalized form regardless of input shape.

        Downstream supplement / cache / comparison logic should see a single
        canonical code, not the user's original input (HK00700, 0700.HK, etc.).
        """
        mock_get.return_value = _mock_response(TENCENT_HK_PAYLOAD_00700)
        for raw_input in ("00700", "HK00700", "hk00700", "0700.HK", "0700"):
            with self.subTest(raw_input=raw_input):
                quote = self.fetcher._get_hk_realtime_quote_from_tencent(
                    raw_input, "00700"
                )
                self.assertIsNotNone(quote)
                self.assertEqual(quote.code, "00700")


if __name__ == "__main__":
    unittest.main()
