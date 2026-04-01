# -*- coding: utf-8 -*-
"""
Regression tests for the 5-digit stock code ambiguity fix (Issue #946).

Bug: inputting '02714' (intended as A-share 002714 牧原股份) was routed to
HK stock path and returned wrong data (蒙牛乳业 HK:02319).

Covered acceptance criteria:
1. '002714' → '002714'  (6-digit A-share, unchanged)
2. '02714'  → '002714'  (5-digit, auto-padded to A-share SZ code)
3. '00700'  → '00700'   (known HK, kept as-is)
4. 'HK02714' → 'HK02714' (explicit HK prefix, stays HK)
5. '02714.HK' → 'HK02714' (explicit HK suffix, stays HK)
"""

import sys
import unittest
from unittest.mock import MagicMock

# Lightweight stubs for optional LLM runtime deps
if "litellm" not in sys.modules:
    sys.modules["litellm"] = MagicMock()
if "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

try:
    from data_provider.base import normalize_stock_code, _KNOWN_HK_BARE_CODES
    _IMPORTS_OK = True
    _IMPORT_ERROR = ""
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(e)


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

    def test_five_digit_02714_padded_to_ashare(self):
        """'02714' → '002714' (牧原股份 SZ, the bug case)."""
        self.assertEqual(normalize_stock_code("02714"), "002714")

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

    # --- _KNOWN_HK_BARE_CODES sanity check ---

    def test_known_hk_bare_codes_contains_tencent(self):
        """'00700' must be in the HK whitelist."""
        self.assertIn("00700", _KNOWN_HK_BARE_CODES)

    def test_known_hk_bare_codes_does_not_contain_02714(self):
        """'02714' must NOT be in the HK whitelist."""
        self.assertNotIn("02714", _KNOWN_HK_BARE_CODES)

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
