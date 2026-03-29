# -*- coding: utf-8 -*-
"""
Shared stock code utilities.
"""

from __future__ import annotations

import re
from typing import Optional


# Known exchange prefixes (case-insensitive) and the digit lengths they accept.
# e.g. SH600519 -> 600519, HK00700 -> 00700
_PREFIX_DIGIT_LENS: dict = {
    "SH": (6,),
    "SZ": (6,),
    "SS": (6,),
    "HK": (1, 2, 3, 4, 5),
}

_SUFFIX_DIGIT_LENS: dict = {
    ".SH": (6,),
    ".SZ": (6,),
    ".SS": (6,),
    ".HK": (1, 2, 3, 4, 5),
    ".TW": (4,),
}


def _strip_exchange_prefix(text: str) -> Optional[str]:
    """Strip leading exchange prefix (SH/SZ/HK etc.) and return the bare digits, or None."""
    for prefix, digit_lens in _PREFIX_DIGIT_LENS.items():
        if text.startswith(prefix):
            base = text[len(prefix):]
            if base.isdigit() and len(base) in digit_lens:
                return base.zfill(5) if prefix == "HK" else base
    return None


def _strip_exchange_suffix(text: str) -> Optional[str]:
    """Strip exchange suffix (.SH/.SZ/.SS/.HK) and return normalized bare digits, or None."""
    for suffix, digit_lens in _SUFFIX_DIGIT_LENS.items():
        if text.endswith(suffix):
            base = text[: -len(suffix)].strip()
            if base.isdigit() and len(base) in digit_lens:
                if suffix == ".HK":
                    return base.zfill(5)
                if suffix == ".TW":
                    return f"{base}.TW"
                return base
    return None


def normalize_tw_code(raw: str, *, allow_bare: bool = True) -> Optional[str]:
    """Normalize Taiwan stock code to the canonical ``NNNN.TW`` form."""
    text = (raw or "").strip().upper()
    if not text:
        return None

    if text.endswith(".TW"):
        base = text[:-3].strip()
        if base.isdigit() and len(base) == 4:
            return f"{base}.TW"
        return None

    if allow_bare and text.isdigit() and len(text) == 4:
        return f"{text}.TW"

    return None


def is_tw_code(raw: str, *, allow_bare: bool = True) -> bool:
    """Return True when the input looks like a Taiwan stock code."""
    return normalize_tw_code(raw, allow_bare=allow_bare) is not None


def is_code_like(value: str) -> bool:
    """Check if string looks like a stock code (A/H/US/TW suffix, digits, or prefixed code)."""
    text = value.strip().upper()
    if not text:
        return False
    if text.isdigit() and len(text) in (5, 6):
        return True
    if normalize_tw_code(text, allow_bare=False) is not None:
        return True
    if _strip_exchange_suffix(text) is not None:
        return True
    if re.match(r"^[A-Z]{1,5}(?:\.(?:US|[A-Z]))?$", text):
        return True
    # Support exchange-prefixed codes: SH600519, SZ000001, HK00700
    if _strip_exchange_prefix(text) is not None:
        return True
    return False


def normalize_code(raw: str) -> Optional[str]:
    """Normalize and validate a single stock code.

    Supports:
    - Plain digit codes: 600519, 00700
    - Suffix format: 600519.SH, 600519.SZ, 00700.HK
    - Prefix format: SH600519, SZ000001, HK00700 (case-insensitive)
    - Taiwan suffix format: 2330.TW
    - US ticker symbols: AAPL, TSLA
    """
    text = raw.strip().upper()
    if not text:
        return None
    normalized_tw = normalize_tw_code(text, allow_bare=False)
    if normalized_tw is not None:
        return normalized_tw
    if text.isdigit() and len(text) in (5, 6):
        return text
    if re.match(r"^[A-Z]{1,5}(?:\.(?:US|[A-Z]))?$", text):
        return text
    stripped_suffix = _strip_exchange_suffix(text)
    if stripped_suffix is not None:
        return stripped_suffix
    # Support exchange-prefixed codes: SH600519 -> 600519, HK00700 -> 00700
    stripped = _strip_exchange_prefix(text)
    if stripped is not None:
        return stripped
    return None
