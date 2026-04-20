# -*- coding: utf-8 -*-
"""Tests for shared stock history cache behavior."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict

import pandas as pd

from src.services.stock_history_cache import (
    AGENT_HISTORY_BASELINE_DAYS,
    _ensure_min_history_cached_with_bars,
    _resolve_expected_target_date,
    ensure_min_history_cached,
    load_recent_bars_from_db,
    load_recent_history_df,
    reset_agent_frozen_target_date,
    reset_candidate_pick_cache,
    reset_shared_history_runtime,
    set_agent_frozen_target_date,
    set_candidate_pick_cache,
)


@dataclass
class _Bar:
    code: str
    date: date
    close: float
    data_source: str = "seed"

    def to_dict(self):
        return {
            "code": self.code,
            "date": self.date,
            "open": self.close - 1,
            "high": self.close + 1,
            "low": self.close - 2,
            "close": self.close,
            "volume": 1000.0,
            "amount": 10000.0,
            "pct_chg": 1.0,
            "ma5": self.close,
            "ma10": self.close,
            "ma20": self.close,
            "volume_ratio": 1.0,
            "data_source": self.data_source,
        }


class _DummyDB:
    def __init__(self):
        self._rows: Dict[str, Dict[date, _Bar]] = {}

    def seed(self, code: str, count: int, *, end_date: date, source: str = "seed") -> None:
        self._rows.setdefault(code, {})
        for idx in range(count):
            current_date = end_date - timedelta(days=count - idx - 1)
            self._rows[code][current_date] = _Bar(
                code=code,
                date=current_date,
                close=100 + idx,
                data_source=source,
            )

    def get_latest_data(self, code: str, days: int = 2):
        rows = sorted(self._rows.get(code, {}).values(), key=lambda bar: bar.date, reverse=True)
        return rows[:days]

    def get_data_range(self, code: str, start_date: date, end_date: date):
        rows = [
            bar
            for bar in self._rows.get(code, {}).values()
            if start_date <= bar.date <= end_date
        ]
        return sorted(rows, key=lambda bar: bar.date)

    def save_daily_data(self, df: pd.DataFrame, code: str, data_source: str = "Unknown") -> int:
        bucket = self._rows.setdefault(code, {})
        new_count = 0
        for row in df.to_dict(orient="records"):
            row_date = row["date"]
            if isinstance(row_date, pd.Timestamp):
                row_date = row_date.date()
            if row_date not in bucket:
                new_count += 1
            bucket[row_date] = _Bar(
                code=code,
                date=row_date,
                close=float(row.get("close", 0) or 0),
                data_source=data_source,
            )
        return new_count


def _make_history_df(code: str, count: int, *, end_date: date) -> pd.DataFrame:
    rows = []
    for idx in range(count):
        current_date = end_date - timedelta(days=count - idx - 1)
        close = 200 + idx
        rows.append(
            {
                "date": current_date,
                "open": close - 1,
                "high": close + 1,
                "low": close - 2,
                "close": close,
                "volume": 1000.0,
                "amount": 10000.0,
                "pct_chg": 1.0,
                "ma5": close,
                "ma10": close,
                "ma20": close,
                "volume_ratio": 1.0,
            }
        )
    return pd.DataFrame(rows)


class StockHistoryCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        reset_shared_history_runtime()

    def tearDown(self) -> None:
        reset_shared_history_runtime()

    def test_load_recent_history_uses_db_when_sufficient(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 240, end_date=target_date, source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        with patch("src.services.stock_history_cache.get_db", return_value=db):
            df, source = load_recent_history_df(
                "600519",
                days=120,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertEqual(len(df), 120)
        self.assertEqual(source, "seed")
        manager.get_daily_data.assert_not_called()

    def test_ensure_min_history_cached_backfills_once_and_saves(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 30, end_date=target_date, source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", AGENT_HISTORY_BASELINE_DAYS, end_date=target_date),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            ok, source = ensure_min_history_cached(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )
            df, df_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertTrue(ok)
        self.assertEqual(source, "Fetcher")
        self.assertEqual(df_source, "Fetcher")
        self.assertEqual(len(df), 60)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_short_history_blocks_retry_for_same_target_date(self) -> None:
        """When the upstream returns insufficient history (<requested_days),
        the persisted-validation-failure branch now records an actual error
        message (Phase E-1), which trips the circuit-breaker for the same
        ``(canonical_code, target_date)`` attempt key so subsequent calls
        don't spam the fetcher with the exact same request (#1066 root fix).

        Callers can still force a retry via ``force_refresh=True`` or by
        moving to a different ``target_date``.
        """
        target_date = date(2026, 4, 16)
        db = _DummyDB()

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.side_effect = [
            (
                _make_history_df("600519", 20, end_date=target_date),
                "Fetcher",
            ),
            (
                _make_history_df("600519", AGENT_HISTORY_BASELINE_DAYS, end_date=target_date),
                "Fetcher",
            ),
        ]

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            first_df, first_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )
            second_df, second_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )
            third_df, third_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
                force_refresh=True,
            )

        # First call returns what the fetcher produced (20 fresh bars, which
        # is below the 60-day ask); load_recent_history_df does not force
        # a minimum count when bars are fresh, so caller sees the short df.
        self.assertEqual(len(first_df), 20)
        self.assertEqual(first_source, "Fetcher")
        # Second call is short-circuited by the recorded ``last_error``
        # (Phase E-1); no new fetch is issued. load_recent_history_df still
        # returns the last DB snapshot (20 fresh bars) but with the stale
        # attempt's error as source for diagnostics.
        self.assertEqual(len(second_df), 20)
        # force_refresh=True bypasses the circuit-breaker and delivers the
        # full fresh history.
        self.assertEqual(len(third_df), 60)
        self.assertEqual(third_source, "Fetcher")
        self.assertEqual(manager.get_daily_data.call_count, 2)

    def test_ensure_min_history_cached_returns_false_for_stale_saved_history(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df(
                "600519",
                AGENT_HISTORY_BASELINE_DAYS,
                end_date=target_date - timedelta(days=1),
            ),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            ok, source = ensure_min_history_cached(
                "600519",
                days=120,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertFalse(ok)
        self.assertIn("Stale historical data", source)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_stale_history_without_target_date_triggers_refresh(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 240, end_date=target_date - timedelta(days=1), source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", AGENT_HISTORY_BASELINE_DAYS, end_date=target_date),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db), patch(
            "src.core.trading_calendar.get_market_for_stock",
            return_value="cn",
        ), patch(
            "src.core.trading_calendar.get_effective_trading_date",
            return_value=target_date,
        ):
            df, source = load_recent_history_df(
                "600519",
                days=120,
                fetcher_manager=manager,
            )

        self.assertEqual(len(df), 120)
        self.assertEqual(source, "Fetcher")
        self.assertEqual(df.iloc[-1]["date"], target_date)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_normalized_code_lookup_hits_existing_db_cache(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 240, end_date=target_date, source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        with patch("src.services.stock_history_cache.get_db", return_value=db):
            df, source = load_recent_history_df(
                "SH600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertEqual(len(df), 60)
        self.assertEqual(source, "seed")
        manager.get_daily_data.assert_not_called()

    def test_backfill_saves_under_normalized_code(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", AGENT_HISTORY_BASELINE_DAYS, end_date=target_date),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            ok, source = ensure_min_history_cached(
                "600519.SH",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertTrue(ok)
        self.assertEqual(source, "Fetcher")
        self.assertIn("600519", db._rows)
        self.assertNotIn("600519.SH", db._rows)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_failed_fetch_is_not_retried_twice_in_same_process(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.side_effect = RuntimeError("boom")

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            first_df, first_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )
            second_df, second_source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertTrue(first_df.empty)
        self.assertTrue(second_df.empty)
        self.assertIn("boom", first_source)
        self.assertIn("boom", second_source)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_stale_db_is_not_returned_when_refresh_fails(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 120, end_date=target_date - timedelta(days=1), source="stale-db")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.side_effect = RuntimeError("boom")

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            df, source = load_recent_history_df(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertTrue(df.empty)
        self.assertIn("boom", source)
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_stale_successful_attempt_blocks_plain_retry_but_honours_force_refresh(self) -> None:
        """Stale persisted bars also trip the circuit-breaker (Phase E-1).

        Previously the persisted-validation-failure branch recorded
        ``last_error=None``, which allowed the caller to silently retry
        the exact same stale request in-process (the root cause of the
        #1066 "45 HTTP requests" observation). Now subsequent plain calls
        are short-circuited with the recorded stale-error message; callers
        must opt-in via ``force_refresh=True`` to re-poll the provider.
        """
        target_date = date(2026, 4, 16)
        db = _DummyDB()

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.side_effect = [
            (
                _make_history_df(
                    "600519",
                    AGENT_HISTORY_BASELINE_DAYS,
                    end_date=target_date - timedelta(days=1),
                ),
                "Fetcher",
            ),
            (
                _make_history_df(
                    "600519",
                    AGENT_HISTORY_BASELINE_DAYS,
                    end_date=target_date,
                ),
                "Fetcher",
            ),
        ]

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            first_df, first_source = load_recent_history_df(
                "600519",
                days=120,
                target_date=target_date,
                fetcher_manager=manager,
            )
            second_df, second_source = load_recent_history_df(
                "600519",
                days=120,
                target_date=target_date,
                fetcher_manager=manager,
            )
            third_df, third_source = load_recent_history_df(
                "600519",
                days=120,
                target_date=target_date,
                fetcher_manager=manager,
                force_refresh=True,
            )

        self.assertTrue(first_df.empty)
        self.assertTrue(second_df.empty)
        self.assertIn("Stale historical data", first_source)
        self.assertIn("Stale historical data", second_source)
        self.assertEqual(len(third_df), 120)
        self.assertEqual(third_df.iloc[-1]["date"], target_date)
        self.assertEqual(third_source, "Fetcher")
        # First call fetches (stale) and second short-circuits; force_refresh
        # re-polls and succeeds → exactly two provider calls total.
        self.assertEqual(manager.get_daily_data.call_count, 2)

    def test_load_recent_history_prefers_fresher_bucket_over_longer_legacy_bucket(self) -> None:
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("SH600519", 120, end_date=target_date - timedelta(days=1), source="legacy")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", 50, end_date=target_date),
            "Fetcher",
        )

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            df, source = load_recent_history_df(
                "SH600519",
                days=90,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertEqual(len(df), 50)
        self.assertEqual(df.iloc[-1]["date"], target_date)
        self.assertEqual(source, "Fetcher")
        manager.get_daily_data.assert_called_once_with("600519", days=AGENT_HISTORY_BASELINE_DAYS)

    def test_resolve_expected_target_date_prefers_contextvar(self) -> None:
        """Phase A: the pipeline-frozen ContextVar wins over wall-clock fallback."""
        from unittest.mock import patch

        frozen = date(2026, 4, 16)
        wallclock = date(2026, 4, 20)

        token = set_agent_frozen_target_date(frozen)
        try:
            with patch(
                "src.core.trading_calendar.get_market_for_stock",
                return_value="cn",
            ), patch(
                "src.core.trading_calendar.get_effective_trading_date",
                return_value=wallclock,
            ):
                resolved = _resolve_expected_target_date("600519", None)
        finally:
            reset_agent_frozen_target_date(token)

        self.assertEqual(resolved, frozen)

    def test_resolve_expected_target_date_falls_back_when_no_contextvar(self) -> None:
        """Without a ContextVar, wall-clock path is preserved."""
        from unittest.mock import patch

        wallclock = date(2026, 4, 20)
        with patch(
            "src.core.trading_calendar.get_market_for_stock",
            return_value="cn",
        ), patch(
            "src.core.trading_calendar.get_effective_trading_date",
            return_value=wallclock,
        ):
            resolved = _resolve_expected_target_date("600519", None)

        self.assertEqual(resolved, wallclock)

    def test_resolve_expected_target_date_explicit_arg_overrides_contextvar(self) -> None:
        """Explicit target_date argument still wins over a ContextVar value."""
        frozen = date(2026, 4, 10)
        explicit = date(2026, 4, 16)

        token = set_agent_frozen_target_date(frozen)
        try:
            resolved = _resolve_expected_target_date("600519", explicit)
        finally:
            reset_agent_frozen_target_date(token)

        self.assertEqual(resolved, explicit)

    def test_ensure_min_history_cached_with_bars_returns_bars_on_hit(self) -> None:
        """Phase E-2: internal helper returns the final bars list on the hot path."""
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 240, end_date=target_date, source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        with patch("src.services.stock_history_cache.get_db", return_value=db):
            ok_internal, source_internal, bars = _ensure_min_history_cached_with_bars(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )
            public_result = ensure_min_history_cached(
                "600519",
                days=60,
                target_date=target_date,
                fetcher_manager=manager,
            )

        self.assertTrue(ok_internal)
        self.assertEqual(source_internal, "seed")
        self.assertEqual(len(bars), 60)
        # Public wrapper keeps the 2-tuple signature (no bars leak) and the
        # same cache-hit semantics as the internal 3-tuple variant.
        self.assertEqual(len(public_result), 2)
        ok_public, source_public = public_result
        self.assertTrue(ok_public)
        self.assertEqual(source_public, source_internal)
        manager.get_daily_data.assert_not_called()

    def test_candidate_pick_cache_scoped_by_contextvar(self) -> None:
        """Phase E-5: installed cache short-circuits candidate enumeration."""
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 30, end_date=target_date, source="seed")

        from unittest.mock import patch

        calls = []
        real_get_latest = db.get_latest_data
        real_get_range = db.get_data_range

        def tracking_get_latest(code, days=2):
            calls.append(("latest", code, days))
            return real_get_latest(code, days=days)

        def tracking_get_range(code, start_date, end_date):
            calls.append(("range", code, start_date, end_date))
            return real_get_range(code, start_date, end_date)

        db.get_latest_data = tracking_get_latest  # type: ignore
        db.get_data_range = tracking_get_range    # type: ignore

        with patch("src.services.stock_history_cache.get_db", return_value=db):
            # Without a cache, two back-to-back calls hit the DB twice each,
            # once per candidate code.
            _bars_a, _src_a, _c_a = load_recent_bars_from_db(
                "600519", 30, target_date=target_date
            )
            uncached_calls = list(calls)
            calls.clear()

            cache_token = set_candidate_pick_cache({})
            try:
                _bars_b, _src_b, _c_b = load_recent_bars_from_db(
                    "600519", 30, target_date=target_date
                )
                first_call_db_hits = list(calls)
                calls.clear()
                _bars_c, _src_c, _c_c = load_recent_bars_from_db(
                    "600519", 30, target_date=target_date
                )
                second_call_db_hits = list(calls)
            finally:
                reset_candidate_pick_cache(cache_token)

        # Without cache we issued some DB reads; with cache installed, the
        # second call issues zero DB reads (candidate pick is cached).
        self.assertGreater(len(uncached_calls), 0)
        self.assertGreater(len(first_call_db_hits), 0)
        self.assertEqual(second_call_db_hits, [])

    def test_candidate_pick_cache_invalidated_after_save(self) -> None:
        """R7: cache must not serve stale pre-save snapshot after a refresh."""
        target_date = date(2026, 4, 16)
        db = _DummyDB()
        # Seed an initial stale bucket so the pre-check reads bars but rejects
        # them as stale.
        db.seed("600519", 30, end_date=target_date - timedelta(days=1), source="seed")

        from unittest.mock import MagicMock, patch

        manager = MagicMock()
        manager.get_daily_data.return_value = (
            _make_history_df("600519", AGENT_HISTORY_BASELINE_DAYS, end_date=target_date),
            "Fetcher",
        )

        cache_token = set_candidate_pick_cache({})
        try:
            with patch("src.services.stock_history_cache.get_db", return_value=db):
                df, source = load_recent_history_df(
                    "600519",
                    days=60,
                    target_date=target_date,
                    fetcher_manager=manager,
                )
        finally:
            reset_candidate_pick_cache(cache_token)

        self.assertEqual(len(df), 60)
        self.assertEqual(df.iloc[-1]["date"], target_date)
        self.assertEqual(source, "Fetcher")

    def test_candidate_pick_cache_released_even_when_caller_raises(self) -> None:
        """R7: ContextVar contract — reset releases the cache even on error."""
        from src.services.stock_history_cache import get_candidate_pick_cache

        class _Boom(Exception):
            pass

        token = set_candidate_pick_cache({})
        try:
            self.assertIsNotNone(get_candidate_pick_cache())
            raise _Boom("agent raised")
        except _Boom:
            pass
        finally:
            reset_candidate_pick_cache(token)

        self.assertIsNone(get_candidate_pick_cache())

    def test_load_recent_bars_from_db_honours_contextvar_when_target_none(self) -> None:
        """P2 回归锁：target_date=None + ContextVar 设置 T 时，
        ``load_recent_bars_from_db`` 的 resolver 必须与
        ``_ensure_min_history_cached_with_bars`` 的 save-后失效 key 对称
        （都基于 ``_resolve_expected_target_date``），避免未来新增
        ``target_date=None`` 调用时缓存 key 与失效 key 错位（#1066 follow-up）。
        """
        from unittest.mock import patch

        frozen = date(2026, 4, 16)
        db = _DummyDB()
        db.seed("600519", 30, end_date=frozen, source="seed")

        token = set_agent_frozen_target_date(frozen)
        try:
            with patch("src.services.stock_history_cache.get_db", return_value=db):
                bars_via_none, _, _ = load_recent_bars_from_db(
                    "600519", 30, target_date=None
                )
                bars_via_explicit, _, _ = load_recent_bars_from_db(
                    "600519", 30, target_date=frozen
                )
        finally:
            reset_agent_frozen_target_date(token)

        self.assertEqual(len(bars_via_none), 30)
        self.assertEqual(len(bars_via_explicit), 30)
        latest_date_none = getattr(bars_via_none[-1], "date", None)
        latest_date_explicit = getattr(bars_via_explicit[-1], "date", None)
        # 两条路径在 ContextVar 作用下都应解析到同一交易日 T，端条 bar 日期一致
        self.assertEqual(latest_date_none, frozen)
        self.assertEqual(latest_date_explicit, frozen)


if __name__ == "__main__":
    unittest.main()
