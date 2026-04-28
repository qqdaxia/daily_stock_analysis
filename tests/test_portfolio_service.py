# -*- coding: utf-8 -*-
"""Unit tests for portfolio replay service (P0 PR1 scope)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pandas as pd
from sqlalchemy.exc import OperationalError
from sqlalchemy import select

from src.config import Config
from src.repositories.portfolio_repo import PortfolioBusyError, PortfolioRepository
from src.services.portfolio_service import _AvgState, PortfolioConflictError, PortfolioOversellError, PortfolioService
from src.storage import DatabaseManager, PortfolioDailySnapshot, PortfolioPosition, PortfolioPositionLot, PortfolioTrade


class PortfolioServiceTestCase(unittest.TestCase):
    """Portfolio service replay tests for FIFO/AVG and corporate actions."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_path = Path(self.temp_dir.name) / ".env"
        self.db_path = Path(self.temp_dir.name) / "portfolio_test.db"
        self.env_path.write_text(
            "\n".join(
                [
                    "STOCK_LIST=600519",
                    "GEMINI_API_KEY=test",
                    "ADMIN_AUTH_ENABLED=false",
                    f"DATABASE_PATH={self.db_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        os.environ["ENV_FILE"] = str(self.env_path)
        os.environ["DATABASE_PATH"] = str(self.db_path)
        Config.reset_instance()
        DatabaseManager.reset_instance()

        self.db = DatabaseManager.get_instance()
        self.service = PortfolioService()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        os.environ.pop("ENV_FILE", None)
        os.environ.pop("DATABASE_PATH", None)
        self.temp_dir.cleanup()

    def _save_close(self, symbol: str, on_date: date, close: float) -> None:
        df = pd.DataFrame(
            [
                {
                    "date": on_date,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1.0,
                    "amount": close,
                    "pct_chg": 0.0,
                }
            ]
        )
        self.db.save_daily_data(df, code=symbol, data_source="unit-test")

    def _create_account_with_position(
        self,
        *,
        market: str,
        currency: str,
        symbol: str,
        quantity: float = 10.0,
        price: float = 100.0,
        close: Optional[float] = None,
        close_date: Optional[date] = None,
    ) -> int:
        account = self.service.create_account(name=f"{market}-account", broker="Demo", market=market, base_currency=currency)
        aid = account["id"]
        self.service.record_cash_ledger(
            account_id=aid,
            event_date=date(2026, 1, 1),
            direction="in",
            amount=100000,
            currency=currency,
        )
        self.service.record_trade(
            account_id=aid,
            symbol=symbol,
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=quantity,
            price=price,
            market=market,
            currency=currency,
        )
        if close is not None:
            self._save_close(self.service._normalize_symbol(symbol), close_date or date(2026, 1, 3), close)
        return aid

    def test_snapshot_fifo_vs_avg_on_partial_sell(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.record_cash_ledger(
            account_id=aid,
            event_date=date(2026, 1, 1),
            direction="in",
            amount=100000,
            currency="CNY",
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=100,
            price=10,
            fee=10,
            tax=0,
            market="cn",
            currency="CNY",
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 3),
            side="buy",
            quantity=100,
            price=20,
            fee=10,
            tax=0,
            market="cn",
            currency="CNY",
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 4),
            side="sell",
            quantity=150,
            price=30,
            fee=10,
            tax=5,
            market="cn",
            currency="CNY",
        )
        self._save_close("600519", date(2026, 1, 5), 25)

        fifo = self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 5), cost_method="fifo")
        avg = self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 5), cost_method="avg")

        fifo_acc = fifo["accounts"][0]
        avg_acc = avg["accounts"][0]
        self.assertAlmostEqual(fifo_acc["total_equity"], avg_acc["total_equity"], places=6)

        self.assertAlmostEqual(fifo_acc["realized_pnl"], 2470.0, places=6)
        self.assertAlmostEqual(avg_acc["realized_pnl"], 2220.0, places=6)
        self.assertAlmostEqual(fifo_acc["unrealized_pnl"], 245.0, places=6)
        self.assertAlmostEqual(avg_acc["unrealized_pnl"], 495.0, places=6)

        self.assertEqual(len(fifo_acc["positions"]), 1)
        self.assertEqual(len(avg_acc["positions"]), 1)
        self.assertAlmostEqual(fifo_acc["positions"][0]["quantity"], 50.0, places=6)
        self.assertAlmostEqual(avg_acc["positions"][0]["quantity"], 50.0, places=6)

    def test_snapshot_position_price_metadata_uses_backend_values_for_cn_hk_us(self) -> None:
        for market, currency, symbol, close, expected_symbol in [
            ("cn", "CNY", "600519", 12.5, "600519"),
            ("hk", "HKD", "hk700", 420.0, "HK00700"),
            ("us", "USD", "aapl", 210.0, "AAPL"),
        ]:
            with self.subTest(market=market):
                aid = self._create_account_with_position(market=market, currency=currency, symbol=symbol, close=close)
                position = self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 3), cost_method="fifo")["accounts"][0]["positions"][0]

                self.assertEqual(position["symbol"], expected_symbol)
                self.assertEqual(position["price_source"], "recent_close")
                self.assertEqual(position["price_date"], "2026-01-03")
                self.assertFalse(position["is_stale"])
                self.assertFalse(position["is_fallback"])
                self.assertAlmostEqual(position["last_price"], close, places=6)
                self.assertAlmostEqual(position["market_value_base"], close * 10, places=6)
                self.assertAlmostEqual(position["unrealized_pnl_base"], close * 10 - 1000, places=6)
                self.assertAlmostEqual(position["unrealized_pnl_pct"], (close * 10 - 1000) / 1000 * 100, places=6)

    def test_snapshot_marks_stale_close_and_avg_cost_fallback_price(self) -> None:
        aid = self._create_account_with_position(
            market="cn",
            currency="CNY",
            symbol="600519",
            close=110,
            close_date=date(2026, 1, 2),
        )
        self.service.record_trade(
            account_id=aid,
            symbol="000001",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=5,
            price=20,
            market="cn",
            currency="CNY",
        )
        self._save_close("600519", date(2026, 1, 2), 110)

        snapshot = self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 3), cost_method="fifo")
        positions = {item["symbol"]: item for item in snapshot["accounts"][0]["positions"]}

        stale_close = positions["600519"]
        self.assertEqual(stale_close["price_source"], "recent_close")
        self.assertEqual(stale_close["price_date"], "2026-01-02")
        self.assertTrue(stale_close["is_stale"])
        self.assertFalse(stale_close["is_fallback"])
        self.assertAlmostEqual(stale_close["last_price"], 110.0, places=6)
        self.assertAlmostEqual(stale_close["unrealized_pnl_pct"], 10.0, places=6)

        fallback = positions["000001"]
        self.assertEqual(fallback["price_source"], "avg_cost_fallback")
        self.assertIsNone(fallback["price_date"])
        self.assertTrue(fallback["is_stale"])
        self.assertTrue(fallback["is_fallback"])
        self.assertAlmostEqual(fallback["last_price"], 20.0, places=6)
        self.assertAlmostEqual(fallback["market_value_base"], 100.0, places=6)
        self.assertAlmostEqual(fallback["unrealized_pnl_base"], 0.0, places=6)
        self.assertAlmostEqual(fallback["unrealized_pnl_pct"], 0.0, places=6)

    def test_build_positions_handles_zero_cost_without_division(self) -> None:
        account = SimpleNamespace(base_currency="CNY")

        positions, _, _, _, _ = self.service._build_positions(
            account=account,
            as_of_date=date(2026, 1, 3),
            cost_method="avg",
            fifo_lots={},
            avg_state={("AAPL", "us", "USD"): _AvgState(quantity=10.0, total_cost=0.0)},
        )

        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["price_source"], "avg_cost_fallback")
        self.assertIsNone(positions[0]["unrealized_pnl_pct"])
        self.assertAlmostEqual(positions[0]["last_price"], 0.0, places=6)

    def test_corporate_actions_dividend_and_split(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.record_cash_ledger(
            account_id=aid,
            event_date=date(2026, 1, 1),
            direction="in",
            amount=10000,
            currency="CNY",
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=100,
            price=10,
            fee=0,
            tax=0,
            market="cn",
            currency="CNY",
        )
        self.service.record_corporate_action(
            account_id=aid,
            symbol="600519",
            effective_date=date(2026, 1, 3),
            action_type="cash_dividend",
            market="cn",
            currency="CNY",
            cash_dividend_per_share=1.0,
        )
        self.service.record_corporate_action(
            account_id=aid,
            symbol="600519",
            effective_date=date(2026, 1, 4),
            action_type="split_adjustment",
            market="cn",
            currency="CNY",
            split_ratio=2.0,
        )
        self._save_close("600519", date(2026, 1, 5), 6.0)

        snapshot = self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 5), cost_method="fifo")
        acc = snapshot["accounts"][0]
        pos = acc["positions"][0]

        self.assertAlmostEqual(acc["total_cash"], 9100.0, places=6)
        self.assertAlmostEqual(acc["total_market_value"], 1200.0, places=6)
        self.assertAlmostEqual(acc["total_equity"], 10300.0, places=6)
        self.assertAlmostEqual(pos["quantity"], 200.0, places=6)
        self.assertAlmostEqual(pos["avg_cost"], 5.0, places=6)

    def test_same_day_dividend_processed_before_trade(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.record_cash_ledger(
            account_id=aid,
            event_date=date(2026, 1, 1),
            direction="in",
            amount=2000,
            currency="CNY",
        )
        self.service.record_corporate_action(
            account_id=aid,
            symbol="600519",
            effective_date=date(2026, 1, 2),
            action_type="cash_dividend",
            market="cn",
            currency="CNY",
            cash_dividend_per_share=1.0,
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=100,
            price=10,
            market="cn",
            currency="CNY",
        )
        self._save_close("600519", date(2026, 1, 2), 10.0)

        snapshot = self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 2), cost_method="fifo")
        acc = snapshot["accounts"][0]

        self.assertAlmostEqual(acc["total_cash"], 1000.0, places=6)
        self.assertAlmostEqual(acc["total_market_value"], 1000.0, places=6)
        self.assertAlmostEqual(acc["total_equity"], 2000.0, places=6)

    def test_same_day_split_processed_before_trade(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.record_cash_ledger(
            account_id=aid,
            event_date=date(2026, 1, 1),
            direction="in",
            amount=2000,
            currency="CNY",
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 1),
            side="buy",
            quantity=100,
            price=10,
            market="cn",
            currency="CNY",
        )
        self.service.record_corporate_action(
            account_id=aid,
            symbol="600519",
            effective_date=date(2026, 1, 2),
            action_type="split_adjustment",
            market="cn",
            currency="CNY",
            split_ratio=2.0,
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 2),
            side="sell",
            quantity=100,
            price=6,
            market="cn",
            currency="CNY",
        )
        self._save_close("600519", date(2026, 1, 2), 6.0)

        snapshot = self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 2), cost_method="fifo")
        acc = snapshot["accounts"][0]
        pos = acc["positions"][0]

        self.assertAlmostEqual(acc["realized_pnl"], 100.0, places=6)
        self.assertAlmostEqual(acc["total_cash"], 1600.0, places=6)
        self.assertAlmostEqual(pos["quantity"], 100.0, places=6)
        self.assertAlmostEqual(pos["avg_cost"], 5.0, places=6)

    def test_sell_oversell_rejected_before_write(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=10,
            price=10,
            market="cn",
            currency="CNY",
        )

        with self.assertRaises(PortfolioOversellError):
            self.service.record_trade(
                account_id=aid,
                symbol="600519",
                trade_date=date(2026, 1, 3),
                side="sell",
                quantity=20,
                price=11,
                market="cn",
                currency="CNY",
            )

        trades = self.service.list_trade_events(account_id=aid, page=1, page_size=20)
        self.assertEqual(len(trades["items"]), 1)
        self.assertEqual(trades["items"][0]["side"], "buy")

    def test_duplicate_full_close_sell_keeps_conflict_semantics(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 1),
            side="buy",
            quantity=10,
            price=10,
            market="cn",
            currency="CNY",
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 2),
            side="sell",
            quantity=10,
            price=11,
            market="cn",
            currency="CNY",
            trade_uid="sell-full-close-1",
        )

        with self.assertRaises(PortfolioConflictError) as ctx:
            self.service.record_trade(
                account_id=aid,
                symbol="600519",
                trade_date=date(2026, 1, 2),
                side="sell",
                quantity=10,
                price=11,
                market="cn",
                currency="CNY",
                trade_uid="sell-full-close-1",
            )

        self.assertIn("Duplicate trade_uid", str(ctx.exception))

    def test_backdated_trade_write_invalidates_future_cache(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.record_cash_ledger(
            account_id=aid,
            event_date=date(2026, 1, 1),
            direction="in",
            amount=10000,
            currency="CNY",
        )
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 3),
            side="buy",
            quantity=10,
            price=100,
            market="cn",
            currency="CNY",
        )
        self._save_close("600519", date(2026, 1, 3), 100.0)
        self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 3), cost_method="fifo")

        with self.db.get_session() as session:
            snapshot_count = session.execute(
                select(PortfolioDailySnapshot).where(PortfolioDailySnapshot.account_id == aid)
            ).scalars().all()
            position_count = session.execute(
                select(PortfolioPosition).where(PortfolioPosition.account_id == aid)
            ).scalars().all()
            lot_count = session.execute(
                select(PortfolioPositionLot).where(PortfolioPositionLot.account_id == aid)
            ).scalars().all()
        self.assertEqual(len(snapshot_count), 1)
        self.assertEqual(len(position_count), 1)
        self.assertEqual(len(lot_count), 1)

        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=5,
            price=80,
            market="cn",
            currency="CNY",
        )

        with self.db.get_session() as session:
            snapshot_rows = session.execute(
                select(PortfolioDailySnapshot).where(PortfolioDailySnapshot.account_id == aid)
            ).scalars().all()
            position_rows = session.execute(
                select(PortfolioPosition).where(PortfolioPosition.account_id == aid)
            ).scalars().all()
            lot_rows = session.execute(
                select(PortfolioPositionLot).where(PortfolioPositionLot.account_id == aid)
            ).scalars().all()
        self.assertEqual(len(snapshot_rows), 0)
        self.assertEqual(len(position_rows), 0)
        self.assertEqual(len(lot_rows), 0)

    def test_delete_trade_invalidates_cache_and_removes_source_event(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.record_cash_ledger(
            account_id=aid,
            event_date=date(2026, 1, 1),
            direction="in",
            amount=10000,
            currency="CNY",
        )
        trade = self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=10,
            price=100,
            market="cn",
            currency="CNY",
        )
        self._save_close("600519", date(2026, 1, 2), 100.0)
        self.service.get_portfolio_snapshot(account_id=aid, as_of=date(2026, 1, 2), cost_method="fifo")

        self.assertTrue(self.service.delete_trade_event(trade["id"]))

        with self.db.get_session() as session:
            trade_rows = session.execute(
                select(PortfolioTrade).where(PortfolioTrade.account_id == aid)
            ).scalars().all()
            snapshot_rows = session.execute(
                select(PortfolioDailySnapshot).where(PortfolioDailySnapshot.account_id == aid)
            ).scalars().all()
            lot_rows = session.execute(
                select(PortfolioPositionLot).where(PortfolioPositionLot.account_id == aid)
            ).scalars().all()
        self.assertEqual(len(trade_rows), 0)
        self.assertEqual(len(snapshot_rows), 0)
        self.assertEqual(len(lot_rows), 0)

    def test_concurrent_sell_race_allows_only_one_write(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 1),
            side="buy",
            quantity=10,
            price=10,
            market="cn",
            currency="CNY",
        )

        barrier = threading.Barrier(3)
        results: list[str] = []
        errors: list[Exception] = []

        def _worker(uid: str) -> None:
            svc = PortfolioService()
            barrier.wait()
            try:
                svc.record_trade(
                    account_id=aid,
                    symbol="600519",
                    trade_date=date(2026, 1, 2),
                    side="sell",
                    quantity=10,
                    price=11,
                    market="cn",
                    currency="CNY",
                    trade_uid=uid,
                )
                results.append(uid)
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(f"sell-race-{idx}",), daemon=True)
            for idx in range(2)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], PortfolioOversellError)

        trades = self.service.list_trade_events(account_id=aid, page=1, page_size=20)
        sell_count = sum(1 for item in trades["items"] if item["side"] == "sell")
        self.assertEqual(sell_count, 1)

    def test_concurrent_duplicate_full_close_sell_keeps_conflict_semantics(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]
        self.service.record_trade(
            account_id=aid,
            symbol="600519",
            trade_date=date(2026, 1, 1),
            side="buy",
            quantity=10,
            price=10,
            market="cn",
            currency="CNY",
        )

        barrier = threading.Barrier(3)
        results: list[str] = []
        errors: list[Exception] = []

        def _worker() -> None:
            svc = PortfolioService()
            barrier.wait()
            try:
                svc.record_trade(
                    account_id=aid,
                    symbol="600519",
                    trade_date=date(2026, 1, 2),
                    side="sell",
                    quantity=10,
                    price=11,
                    market="cn",
                    currency="CNY",
                    trade_uid="dup-race-sell-1",
                )
                results.append("ok")
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        threads = [threading.Thread(target=_worker, daemon=True) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], PortfolioConflictError)
        self.assertIn("Duplicate trade_uid", str(errors[0]))

    def test_event_symbol_filters_match_legacy_prefixed_symbols(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.repo.add_trade(
            account_id=aid,
            trade_uid="legacy-prefixed-trade",
            symbol="SH600519",
            market="cn",
            currency="CNY",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=10,
            price=100,
            fee=0,
            tax=0,
        )
        self.service.repo.add_corporate_action(
            account_id=aid,
            symbol="SH600519",
            market="cn",
            currency="CNY",
            effective_date=date(2026, 1, 3),
            action_type="cash_dividend",
            cash_dividend_per_share=1.0,
        )

        trades = self.service.list_trade_events(account_id=aid, symbol="600519", page=1, page_size=20)
        actions = self.service.list_corporate_action_events(account_id=aid, symbol="600519", page=1, page_size=20)

        self.assertEqual(trades["total"], 1)
        self.assertEqual(actions["total"], 1)
        self.assertEqual(trades["items"][0]["symbol"], "SH600519")
        self.assertEqual(actions["items"][0]["symbol"], "SH600519")

    def test_event_symbol_filters_match_legacy_suffix_symbols(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="cn", base_currency="CNY")
        aid = account["id"]

        self.service.repo.add_trade(
            account_id=aid,
            trade_uid="legacy-suffix-trade",
            symbol="600519.SH",
            market="cn",
            currency="CNY",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=10,
            price=100,
            fee=0,
            tax=0,
        )
        self.service.repo.add_corporate_action(
            account_id=aid,
            symbol="600519.SH",
            market="cn",
            currency="CNY",
            effective_date=date(2026, 1, 3),
            action_type="cash_dividend",
            cash_dividend_per_share=1.0,
        )

        trades = self.service.list_trade_events(account_id=aid, symbol="600519", page=1, page_size=20)
        actions = self.service.list_corporate_action_events(account_id=aid, symbol="600519", page=1, page_size=20)

        self.assertEqual(trades["total"], 1)
        self.assertEqual(actions["total"], 1)
        self.assertEqual(trades["items"][0]["symbol"], "600519.SH")
        self.assertEqual(actions["items"][0]["symbol"], "600519.SH")

    def test_event_symbol_filters_match_legacy_hk_variants(self) -> None:
        account = self.service.create_account(name="Main", broker="Demo", market="hk", base_currency="HKD")
        aid = account["id"]

        self.service.repo.add_trade(
            account_id=aid,
            trade_uid="legacy-hk-prefixed-trade",
            symbol="HK700",
            market="hk",
            currency="HKD",
            trade_date=date(2026, 1, 2),
            side="buy",
            quantity=10,
            price=400,
            fee=0,
            tax=0,
        )
        self.service.repo.add_trade(
            account_id=aid,
            trade_uid="legacy-hk-suffix-trade",
            symbol="00700.HK",
            market="hk",
            currency="HKD",
            trade_date=date(2026, 1, 3),
            side="buy",
            quantity=5,
            price=410,
            fee=0,
            tax=0,
        )
        self.service.repo.add_corporate_action(
            account_id=aid,
            symbol="HK700",
            market="hk",
            currency="HKD",
            effective_date=date(2026, 1, 4),
            action_type="cash_dividend",
            cash_dividend_per_share=1.0,
        )
        self.service.repo.add_corporate_action(
            account_id=aid,
            symbol="00700.HK",
            market="hk",
            currency="HKD",
            effective_date=date(2026, 1, 5),
            action_type="cash_dividend",
            cash_dividend_per_share=1.5,
        )

        trades = self.service.list_trade_events(account_id=aid, symbol="HK00700", page=1, page_size=20)
        actions = self.service.list_corporate_action_events(account_id=aid, symbol="HK00700", page=1, page_size=20)

        self.assertEqual(trades["total"], 2)
        self.assertEqual(actions["total"], 2)
        self.assertEqual({item["symbol"] for item in trades["items"]}, {"HK700", "00700.HK"})
        self.assertEqual({item["symbol"] for item in actions["items"]}, {"HK700", "00700.HK"})

    def test_portfolio_write_session_maps_sqlite_locked_error(self) -> None:
        repo = PortfolioRepository(db_manager=self.db)
        session = self.db.get_session()
        stmt_exc = OperationalError(
            "BEGIN IMMEDIATE",
            None,
            sqlite3.OperationalError("database is locked"),
        )

        with patch.object(self.db, "get_session", return_value=session):
            with patch.object(
                session.connection(),
                "exec_driver_sql",
                side_effect=stmt_exc,
            ):
                with self.assertRaises(PortfolioBusyError):
                    with repo.portfolio_write_session():
                        pass


if __name__ == "__main__":
    unittest.main()
